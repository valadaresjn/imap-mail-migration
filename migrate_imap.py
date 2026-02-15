#!/usr/bin/env python3
"""
IMAP mailbox migration tool.

Copies messages from a source account to a destination account over IMAP with SSL/TLS.
Preserves flags and internal date.
"""

import argparse
import configparser
import csv
import imaplib
import os
import sys
import time
import re
from datetime import datetime
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(key, default)


REPORT_FILE_HANDLE = None


def open_report_file(path: str) -> None:
    global REPORT_FILE_HANDLE
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    REPORT_FILE_HANDLE = open(path, "a", encoding="utf-8")
    ts = datetime.now().isoformat(timespec="seconds")
    REPORT_FILE_HANDLE.write(f"=== Migration run started {ts} ===\n")
    REPORT_FILE_HANDLE.flush()


def close_report_file() -> None:
    global REPORT_FILE_HANDLE
    if REPORT_FILE_HANDLE:
        ts = datetime.now().isoformat(timespec="seconds")
        REPORT_FILE_HANDLE.write(f"=== Migration run ended {ts} ===\n")
        REPORT_FILE_HANDLE.flush()
        REPORT_FILE_HANDLE.close()
        REPORT_FILE_HANDLE = None


def log_message(message: str) -> None:
    print(message)
    if REPORT_FILE_HANDLE:
        ts = datetime.now().isoformat(timespec="seconds")
        REPORT_FILE_HANDLE.write(f"{ts} {message}\n")
        REPORT_FILE_HANDLE.flush()


def retry_call(func, label: str, retries: int = 4, base_delay: float = 0.75):
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
            delay = base_delay * (2 ** (attempt - 1))
            log_message(f"  [WARN] {label} failed (attempt {attempt}/{retries}): {exc} -> retry in {delay:.2f}s")
            time.sleep(delay)
    raise RuntimeError(f"{label} failed after {retries} attempts: {last_exc}")


def format_progress(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[----------------------------] 0%"
    filled = int((done / total) * width)
    filled = min(filled, width)
    bar = "#" * filled + "-" * (width - filled)
    pct = int((done / total) * 100)
    return f"[{bar}] {pct:3d}%"


def format_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_eta(done: int, total: int, elapsed_seconds: float) -> str:
    if done <= 0 or elapsed_seconds <= 0:
        return "--:--:--"
    remaining = total - done
    rate = done / elapsed_seconds
    if rate <= 0:
        return "--:--:--"
    eta_seconds = int(remaining / rate)
    return format_elapsed(eta_seconds)


def normalize_internaldate(value: str) -> Optional[datetime]:
    if not value:
        return None
    # IMAP INTERNALDATE format: "01-Jan-2024 10:00:00 +0000"
    try:
        return datetime.strptime(value, "%d-%b-%Y %H:%M:%S %z")
    except Exception:
        return None


@dataclass
class ImapConfig:
    host: str
    port: int
    username: str
    password: str
    use_ssl: bool
    use_starttls: bool


@dataclass
class MigrationConfig:
    src: ImapConfig
    dst: ImapConfig
    mailboxes: List[str]
    since: Optional[str]
    batch_size: int
    dry_run: bool
    max_messages: Optional[int]
    verify_dest: bool
    mark_source_as_migrated: bool
    migrated_flag: str
    sleep_between_batches: float
    reconnect_on_failure: bool
    sleep_per_message: float
    keepalive_every: int
    socket_timeout: float
    start_from: int


@dataclass
class MigrationSettings:
    mailboxes: List[str]
    since: Optional[str]
    batch_size: int
    dry_run: bool
    max_messages: Optional[int]
    verify_dest: bool
    mark_source_as_migrated: bool
    migrated_flag: str
    sleep_between_batches: float
    reconnect_on_failure: bool
    sleep_per_message: float
    keepalive_every: int
    socket_timeout: float
    start_from: int


def load_config(path: Optional[str]) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()

    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        cfg.read(path)

    return cfg


def get_imap_config(cfg: configparser.ConfigParser, section: str, prefix: str) -> ImapConfig:
    host = cfg.get(section, "host", fallback=_env(f"{prefix}_HOST"))
    port = cfg.getint(section, "port", fallback=int(_env(f"{prefix}_PORT", "993")))
    username = cfg.get(section, "username", fallback=_env(f"{prefix}_USERNAME"))
    password = cfg.get(section, "password", fallback=_env(f"{prefix}_PASSWORD"))
    use_ssl = cfg.getboolean(section, "use_ssl", fallback=_env(f"{prefix}_USE_SSL", "true").lower() == "true")
    use_starttls = cfg.getboolean(
        section, "use_starttls", fallback=_env(f"{prefix}_USE_STARTTLS", "false").lower() == "true"
    )

    missing = [
        k for k, v in {
            "host": host,
            "username": username,
            "password": password,
        }.items() if not v
    ]
    if missing:
        raise ValueError(f"Missing required config in {section}: {', '.join(missing)}")

    return ImapConfig(
        host=host,
        port=port,
        username=username,
        password=password,
        use_ssl=use_ssl,
        use_starttls=use_starttls,
    )


def get_imap_defaults(cfg: configparser.ConfigParser, section: str, prefix: str) -> Dict[str, Optional[str]]:
    if cfg.has_section(section):
        host = cfg.get(section, "host", fallback=_env(f"{prefix}_HOST"))
        port = cfg.getint(section, "port", fallback=int(_env(f"{prefix}_PORT", "993")))
        use_ssl = cfg.getboolean(section, "use_ssl", fallback=_env(f"{prefix}_USE_SSL", "true").lower() == "true")
        use_starttls = cfg.getboolean(
            section, "use_starttls", fallback=_env(f"{prefix}_USE_STARTTLS", "false").lower() == "true"
        )
    else:
        host = _env(f"{prefix}_HOST")
        port = int(_env(f"{prefix}_PORT", "993"))
        use_ssl = _env(f"{prefix}_USE_SSL", "true").lower() == "true"
        use_starttls = _env(f"{prefix}_USE_STARTTLS", "false").lower() == "true"
    return {
        "host": host,
        "port": port,
        "use_ssl": use_ssl,
        "use_starttls": use_starttls,
    }


def parse_bool(value, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    val = str(value).strip().lower()
    if val in {"1", "true", "yes", "y", "t"}:
        return True
    if val in {"0", "false", "no", "n", "f"}:
        return False
    return default


def parse_int(value, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(str(value).strip())
    except Exception:
        return default


def normalize_row(row: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in row.items():
        if k is None:
            continue
        key = str(k).strip().lower()
        if not key:
            continue
        out[key] = v.strip() if isinstance(v, str) else str(v)
    return out


def get_row_value(row: Dict[str, str], *keys: str, default: Optional[str] = None) -> Optional[str]:
    for key in keys:
        if key in row and row[key] != "":
            return row[key]
    return default


def parse_mailboxes_field(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [m.strip() for m in value.split(",") if m.strip()]


def build_imap_config_from_row(
    row: Dict[str, str],
    prefix: str,
    defaults: Dict[str, Optional[str]],
    row_num: int,
) -> ImapConfig:
    host = defaults.get("host")
    port = defaults.get("port")
    use_ssl = defaults.get("use_ssl", True)
    use_starttls = defaults.get("use_starttls", False)
    username = get_row_value(row, f"{prefix}_username", f"{prefix}_user", f"{prefix}_email")
    password = get_row_value(row, f"{prefix}_password", f"{prefix}_pass")

    missing = []
    if not host:
        missing.append("host (from config)")
    if not port:
        missing.append("port (from config)")
    if username is None or username == "":
        missing.append("username")
    if password is None or password == "":
        missing.append("password")
    if missing:
        raise ValueError(f"CSV row {row_num}: missing {prefix} fields: {', '.join(missing)}")

    return ImapConfig(
        host=str(host),
        port=int(port),
        username=str(username),
        password=str(password),
        use_ssl=bool(use_ssl),
        use_starttls=bool(use_starttls),
    )


def build_migration_settings(args: argparse.Namespace, cfg: configparser.ConfigParser) -> MigrationSettings:
    if not cfg.has_section("migration"):
        cfg.add_section("migration")
    mailboxes = []
    if args.mailboxes:
        mailboxes = [m.strip() for m in args.mailboxes.split(",") if m.strip()]
    else:
        raw = cfg.get("migration", "mailboxes", fallback="")
        if raw:
            mailboxes = [m.strip() for m in raw.split(",") if m.strip()]

    since = args.since or cfg.get("migration", "since", fallback=None)
    batch_size = args.batch_size or cfg.getint("migration", "batch_size", fallback=200)
    dry_run = args.dry_run or cfg.getboolean("migration", "dry_run", fallback=False)
    max_messages = args.max_messages or cfg.getint("migration", "max_messages", fallback=0) or None
    verify_dest = args.verify_dest or cfg.getboolean("migration", "verify_dest", fallback=True)
    mark_source_as_migrated = args.mark_source_as_migrated or cfg.getboolean(
        "migration", "mark_source_as_migrated", fallback=False
    )
    migrated_flag = args.migrated_flag or cfg.get("migration", "migrated_flag", fallback="Migrated")
    sleep_between_batches = args.sleep_between_batches or cfg.getfloat(
        "migration", "sleep_between_batches", fallback=0.0
    )
    reconnect_on_failure = args.reconnect_on_failure or cfg.getboolean(
        "migration", "reconnect_on_failure", fallback=True
    )
    sleep_per_message = args.sleep_per_message or cfg.getfloat("migration", "sleep_per_message", fallback=0.0)
    keepalive_every = args.keepalive_every or cfg.getint("migration", "keepalive_every", fallback=0)
    socket_timeout = args.socket_timeout or cfg.getfloat("migration", "socket_timeout", fallback=60.0)
    start_from = args.start_from or cfg.getint("migration", "start_from", fallback=1)
    if start_from < 1:
        start_from = 1

    return MigrationSettings(
        mailboxes=mailboxes,
        since=since,
        batch_size=batch_size,
        dry_run=dry_run,
        max_messages=max_messages,
        verify_dest=verify_dest,
        mark_source_as_migrated=mark_source_as_migrated,
        migrated_flag=migrated_flag,
        sleep_between_batches=sleep_between_batches,
        reconnect_on_failure=reconnect_on_failure,
        sleep_per_message=sleep_per_message,
        keepalive_every=keepalive_every,
        socket_timeout=socket_timeout,
        start_from=start_from,
    )


def build_migration_config(args: argparse.Namespace) -> MigrationConfig:
    cfg = load_config(args.config)
    settings = build_migration_settings(args, cfg)

    src = get_imap_config(cfg, "source", "SRC")
    dst = get_imap_config(cfg, "destination", "DST")

    return MigrationConfig(
        src=src,
        dst=dst,
        **vars(settings),
    )


def connect_imap(cfg: ImapConfig, label: str, socket_timeout: float) -> imaplib.IMAP4:
    def _do_connect() -> imaplib.IMAP4:
        if cfg.use_ssl:
            conn = imaplib.IMAP4_SSL(cfg.host, cfg.port)
        else:
            conn = imaplib.IMAP4(cfg.host, cfg.port)
            if cfg.use_starttls:
                conn.starttls()
        return conn

    conn = retry_call(
        _do_connect,
        label=f"{label} connect",
    )
    try:
        conn.sock.settimeout(socket_timeout)
    except Exception:
        pass

    typ, _ = retry_call(
        lambda: conn.login(cfg.username, cfg.password),
        label=f"{label} login",
    )
    if typ != "OK":
        raise RuntimeError(f"{label} login failed")
    return conn


def list_mailboxes(conn: imaplib.IMAP4) -> List[str]:
    typ, data = retry_call(lambda: conn.list(), label="list mailboxes")
    if typ != "OK":
        raise RuntimeError("Unable to list mailboxes")

    mailboxes: List[str] = []
    for line in data:
        if not line:
            continue
        raw = line.decode(errors="ignore") if isinstance(line, (bytes, bytearray)) else str(line)
        name, _attrs = parse_list_line(raw)
        if name:
            mailboxes.append(name)
    return mailboxes


def list_mailboxes_detailed(conn: imaplib.IMAP4) -> List[Tuple[str, List[str], str]]:
    typ, data = retry_call(lambda: conn.list(), label="list mailboxes")
    if typ != "OK":
        raise RuntimeError("Unable to list mailboxes")

    results: List[Tuple[str, List[str], str]] = []
    for line in data:
        if not line:
            continue
        raw = line.decode(errors="ignore") if isinstance(line, (bytes, bytearray)) else str(line)
        name, attrs = parse_list_line(raw)
        if name:
            results.append((name, attrs, raw))
    return results


def parse_list_line(raw: str) -> Tuple[Optional[str], List[str]]:
    attrs: List[str] = []
    m_attrs = re.search(r"^\((.*?)\)", raw)
    if m_attrs:
        attrs = [a for a in m_attrs.group(1).split() if a]

    # Common LIST format: (<attrs>) "<delimiter>" <mailbox>
    m = re.match(r'^\((?P<attrs>.*?)\)\s+"(?P<delim>.*)"\s+(?P<name>.+)$', raw)
    if m:
        name = m.group("name").strip()
        if name.startswith('"') and name.endswith('"'):
            name = name[1:-1]
        name = name.replace('\\"', '"').replace("\\\\", "\\")
        return name, attrs

    # Fallback: last quoted string
    m_name = re.search(r"\"((?:\\\\.|[^\"])*)\"\\s*$", raw)
    if m_name:
        return m_name.group(1), attrs

    parts = raw.split()
    if parts:
        return parts[-1].strip('"'), attrs
    return None, attrs


def select_mailbox(conn: imaplib.IMAP4, mailbox: str) -> int:
    typ, data = retry_call(
        lambda: conn.select(encode_mailbox(mailbox), readonly=False),
        label=f"select {mailbox}",
    )
    if typ != "OK":
        raise RuntimeError(f"Failed to select mailbox: {mailbox}. Response: {data}")
    return int(data[0]) if data and data[0] else 0


def search_messages(conn: imaplib.IMAP4, since: Optional[str]) -> List[bytes]:
    if since:
        # IMAP date format: 01-Jan-2024
        criteria = f'(SINCE "{since}")'
    else:
        criteria = "ALL"

    typ, data = retry_call(lambda: conn.search(None, criteria), label=f"search {criteria}")
    if typ != "OK":
        raise RuntimeError("Search failed")
    if not data or not data[0]:
        return []
    return data[0].split()


def chunked(items: List[bytes], size: int) -> Iterable[List[bytes]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_message(conn: imaplib.IMAP4, uid: bytes) -> Tuple[bytes, str, str]:
    size, flags, internaldate = fetch_message_meta(conn, uid)
    raw_msg = fetch_message_body(conn, uid)
    return raw_msg, flags, internaldate


def uid_str(uid) -> str:
    if isinstance(uid, bytes):
        return uid.decode(errors="ignore")
    return str(uid)


def uid_bytes(uid) -> bytes:
    if isinstance(uid, bytes):
        return uid
    if isinstance(uid, int):
        return str(uid).encode()
    return str(uid).encode()


def safe_decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="ignore")
    return str(value)


def encode_mailbox(name: str):
    if isinstance(name, bytes):
        return name
    if name.startswith('"') and name.endswith('"'):
        return name
    try:
        name.encode("ascii")
        enc = name
    except Exception:
        enc = imaplib.IMAP4._encode_utf7(name)
    enc = enc.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{enc}"'


def fetch_message_meta(conn: imaplib.IMAP4, uid) -> Tuple[int, str, str]:
    # Fetch RFC822.SIZE, flags, and INTERNALDATE
    typ, data = retry_call(
        lambda: conn.fetch(uid_bytes(uid), "(RFC822.SIZE FLAGS INTERNALDATE)"),
        label=f"fetch meta UID {uid_str(uid)}",
    )
    if typ != "OK" or not data or not data[0]:
        raise RuntimeError(f"Fetch meta failed for UID {uid_str(uid)}")

    meta_part = data[0][0] if isinstance(data[0], tuple) else data[0]
    meta = safe_decode(meta_part)

    flags = ""
    internaldate = ""
    size = 0

    if "RFC822.SIZE" in meta:
        try:
            idx = meta.find("RFC822.SIZE")
            sub = meta[idx:]
            parts = sub.split()
            if len(parts) >= 2:
                size = int(parts[1])
        except Exception:
            size = 0

    if "FLAGS" in meta:
        start = meta.find("FLAGS")
        if start >= 0:
            sub = meta[start:]
            lpar = sub.find("(")
            rpar = sub.find(")")
            if lpar >= 0 and rpar >= 0 and rpar > lpar:
                flags = sub[lpar + 1 : rpar]

    if "INTERNALDATE" in meta:
        start = meta.find("INTERNALDATE")
        if start >= 0:
            sub = meta[start:]
            first_quote = sub.find('"')
            second_quote = sub.find('"', first_quote + 1)
            if first_quote >= 0 and second_quote > first_quote:
                internaldate = sub[first_quote + 1 : second_quote]

    return size, flags, internaldate


def fetch_message_body(conn: imaplib.IMAP4, uid) -> bytes:
    typ, data = retry_call(
        lambda: conn.fetch(uid_bytes(uid), "(RFC822)"),
        label=f"fetch body UID {uid_str(uid)}",
    )
    if typ != "OK" or not data or not data[0]:
        raise RuntimeError(f"Fetch body failed for UID {uid_str(uid)}")
    return data[0][1]


def get_append_limit(conn: imaplib.IMAP4) -> Optional[int]:
    try:
        typ, data = retry_call(lambda: conn.capability(), label="capability")
        if typ != "OK" or not data:
            return None
        caps = data[0].decode(errors="ignore").upper().split()
        for cap in caps:
            if cap.startswith("APPENDLIMIT="):
                return int(cap.split("=", 1)[1])
        return None
    except Exception:
        return None


def keepalive(conn: imaplib.IMAP4, label: str) -> None:
    try:
        retry_call(lambda: conn.noop(), label=f"{label} NOOP")
    except Exception as exc:
        log_message(f"  [WARN] {label} keepalive failed: {exc}")


def append_message(
    conn: imaplib.IMAP4,
    mailbox: str,
    raw_msg: bytes,
    flags: str,
    internaldate: str,
) -> None:
    flag_list = None
    if flags:
        flag_list = f"({flags})"

    date_value = normalize_internaldate(internaldate)
    mailbox_enc = encode_mailbox(mailbox)
    typ, data = retry_call(
        lambda: conn.append(mailbox_enc, flag_list, date_value, raw_msg),
        label=f"append {mailbox}",
    )
    if typ != "OK":
        raise RuntimeError(f"Append failed for mailbox {mailbox}. Response: {data}")


def add_flag(conn: imaplib.IMAP4, uid, flag: str) -> None:
    typ, _ = retry_call(
        lambda: conn.store(uid_bytes(uid), "+FLAGS", f"({flag})"),
        label=f"store flag {flag} for UID {uid_str(uid)}",
    )
    if typ != "OK":
        raise RuntimeError(f"Failed to add flag {flag} for UID {uid_str(uid)}")


def ensure_mailbox_exists(conn: imaplib.IMAP4, mailbox: str) -> None:
    typ, _ = retry_call(lambda: conn.select(encode_mailbox(mailbox)), label=f"select {mailbox}")
    if typ == "OK":
        return
    typ, _ = retry_call(lambda: conn.create(encode_mailbox(mailbox)), label=f"create {mailbox}")
    if typ != "OK":
        raise RuntimeError(f"Failed to create mailbox: {mailbox}")


def is_logout_error(exc: Exception) -> bool:
    return "logout" in str(exc).lower()


def is_transport_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    keys = [
        "socket error",
        "broken pipe",
        "bad write retry",
        "timed out",
        "timeout",
        "connection reset",
        "eof",
        "disconnected",
        "cannot read from timed out object",
    ]
    return any(k in msg for k in keys)


def select_with_reconnect(
    conn: imaplib.IMAP4,
    cfg: ImapConfig,
    label: str,
    mailbox: str,
    socket_timeout: float,
) -> Tuple[imaplib.IMAP4, int]:
    try:
        total = select_mailbox(conn, mailbox)
        return conn, total
    except Exception as exc:
        if is_logout_error(exc):
            log_message(f"  [WARN] {label} connection was logged out. Reconnecting...")
            try:
                conn.logout()
            except Exception:
                pass
            conn = connect_imap(cfg, label, socket_timeout)
            total = select_mailbox(conn, mailbox)
            return conn, total
        raise


def ensure_mailbox_with_reconnect(
    conn: imaplib.IMAP4,
    cfg: ImapConfig,
    label: str,
    mailbox: str,
    socket_timeout: float,
) -> imaplib.IMAP4:
    try:
        ensure_mailbox_exists(conn, mailbox)
        select_mailbox(conn, mailbox)
        return conn
    except Exception as exc:
        if is_logout_error(exc):
            log_message(f"  [WARN] {label} connection was logged out. Reconnecting...")
            try:
                conn.logout()
            except Exception:
                pass
            conn = connect_imap(cfg, label, socket_timeout)
            ensure_mailbox_exists(conn, mailbox)
            select_mailbox(conn, mailbox)
            return conn
        raise


def migrate_mailbox(
    src: imaplib.IMAP4,
    dst: imaplib.IMAP4,
    mailbox: str,
    cfg: MigrationConfig,
) -> Tuple[int, int, imaplib.IMAP4, imaplib.IMAP4]:
    log_message(f"\n==> Migrating mailbox: {mailbox}")
    log_message(f"Source account: {cfg.src.username} -> Destination account: {cfg.dst.username}")

    src, total_src = select_with_reconnect(src, cfg.src, "source", mailbox, cfg.socket_timeout)
    log_message(f"Source messages (reported): {total_src}")

    dst = ensure_mailbox_with_reconnect(dst, cfg.dst, "destination", mailbox, cfg.socket_timeout)
    append_limit = get_append_limit(dst)
    if append_limit:
        log_message(f"Destination APPEND limit: {append_limit} bytes")

    uids = search_messages(src, cfg.since)
    if cfg.max_messages:
        uids = uids[: cfg.max_messages]

    if cfg.start_from > 1:
        if cfg.start_from > len(uids):
            log_message(f"Start_from {cfg.start_from} is beyond total messages {len(uids)}. Nothing to do.")
            return 0, 0
        log_message(f"Resuming from message #{cfg.start_from} (1-based index).")
        uids = uids[cfg.start_from - 1 :]

    if not uids:
        log_message("No messages to migrate.")
        return 0, 0, src, dst

    migrated = 0
    failed = 0

    total = len(uids)
    last_progress_print = 0
    start_time = time.time()
    for batch in chunked(uids, cfg.batch_size):
        for uid in batch:
            try:
                try:
                    size, flags, internaldate = fetch_message_meta(src, uid)
                    if append_limit and size > append_limit:
                        failed += 1
                        log_message(
                            f"  [ERROR] UID {uid_str(uid)} - message size {size} exceeds destination APPEND limit {append_limit}"
                        )
                        continue
                    raw_msg = fetch_message_body(src, uid)
                except Exception as exc:
                    if cfg.reconnect_on_failure and is_transport_error(exc):
                        log_message("  [WARN] fetch failed due to socket error. Reconnecting source...")
                        try:
                            src.logout()
                        except Exception:
                            pass
                        src = connect_imap(cfg.src, "source", cfg.socket_timeout)
                        select_mailbox(src, mailbox)
                        size, flags, internaldate = fetch_message_meta(src, uid)
                        if append_limit and size > append_limit:
                            failed += 1
                            log_message(
                            f"  [ERROR] UID {uid_str(uid)} - message size {size} exceeds destination APPEND limit {append_limit}"
                            )
                            continue
                        raw_msg = fetch_message_body(src, uid)
                    else:
                        raise
                if cfg.dry_run:
                    migrated += 1
                else:
                    try:
                        append_message(dst, mailbox, raw_msg, flags, internaldate)
                        migrated += 1
                    except Exception as exc:
                        if cfg.reconnect_on_failure and is_transport_error(exc):
                            log_message("  [WARN] append failed due to socket error. Reconnecting destination...")
                            try:
                                dst.logout()
                            except Exception:
                                pass
                            dst = connect_imap(cfg.dst, "destination", cfg.socket_timeout)
                            ensure_mailbox_exists(dst, mailbox)
                            select_mailbox(dst, mailbox)
                            append_message(dst, mailbox, raw_msg, flags, internaldate)
                            migrated += 1
                        else:
                            raise

                if cfg.mark_source_as_migrated:
                    add_flag(src, uid, cfg.migrated_flag)

            except Exception as exc:
                failed += 1
                log_message(f"  [ERROR] UID {uid_str(uid)} - {exc}")
            finally:
                done = migrated + failed
                if done - last_progress_print >= 25 or done == total:
                    elapsed_seconds = time.time() - start_time
                    elapsed = format_elapsed(elapsed_seconds)
                    eta = format_eta(done, total, elapsed_seconds)
                    log_message(
                        f"  Progress {format_progress(done, total)} ({done}/{total}) Elapsed {elapsed} ETA {eta}"
                    )
                    last_progress_print = done
                if cfg.keepalive_every and done % cfg.keepalive_every == 0:
                    keepalive(src, "source")
                    keepalive(dst, "destination")
                if cfg.sleep_per_message > 0:
                    time.sleep(cfg.sleep_per_message)

        if cfg.sleep_between_batches > 0:
            time.sleep(cfg.sleep_between_batches)

    if cfg.verify_dest and not cfg.dry_run:
        dest_total = select_mailbox(dst, mailbox)
        log_message(f"Destination messages (reported): {dest_total}")

    return migrated, failed, src, dst


def migrate_account(cfg: MigrationConfig) -> Tuple[int, int]:
    try:
        src_conn = connect_imap(cfg.src, "source", cfg.socket_timeout)
        dst_conn = connect_imap(cfg.dst, "destination", cfg.socket_timeout)
    except Exception as exc:
        log_message(f"Connection error: {exc}")
        return 0, 1

    try:
        if cfg.mailboxes:
            mailboxes = cfg.mailboxes
        else:
            mailboxes = list_mailboxes(src_conn)

        if not mailboxes:
            log_message("No mailboxes found on source.")
            return 0, 0

        log_message(f"Mailboxes to migrate: {', '.join(mailboxes)}")

        total_migrated = 0
        total_failed = 0
        for mailbox in mailboxes:
            migrated, failed, src_conn, dst_conn = migrate_mailbox(src_conn, dst_conn, mailbox, cfg)
            total_migrated += migrated
            total_failed += failed

        return total_migrated, total_failed
    finally:
        try:
            src_conn.logout()
        except Exception:
            pass
        try:
            dst_conn.logout()
        except Exception:
            pass


def load_batch_rows(path: str) -> List[Tuple[int, Dict[str, str]]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    rows: List[Tuple[int, Dict[str, str]]] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        if not reader.fieldnames:
            raise ValueError("CSV file has no header row")
        for idx, row in enumerate(reader, start=2):
            norm = normalize_row(row)
            if not norm or all(v == "" for v in norm.values()):
                continue
            rows.append((idx, norm))
    return rows


def run_batch_migrations(batch_csv: str, settings: MigrationSettings, cfg: configparser.ConfigParser) -> int:
    defaults_src = get_imap_defaults(cfg, "source", "SRC")
    defaults_dst = get_imap_defaults(cfg, "destination", "DST")

    rows = load_batch_rows(batch_csv)
    if not rows:
        log_message("No rows found in CSV.")
        return 0

    total_migrated = 0
    total_failed = 0

    for row_num, row in rows:
        try:
            src = build_imap_config_from_row(row, "src", defaults_src, row_num)
            dst = build_imap_config_from_row(row, "dst", defaults_dst, row_num)

            row_settings = MigrationSettings(**vars(settings))
            # In batch mode, per-row fields are limited to account credentials.
            # All migration options come from config/CLI.

            cfg_row = MigrationConfig(
                src=src,
                dst=dst,
                **vars(row_settings),
            )

            log_message(f"\n=== Batch row {row_num}: {src.username} -> {dst.username} ===")
            migrated, failed = migrate_account(cfg_row)
            total_migrated += migrated
            total_failed += failed
        except Exception as exc:
            total_failed += 1
            log_message(f"[ERROR] Row {row_num}: {exc}")

    log_message(f"\nBatch done. Migrated: {total_migrated}, Failed: {total_failed}")
    return 0 if total_failed == 0 else 1


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IMAP mailbox migration tool")
    parser.add_argument("--config", help="Path to INI config file")
    parser.add_argument("--mailboxes", help="Comma-separated mailboxes to migrate (default: all)")
    parser.add_argument("--since", help='Only migrate messages since date, e.g. "01-Jan-2024"')
    parser.add_argument("--batch-size", type=int, help="Messages per batch")
    parser.add_argument("--max-messages", type=int, help="Limit number of messages")
    parser.add_argument("--dry-run", action="store_true", help="Do not append messages to destination")
    parser.add_argument("--verify-dest", action="store_true", help="Show destination mailbox counts")
    parser.add_argument("--mark-source-as-migrated", action="store_true", help="Add migrated flag on source")
    parser.add_argument("--migrated-flag", help='IMAP flag name to mark on source (default: "Migrated")')
    parser.add_argument("--sleep-between-batches", type=float, help="Seconds to sleep between batches")
    parser.add_argument(
        "--reconnect-on-failure",
        action="store_true",
        help="Reconnect source/destination on socket errors (default: true)",
    )
    parser.add_argument("--sleep-per-message", type=float, help="Seconds to sleep between each message")
    parser.add_argument("--keepalive-every", type=int, help="Send NOOP every N messages (0 = off)")
    parser.add_argument("--socket-timeout", type=float, help="Socket timeout in seconds")
    parser.add_argument("--start-from", type=int, help="Start from message number in search results (1-based)")
    parser.add_argument("--report-file", help="Path to report log file")
    parser.add_argument("--list-mailboxes", action="store_true", help="List mailboxes on source and exit")
    parser.add_argument(
        "--list-mailboxes-raw",
        action="store_true",
        help="List mailboxes with raw IMAP LIST lines (for debugging)",
    )
    parser.add_argument("--batch-csv", help="Path to CSV file for batch migrations")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    cfg_parser = load_config(args.config)
    report_path = args.report_file
    if not report_path and cfg_parser.has_section("report"):
        report_path = cfg_parser.get("report", "path", fallback=None)
    if report_path:
        open_report_file(report_path)

    try:
        try:
            settings = build_migration_settings(args, cfg_parser)
        except Exception as exc:
            log_message(f"Config error: {exc}")
            return 2

        batch_csv = args.batch_csv
        if not batch_csv and cfg_parser.has_section("batch"):
            batch_csv = cfg_parser.get("batch", "csv_path", fallback=None)
        if batch_csv:
            return run_batch_migrations(batch_csv, settings, cfg_parser)

        try:
            src = get_imap_config(cfg_parser, "source", "SRC")
            dst = get_imap_config(cfg_parser, "destination", "DST")
            cfg = MigrationConfig(src=src, dst=dst, **vars(settings))

            if args.list_mailboxes or args.list_mailboxes_raw:
                src_conn = connect_imap(cfg.src, "source", cfg.socket_timeout)
                try:
                    if args.list_mailboxes_raw:
                        boxes = list_mailboxes_detailed(src_conn)
                        log_message("Mailboxes on source (name | attrs | raw):")
                        for name, attrs, raw in boxes:
                            attrs_str = ",".join(attrs) if attrs else "-"
                            log_message(f"  - {name} | {attrs_str} | {raw}")
                    else:
                        boxes = list_mailboxes(src_conn)
                        log_message("Mailboxes on source:")
                        for name in boxes:
                            log_message(f"  - {name}")
                    return 0
                finally:
                    try:
                        src_conn.logout()
                    except Exception:
                        pass

            total_migrated, total_failed = migrate_account(cfg)
            log_message(f"\nDone. Migrated: {total_migrated}, Failed: {total_failed}")
            return 0 if total_failed == 0 else 1
        except KeyboardInterrupt:
            log_message("\nInterrupted by user. Partial migration may have completed.")
            return 130
    finally:
        close_report_file()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
