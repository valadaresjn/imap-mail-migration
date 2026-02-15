# IMAP Mailbox Migration

A simple, configurable Python tool to copy mailboxes from one IMAP account to another over SSL/TLS. It preserves flags and internal dates and includes retries, reconnects, progress, and ETA.

**What it does**
- Copies messages via IMAP `APPEND`
- Preserves flags and INTERNALDATE
- Supports SSL/TLS and STARTTLS
- Retries with exponential backoff
- Reconnects on transport errors
- Shows progress, elapsed time, and ETA

## Requirements
- Python 3.8+ (stdlib only)

## Setup
1. Copy the sample config and fill in credentials.
2. Run the script.

```bash
cp config.sample.ini config.ini
./migrate_imap.py --config config.ini
```

## Configuration
Example:
```ini
[source]
host = imap.email.com
port = 993
username = joao@email.com
password = YOUR_SOURCE_PASSWORD
use_ssl = true
use_starttls = false

[destination]
host = imap.novoemail.com
port = 993
username = moacir@novoemail.com
password = YOUR_DEST_PASSWORD
use_ssl = true
use_starttls = false

[migration]
mailboxes =
batch_size = 200
sleep_between_batches = 0
sleep_per_message = 0
keepalive_every = 0
socket_timeout = 60
start_from = 1

[batch]
csv_path = accounts.csv
```

Tip: keep `config.ini` out of git. Store credentials locally.

## Usage
Basic:
```bash
./migrate_imap.py --config config.ini
```

Migrate specific folders:
```bash
./migrate_imap.py --config config.ini --mailboxes "INBOX,Itens Enviados"
```

Resume from a specific position (1-based):
```bash
./migrate_imap.py --config config.ini --mailboxes "INBOX" --start-from 476
```

List folders (useful for localized names):
```bash
./migrate_imap.py --config config.ini --list-mailboxes
./migrate_imap.py --config config.ini --list-mailboxes-raw
```

## Batch Mode (CSV)
You can migrate multiple account pairs using a CSV file. The script will automatically copy **all mailboxes** when the `mailboxes` setting is empty in `config.ini`.

Create a CSV (see `accounts.sample.csv`):
```csv
src_username,src_password,dst_username,dst_password
joao@email.com,SOURCE_PASSWORD,moacir@novoemail.com,DEST_PASSWORD
```

Run batch:
```bash
./migrate_imap.py --config config.ini --batch-csv accounts.csv
```

Notes:
- Server settings always come from `config.ini` `[source]` and `[destination]`.
- The CSV only contains account credentials.

## Common tuning options
Use these when connections drop or the server throttles:
- `--batch-size 20`
- `--sleep-per-message 0.3`
- `--keepalive-every 50`
- `--socket-timeout 120`

## Notes
- This tool does not de-duplicate. If you re-run without `start_from`, duplicates may occur.
- If the server enforces `APPENDLIMIT`, oversized messages are skipped with an explicit error.
- Folder names with spaces or non-ASCII characters are supported, but you must use the exact name shown by `--list-mailboxes-raw`.

## Troubleshooting
- If a folder fails to select, run `--list-mailboxes-raw` and use the exact mailbox name shown.
- If `APPEND` fails repeatedly, lower `batch_size`, increase `sleep_per_message`, and raise `socket_timeout`.

## License
See `LICENSE`.
