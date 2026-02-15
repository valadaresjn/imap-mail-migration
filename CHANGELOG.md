# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- Batch CSV mode for multiple account migrations (credentials only in CSV, server settings from config)
- Report log output to a file via `--report-file` or `[report]` config
- Sample CSV template and documentation updates for batch/report usage

## [1.0.0] - 2026-02-15
- Initial IMAP migration tool with SSL/TLS and STARTTLS support
- Message copy with flags and INTERNALDATE preserved
- Retry/backoff and reconnect handling for transport errors
- Progress bar with elapsed time and ETA
- Mailbox listing helpers and resume support (`start_from`)
