#!/usr/bin/env python3
"""Seed a bootstrap user + API key into the gateway-server SQLite DB.

Uses only the Python standard library (no `server` package import), so it runs
on the ECS host directly against the bind-mounted DB file — e.g. with the
default deploy layout (GATEWAY_DATA_DIR=./data):

    python seed_key.py --db ./data/gateway/gateway.db --principal alice

The gateway must have started at least once (so it created the schema via
SQLAlchemy create_all) before seeding. The secret is printed ONCE — store it;
only its sha256 hash is persisted. Authenticate with `-H 'X-API-Key: <secret>'`.
"""

from __future__ import annotations

import argparse
import hashlib
import secrets
import sqlite3
import sys
import uuid
from datetime import datetime, timezone


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tables_exist(con: sqlite3.Connection) -> bool:
    rows = con.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name IN ('users','api_keys')"
    ).fetchall()
    return {r[0] for r in rows} >= {"users", "api_keys"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Seed a gateway-server user + API key.")
    ap.add_argument("--db", required=True, help="Path to gateway.db (bind-mounted).")
    ap.add_argument("--principal", required=True,
                    help="Principal / username — this is the identity jobs are owned by.")
    ap.add_argument("--secret", default=None,
                    help="API key secret (default: generate a random one).")
    ap.add_argument("--key-id", default=None,
                    help="Key id (default: gk_<random>).")
    ap.add_argument("--display-name", default=None)
    args = ap.parse_args(argv)

    secret = args.secret or secrets.token_urlsafe(24)
    key_id = args.key_id or f"gk_{uuid.uuid4().hex[:12]}"
    secret_hash = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    now = _utcnow_iso()

    con = sqlite3.connect(args.db)
    try:
        if not _tables_exist(con):
            print(
                f"error: tables not found in {args.db!r}; start the gateway once "
                "so it creates the schema, then re-run.",
                file=sys.stderr,
            )
            return 2
        with con:
            con.execute(
                "INSERT OR IGNORE INTO users(principal, display_name, status, created_at) "
                "VALUES (?,?,?,?)",
                (args.principal, args.display_name, "active", now),
            )
            con.execute(
                "INSERT INTO api_keys"
                "(key_id, principal, secret_hash, status, created_at, last_used_at) "
                "VALUES (?,?,?,?,?,?)",
                (key_id, args.principal, secret_hash, "active", now, None),
            )
    except sqlite3.IntegrityError as exc:
        print(f"error: {exc} (key_id {key_id!r} may already exist)", file=sys.stderr)
        return 1
    finally:
        con.close()

    print("Seeded API key (store the secret now — it is not recoverable):")
    print(f"  principal : {args.principal}")
    print(f"  key_id    : {key_id}")
    print(f"  secret    : {secret}")
    print()
    print("Authenticate with:  -H 'X-API-Key: <secret>'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
