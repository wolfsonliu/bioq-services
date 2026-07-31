#!/usr/bin/env python3
"""Seed a bootstrap user + API key into the gateway-server DB.

Two modes:

* SQLite file (host, stdlib-only) — run on the ECS host against the
  bind-mounted DB file, no `server` package / venv needed:

      python seed_key.py --db ./data/gateway/gateway.db --account-id alice

* Any DB URL (SQLAlchemy) — for the bundled/managed **PostgreSQL**, run INSIDE
  the gateway container (which has SQLAlchemy + psycopg + the models). The
  container already has GATEWAY_DB_URL set, so --db-url can be omitted:

      docker compose exec gateway python scripts/seed_key.py --account-id alice

The schema must already exist (the entrypoint runs `alembic upgrade head` on
start) before seeding. The secret is printed ONCE — store it; only its sha256
hash is persisted. Authenticate with `-H 'X-API-Key: <secret>'`.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tables_exist(con: sqlite3.Connection) -> bool:
    rows = con.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name IN ('users','api_keys')"
    ).fetchall()
    return {r[0] for r in rows} >= {"users", "api_keys"}


def _seed_sqlite(db_path: str, *, account_id: str, secret: str, key_id: str,
                 display_name: str | None, role: str = "user") -> int:
    secret_hash = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    now = _utcnow_iso()
    con = sqlite3.connect(db_path)
    try:
        if not _tables_exist(con):
            print(
                f"error: tables not found in {db_path!r}; start the gateway once "
                "so it runs migrations (alembic upgrade head), then re-run.",
                file=sys.stderr,
            )
            return 2
        with con:
            con.execute(
                "INSERT OR IGNORE INTO users(account_id, display_name, status, role, created_at) "
                "VALUES (?,?,?,?,?)",
                (account_id, display_name, "active", role, now),
            )
            if role == "admin":
                # 账户已存在时 INSERT OR IGNORE 不生效，显式提升。
                con.execute("UPDATE users SET role='admin' WHERE account_id=?", (account_id,))
            con.execute(
                "INSERT INTO api_keys"
                "(key_id, account_id, secret_hash, status, created_at, last_used_at) "
                "VALUES (?,?,?,?,?,?)",
                (key_id, account_id, secret_hash, "active", now, None),
            )
    except sqlite3.IntegrityError as exc:
        print(f"error: {exc} (key_id {key_id!r} may already exist)", file=sys.stderr)
        return 1
    finally:
        con.close()
    return 0


def _ensure_server_importable() -> None:
    # Running `python scripts/seed_key.py` puts scripts/ on sys.path, not the dir
    # holding the `server` package. Make `server` importable for both layouts:
    #  - image (/opt/gateway): `server/` is a sibling subdir → add parent to path.
    #  - source tree: pyproject maps package `server` -> the service dir itself
    #    (package-dir {"server": "."}), so bootstrap it from its __init__.py.
    import importlib.util

    if "server" in sys.modules:
        return
    service_dir = Path(__file__).resolve().parent.parent
    init = service_dir / "__init__.py"
    if init.exists():
        spec = importlib.util.spec_from_file_location(
            "server", init, submodule_search_locations=[str(service_dir)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["server"] = module
        spec.loader.exec_module(module)
    else:
        sys.path.insert(0, str(service_dir))


def _seed_via_orm(db_url: str, *, account_id: str, secret: str, key_id: str,
                  display_name: str | None, role: str = "user") -> int:
    # ORM path (postgres/any backend); needs the `server` package + SQLAlchemy,
    # so run it inside the gateway container. Imported lazily so the SQLite
    # host path above stays stdlib-only.
    _ensure_server_importable()
    from sqlalchemy.exc import IntegrityError, OperationalError

    from server.db.store import GatewayDB

    db = GatewayDB(db_url)
    try:
        db.create_user(account_id, display_name, role=role)
        if role == "admin":
            db.set_role(account_id, "admin")   # 提升已存在账户
        db.create_api_key(account_id, secret=secret, key_id=key_id)
    except IntegrityError as exc:
        print(f"error: {exc.orig} (key_id {key_id!r} may already exist)", file=sys.stderr)
        return 1
    except OperationalError as exc:
        print(f"error: cannot reach DB or schema missing ({exc.orig}); ensure the "
              "gateway started and ran migrations, then re-run.", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Seed a gateway-server user + API key.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--db", help="Path to a SQLite gateway.db file (host mode).")
    src.add_argument("--db-url", help="SQLAlchemy DB URL (ORM mode; default: "
                     "$GATEWAY_DB_URL). Use inside the gateway container for postgres.")
    ap.add_argument("--account-id", required=True,
                    help="Account id / username — this is the identity jobs are owned by.")
    ap.add_argument("--secret", default=None,
                    help="API key secret (default: generate a random one).")
    ap.add_argument("--key-id", default=None,
                    help="Key id (default: gk_<random>).")
    ap.add_argument("--display-name", default=None)
    ap.add_argument("--admin", action="store_true",
                    help="Grant the account the admin role (management console access).")
    args = ap.parse_args(argv)

    secret = args.secret or secrets.token_urlsafe(24)
    key_id = args.key_id or f"gk_{uuid.uuid4().hex[:12]}"
    role = "admin" if args.admin else "user"

    db_url = args.db_url or (None if args.db else os.environ.get("GATEWAY_DB_URL"))
    if args.db:
        rc = _seed_sqlite(args.db, account_id=args.account_id, secret=secret,
                          key_id=key_id, display_name=args.display_name, role=role)
    elif db_url:
        rc = _seed_via_orm(db_url, account_id=args.account_id, secret=secret,
                           key_id=key_id, display_name=args.display_name, role=role)
    else:
        print("error: pass --db <sqlite file>, --db-url <url>, or set GATEWAY_DB_URL.",
              file=sys.stderr)
        return 2
    if rc != 0:
        return rc

    print("Seeded API key (store the secret now — it is not recoverable):")
    print(f"  account_id : {args.account_id}")
    print(f"  key_id     : {key_id}")
    print(f"  secret     : {secret}")
    print(f"  role       : {role}")
    print()
    print("Authenticate with:  -H 'X-API-Key: <secret>'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
