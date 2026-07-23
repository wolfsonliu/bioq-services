#!/usr/bin/env bash
# Container entrypoint: bring the DB schema up to date, then start the server.
# Idempotent — `alembic upgrade head` is a no-op when already current, so it is
# safe on every (re)start. GATEWAY_DB_URL drives both alembic (via env.py) and
# the app. For SQLite the /data dir must exist first.
set -euo pipefail

mkdir -p /data/gateway /data/gateway_jobs 2>/dev/null || true

echo "[entrypoint] alembic upgrade head"
alembic upgrade head

echo "[entrypoint] starting uvicorn"
exec python -m uvicorn server.app:app \
    --host 0.0.0.0 --port "${PORT:-9000}" --timeout-keep-alive 900
