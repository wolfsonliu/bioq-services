"""Alembic migration environment.

Pulls the DB URL from GatewaySettings (GATEWAY_DB_URL) so migrations target the
same database as the running app, and uses the ORM metadata for autogenerate.
"""
from __future__ import annotations

import importlib.util
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make the `server` package importable regardless of layout:
#  - source tree: pyproject maps package `server` -> the service dir itself
#    (package-dir {"server": "."}), so this dir *is* the package — bootstrap it
#    from its __init__.py the way conftest.py does.
#  - image (/opt/gateway): `server/` is a real subdir on the path — a plain
#    import resolves it.
SERVICE_DIR = Path(__file__).resolve().parent.parent
_init = SERVICE_DIR / "__init__.py"
if "server" not in sys.modules:
    if _init.exists():
        spec = importlib.util.spec_from_file_location(
            "server", _init, submodule_search_locations=[str(SERVICE_DIR)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["server"] = module
        spec.loader.exec_module(module)
    else:
        sys.path.insert(0, str(SERVICE_DIR))

from server.db.models import Base  # noqa: E402
from server.settings import GatewaySettings  # noqa: E402

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", GatewaySettings().db_url)
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,   # SQLite-safe ALTERs (batch mode)
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
