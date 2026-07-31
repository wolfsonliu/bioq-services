from __future__ import annotations

import importlib.util
from pathlib import Path

from server.db.store import GatewayDB

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "seed_key.py"


def _load_seed_key():
    spec = importlib.util.spec_from_file_location("seed_key", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_seed_sqlite_admin(tmp_path):
    dbfile = tmp_path / "gw.db"
    GatewayDB(f"sqlite:///{dbfile}").create_all()
    sk = _load_seed_key()
    rc = sk.main(["--db", str(dbfile), "--account-id", "root",
                  "--admin", "--secret", "x", "--key-id", "gk_root"])
    assert rc == 0
    assert GatewayDB(f"sqlite:///{dbfile}").get_user("root").role == "admin"


def test_seed_sqlite_default_user_role(tmp_path):
    dbfile = tmp_path / "gw.db"
    GatewayDB(f"sqlite:///{dbfile}").create_all()
    sk = _load_seed_key()
    rc = sk.main(["--db", str(dbfile), "--account-id", "alice",
                  "--secret", "y", "--key-id", "gk_alice"])
    assert rc == 0
    assert GatewayDB(f"sqlite:///{dbfile}").get_user("alice").role == "user"
