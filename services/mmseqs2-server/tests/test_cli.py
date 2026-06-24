"""CLI batch-mode tests for mmseqs2-server.

The ColabFold protocol is HTTP-only, so there are no CLI-side endpoints to
exercise here — the only thing ``python -m server`` does is invoke
``uvicorn.run`` with the app factory output. We pin that behaviour + the
``PORT`` env var override. Covers Task 4.2 of the Stage 4 plan.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SERVICE_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point env vars at tmp_path so importing ``server.app`` doesn't try to
    mkdir(/data/...) on the dev box.

    Autouse because every test in this file ends up importing ``server.app``
    transitively (``server.__main__`` does ``from .app import app``).
    """
    monkeypatch.setenv("MMSEQS2_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("MMSEQS2_DB_DIR", str(tmp_path / "db"))
    monkeypatch.setenv("MMSEQS2_KEEPALIVE_INTERVAL_S", "0")
    (tmp_path / "db").mkdir(parents=True, exist_ok=True)


def _fresh_import_main():
    """Re-import ``server.__main__`` against a clean ``server`` module table.

    Mirrors the pattern in test_app.py: pop everything under ``server.*``,
    re-register the package spec, then ``import server.__main__``.
    """
    for mod in [m for m in sys.modules if m == "server" or m.startswith("server.")]:
        sys.modules.pop(mod, None)
    spec = importlib.util.spec_from_file_location(
        "server",
        SERVICE_DIR / "__init__.py",
        submodule_search_locations=[str(SERVICE_DIR)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["server"] = module
    spec.loader.exec_module(module)
    return importlib.import_module("server.__main__")


def test_main_module_starts_uvicorn() -> None:
    """``python -m server`` calls ``uvicorn.run(app, host="0.0.0.0", port=...)``.

    We don't care about the precise PORT value here — that's covered by the
    next test. The point is to lock in *that* uvicorn is invoked at all, with
    the right host + the imported FastAPI app.
    """
    main_module = _fresh_import_main()
    with patch.object(main_module, "uvicorn") as mock_uvicorn:
        main_module.main()

    assert mock_uvicorn.run.call_count == 1
    args, kwargs = mock_uvicorn.run.call_args
    assert args[0] is main_module.app
    assert kwargs["host"] == "0.0.0.0"
    assert isinstance(kwargs["port"], int)


def test_port_env_var_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``PORT`` env var overrides the default uvicorn port."""
    monkeypatch.setenv("PORT", "12345")
    main_module = _fresh_import_main()
    with patch.object(main_module, "uvicorn") as mock_uvicorn:
        main_module.main()

    assert mock_uvicorn.run.call_args.kwargs["port"] == 12345


def test_port_defaults_to_9000(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``PORT`` set, the server binds the FC default (9000)."""
    monkeypatch.delenv("PORT", raising=False)
    main_module = _fresh_import_main()
    with patch.object(main_module, "uvicorn") as mock_uvicorn:
        main_module.main()

    assert mock_uvicorn.run.call_args.kwargs["port"] == 9000
