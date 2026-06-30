"""CLI batch-mode + uvicorn dispatcher tests for mmseqs2-server.

``python -m server`` dispatches:
- no subcommand        → ``uvicorn.run(app, ...)`` (HTTP service)
- ``msa`` / ``pair``   → ``create_cli`` (one-shot synchronous batch run)

Tests cover both branches without requiring the real mmseqs binary.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from bioagent_service.cli import CLIEndpoint

SERVICE_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point env vars at tmp_path so importing ``server.app`` doesn't try to
    mkdir(/data/...) on the dev box.

    Autouse because every test in this file ends up importing ``server.app``
    transitively (the no-subcommand branch does ``from .app import app``).
    """
    monkeypatch.setenv("MMSEQS2_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("MMSEQS2_DB_DIR", str(tmp_path / "db"))
    monkeypatch.setenv("MMSEQS2_KEEPALIVE_INTERVAL_S", "0")
    (tmp_path / "db").mkdir(parents=True, exist_ok=True)


def _fresh_import_main():
    """Re-import ``server.__main__`` against a clean ``server`` module table."""
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


# ---------------------------------------------------------------------------
# HTTP dispatcher branch: no subcommand → uvicorn
# ---------------------------------------------------------------------------


def test_main_no_args_starts_uvicorn() -> None:
    """``python -m server`` (no args) calls ``uvicorn.run(app, host="0.0.0.0", port=...)``."""
    from fastapi import FastAPI

    main_module = _fresh_import_main()
    with patch.object(sys, "argv", ["server"]), patch("uvicorn.run") as mock_run:
        main_module.main()

    assert mock_run.call_count == 1
    args, kwargs = mock_run.call_args
    # First positional arg is the FastAPI app (imported lazily inside `main`).
    assert isinstance(args[0], FastAPI)
    assert kwargs["host"] == "0.0.0.0"
    assert isinstance(kwargs["port"], int)


def test_port_env_var_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORT", "12345")
    main_module = _fresh_import_main()
    with patch.object(sys, "argv", ["server"]), patch("uvicorn.run") as mock_run:
        main_module.main()
    assert mock_run.call_args.kwargs["port"] == 12345


def test_port_defaults_to_9000(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PORT", raising=False)
    main_module = _fresh_import_main()
    with patch.object(sys, "argv", ["server"]), patch("uvicorn.run") as mock_run:
        main_module.main()
    assert mock_run.call_args.kwargs["port"] == 9000


# ---------------------------------------------------------------------------
# CLI batch-mode branch: subcommand → create_cli
# ---------------------------------------------------------------------------

# Shared fixtures: a valid monomer FASTA, a valid paired FASTA.
_MONOMER_FASTA = ">q1\nMKQHKAMIVALIVICITAVVAAL\n"
_PAIRED_FASTA = ">chainA\nMKQHKAM\n>chainB\nLLLLLLL\n"


@pytest.fixture
def main_module():
    """Re-imported __main__ module for CLI tests."""
    return _fresh_import_main()


# ---- Endpoint registration ----


def test_endpoints_registered(main_module) -> None:
    assert set(main_module.endpoints.keys()) == {"msa", "pair"}
    for name in ("msa", "pair"):
        ep = main_module.endpoints[name]
        assert isinstance(ep, CLIEndpoint)
        assert ep.inputs["input_fasta"][1] is True  # required


def test_msa_endpoint_request_model_carries_mode(main_module) -> None:
    ep = main_module.endpoints["msa"]
    assert "mode" in ep.request_model.model_fields


# ---- build_argv: shared shape + paired-mode validation ----


def test_msa_build_argv_writes_query_fasta(
    main_module, tmp_path: Path
) -> None:
    """``_msa_build`` stages the FASTA and returns a colabfold_search_argv list."""
    fasta = tmp_path / "in.fasta"
    fasta.write_text(_MONOMER_FASTA)
    job_dir = tmp_path / "j"
    job_dir.mkdir()

    s = main_module.settings
    argv = main_module._msa_build(
        main_module.MSARequest(mode="env"),
        {"input_fasta": fasta},
        job_dir,
        s,
    )
    assert argv[0] == "python"
    assert argv[1:3] == ["-m", "server.orchestrator"]
    # argv includes the staged query path under job_dir/input/query.fasta.
    staged = job_dir / "input" / "query.fasta"
    assert staged.exists()
    assert str(staged) in argv
    # mode=env → use_env=1
    assert "--use-env" in argv
    idx = argv.index("--use-env")
    assert argv[idx + 1] == "1"


def test_pair_build_argv_rejects_monomer_mode(
    main_module, tmp_path: Path
) -> None:
    """The pair subcommand refuses non-paired modes before subprocess launch."""
    fasta = tmp_path / "in.fasta"
    fasta.write_text(_PAIRED_FASTA)
    job_dir = tmp_path / "j"
    job_dir.mkdir()

    with pytest.raises(ValueError, match="not a paired mode"):
        main_module._pair_build(
            main_module.MSARequest(mode="env"),
            {"input_fasta": fasta},
            job_dir,
            main_module.settings,
        )


def test_msa_build_argv_rejects_paired_mode(
    main_module, tmp_path: Path
) -> None:
    fasta = tmp_path / "in.fasta"
    fasta.write_text(_MONOMER_FASTA)
    job_dir = tmp_path / "j"
    job_dir.mkdir()

    with pytest.raises(ValueError, match="paired"):
        main_module._msa_build(
            main_module.MSARequest(mode="pairgreedy"),
            {"input_fasta": fasta},
            job_dir,
            main_module.settings,
        )


def test_pair_build_argv_rejects_single_chain(
    main_module, tmp_path: Path
) -> None:
    fasta = tmp_path / "in.fasta"
    fasta.write_text(_MONOMER_FASTA)  # one chain, but paired mode
    job_dir = tmp_path / "j"
    job_dir.mkdir()

    with pytest.raises(ValueError, match=">=2 sequences"):
        main_module._pair_build(
            main_module.MSARequest(mode="pairgreedy"),
            {"input_fasta": fasta},
            job_dir,
            main_module.settings,
        )


# ---- End-to-end create_cli ----


def _cli_with_args(main_module, args: list[str]):
    """Run ``main_module.main`` with patched argv + mocked SubprocessRunner."""
    with patch.object(sys, "argv", ["python -m server", *args]):
        with patch("bioagent_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(
                main_module.adapter, "detect_outputs", return_value=True
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main_module.main()
    return exc_info.value.code, mock_runner


def test_cli_msa_success(main_module, tmp_path: Path) -> None:
    """``python -m server msa --input-fasta ... --mode env --output-dir ...`` returns 0."""
    fasta = tmp_path / "q.fasta"
    fasta.write_text(_MONOMER_FASTA)
    out_dir = tmp_path / "run"

    code, mock_runner = _cli_with_args(
        main_module,
        [
            "msa",
            "--input-fasta", str(fasta),
            "--mode", "env",
            "--output-dir", str(out_dir),
        ],
    )
    assert code == 0
    assert mock_runner.run.call_count == 1


def test_cli_pair_success(main_module, tmp_path: Path) -> None:
    fasta = tmp_path / "q.fasta"
    fasta.write_text(_PAIRED_FASTA)
    out_dir = tmp_path / "run"

    code, mock_runner = _cli_with_args(
        main_module,
        [
            "pair",
            "--input-fasta", str(fasta),
            "--mode", "pairgreedy",
            "--output-dir", str(out_dir),
        ],
    )
    assert code == 0
    assert mock_runner.run.call_count == 1


def test_cli_msa_json_output(
    main_module, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fasta = tmp_path / "q.fasta"
    fasta.write_text(_MONOMER_FASTA)
    out_dir = tmp_path / "run"

    code, _ = _cli_with_args(
        main_module,
        [
            "msa",
            "--input-fasta", str(fasta),
            "--mode", "env",
            "--json",
            "--output-dir", str(out_dir),
        ],
    )
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"
    assert result["return_code"] == 0


def test_cli_unknown_subcommand_falls_through_to_uvicorn(main_module) -> None:
    """A typo'd subcommand isn't recognized by ``_has_cli_subcommand`` and the
    dispatcher falls back to HTTP mode rather than firing argparse errors."""
    with patch.object(sys, "argv", ["server", "bogus"]):
        with patch("uvicorn.run") as mock_run:
            main_module.main()
    assert mock_run.call_count == 1
