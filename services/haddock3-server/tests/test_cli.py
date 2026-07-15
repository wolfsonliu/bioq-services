"""CLI batch-mode tests for haddock3-server.

Covers endpoint registration, build_argv callbacks, and end-to-end create_cli
(with the SubprocessRunner + output detection mocked, so no real haddock3/CNS).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from bioagent_service.cli import CLIEndpoint, create_cli
from server.adapter import Haddock3Adapter
from server.models import (
    ActpassToAmbigRequest,
    RestrainBodiesRequest,
    ScoreRequest,
)
from server.settings import Haddock3Settings
from server.tools import (
    actpass_to_ambig_argv,
    restrain_bodies_argv,
    score_argv,
)

DATA_DIR = Path(__file__).resolve().parent / "data"


class _Off(Haddock3Settings):
    model_config = SettingsConfigDict(
        env_prefix="HADDOCK3_TEST_", env_file=None, extra="ignore",
    )


def _restrain_build(req, inputs, job_dir, s):
    return restrain_bodies_argv(req, pdb=inputs["structure"], job_dir=job_dir, settings=s)


ENDPOINTS = {
    "restrain-bodies": CLIEndpoint(
        name="restrain-bodies",
        help="CNS-free body restraints",
        request_model=RestrainBodiesRequest,
        build_argv=_restrain_build,
        inputs={"structure": ("Multi-chain PDB", True)},
    ),
}


# ---- Build_argv callbacks ----


def test_score_build_argv(tmp_path):
    s = _Off()
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    pdb = tmp_path / "c.pdb"
    pdb.write_text("ATOM")
    argv = score_argv(
        ScoreRequest(full=True, params={"nemsteps": "50"}),
        pdb=pdb, job_dir=job_dir, settings=s,
    )
    assert argv[0] == s.python
    assert s.inference_script in argv
    assert "score" in argv
    assert "--pdb" in argv and str(pdb) in argv
    assert "--output-dir" in argv and str(job_dir / "output") in argv
    assert "--full" in argv
    assert argv[argv.index("-p") + 1:argv.index("-p") + 3] == ["nemsteps", "50"]


def test_restrain_bodies_build_argv(tmp_path):
    s = _Off()
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    pdb = tmp_path / "c.pdb"
    pdb.write_text("ATOM")
    argv = restrain_bodies_argv(
        RestrainBodiesRequest(exclude="A"), pdb=pdb, job_dir=job_dir, settings=s,
    )
    assert "restrain-bodies" in argv
    assert argv[argv.index("--exclude") + 1] == "A"


def test_actpass_build_argv(tmp_path):
    s = _Off()
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    a1 = tmp_path / "a.actpass"
    a1.write_text("1\n2\n")
    a2 = tmp_path / "b.actpass"
    a2.write_text("3\n4\n")
    argv = actpass_to_ambig_argv(
        ActpassToAmbigRequest(segid1="X", segid2="Y"),
        actpass1=a1, actpass2=a2, job_dir=job_dir, settings=s,
    )
    assert "actpass-to-ambig" in argv
    assert argv[argv.index("--segid1") + 1] == "X"
    assert argv[argv.index("--segid2") + 1] == "Y"


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"restrain-bodies"}
    assert ENDPOINTS["restrain-bodies"].inputs["structure"][1] is True


# ---- End-to-end create_cli ----


def test_cli_restrain_bodies_success(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = Haddock3Adapter(settings=s)
    pdb = DATA_DIR / "complex.pdb"
    out = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "restrain-bodies", "--structure", str(pdb), "--output-dir", str(out),
    ]):
        with patch("bioagent_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc.value.code == 0


def test_cli_json_output(tmp_path, capsys):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = Haddock3Adapter(settings=s)
    pdb = DATA_DIR / "complex.pdb"
    out = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "restrain-bodies", "--structure", str(pdb),
        "--json", "--output-dir", str(out),
    ]):
        with patch("bioagent_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc.value.code == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"
    assert result["return_code"] == 0


def test_cli_no_subcommand_exits_2(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = Haddock3Adapter(settings=s)
    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)
