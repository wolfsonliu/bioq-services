"""CLI batch-mode tests for boltz-server.

Tests endpoint registration, build_argv callbacks, and end-to-end create_cli.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic_settings import SettingsConfigDict

from bioq_service.cli import CLIEndpoint, create_cli

from server.adapter import BoltzAdapter
from server.models import PredictAffinityRequest, PredictStructureRequest, SequenceEntry
from server.settings import BoltzSettings
from server.tools import build_yaml, predict_argv


class _Off(BoltzSettings):
    model_config = SettingsConfigDict(env_prefix="BOLTZ_TEST_", env_file=None, extra="ignore")


def _predict_build(req, inputs, job_dir, settings):
    yaml_path = inputs.get("raw_yaml")
    if yaml_path is not None:
        dest = job_dir / "input" / "input.yaml"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(yaml_path, dest)
        req.raw_yaml = dest.read_text(encoding="utf-8")

    yaml_path = build_yaml(
        req,
        job_dir=job_dir,
        settings=settings,
        saved_msa_paths={},
        saved_template_paths={},
    )
    return predict_argv(req, job_dir=job_dir, yaml_path=yaml_path, settings=settings)


ENDPOINTS = {
    "predict_structure": CLIEndpoint(
        name="predict_structure",
        help="Predict 3D structure of a biomolecular complex",
        request_model=PredictStructureRequest,
        build_argv=_predict_build,
        inputs={"raw_yaml": ("Pre-built Boltz YAML input file", False)},
    ),
    "predict_affinity": CLIEndpoint(
        name="predict_affinity",
        help="Predict structure + ligand binding affinity",
        request_model=PredictAffinityRequest,
        build_argv=_predict_build,
        inputs={"raw_yaml": ("Pre-built Boltz YAML input file", False)},
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"predict_structure", "predict_affinity"}


def test_predict_structure_fields():
    ep = ENDPOINTS["predict_structure"]
    assert ep.request_model is PredictStructureRequest
    assert ep.inputs["raw_yaml"][1] is False  # optional


def test_predict_affinity_fields():
    ep = ENDPOINTS["predict_affinity"]
    assert ep.request_model is PredictAffinityRequest
    assert ep.inputs["raw_yaml"][1] is False


# ---- Build_argv callbacks ----


def test_predict_structure_build_argv(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs", root=tmp_path, cache_dir=tmp_path / "cache")
    job_dir = tmp_path / "j"
    job_dir.mkdir(parents=True)

    req = PredictStructureRequest(
        msa_mode="empty",
        sequences=[
            SequenceEntry(type="protein", id="A", sequence="MKT", msa_uri="empty"),
        ],
    )
    argv = _predict_build(req, {}, job_dir, s)
    assert "--model" in argv
    assert argv[argv.index("--model") + 1] == "boltz2"
    assert "--accelerator" in argv


def test_predict_affinity_build_argv(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs", root=tmp_path, cache_dir=tmp_path / "cache")
    job_dir = tmp_path / "j"
    job_dir.mkdir(parents=True)

    req = PredictAffinityRequest(
        binder_id="B",
        msa_mode="empty",
        sequences=[
            SequenceEntry(type="protein", id="A", sequence="MKT", msa_uri="empty"),
            SequenceEntry(type="ligand", id="B", smiles="CCO"),
        ],
    )
    argv = _predict_build(req, {}, job_dir, s)
    assert "--model" in argv

    yaml_path = job_dir / "input" / "input.yaml"
    doc = yaml.safe_load(yaml_path.read_text())
    assert doc["properties"][0]["affinity"]["binder"] == "B"


def test_predict_with_raw_yaml(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs", root=tmp_path, cache_dir=tmp_path / "cache")
    job_dir = tmp_path / "j"
    job_dir.mkdir(parents=True)

    raw_yaml = tmp_path / "input.yaml"
    raw_yaml.write_text("version: 1\nsequences:\n  - protein:\n      id: X\n      sequence: MKT\n")

    req = PredictStructureRequest(raw_yaml=raw_yaml.read_text())
    argv = _predict_build(req, {"raw_yaml": raw_yaml}, job_dir, s)
    assert "--model" in argv


# ---- End-to-end create_cli ----


def test_cli_predict_structure_success(tmp_path, capsys):
    s = _Off(
        jobs_base_dir=tmp_path / "jobs", root=tmp_path,
        binary="/bin/true", cache_dir=tmp_path / "cache",
    )
    adapter = BoltzAdapter(settings=s)
    output_dir = tmp_path / "run" / "output"

    params = json.dumps({
        "msa_mode": "empty",
        "sequences": [{"type": "protein", "id": "A", "sequence": "MKT", "msa_uri": "empty"}],
    })

    with patch.object(sys, "argv", [
        "prog", "predict_structure",
        "--params-json", params,
        "--json", "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"


def test_cli_no_subcommand_exits_2(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs", root=tmp_path, cache_dir=tmp_path / "cache")
    adapter = BoltzAdapter(settings=s)

    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)
