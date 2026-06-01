"""CLI batch-mode tests for genie3-server.

Tests endpoint registration, build_argv callbacks, and end-to-end create_cli.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic_settings import SettingsConfigDict

from bioagent_service.cli import CLIEndpoint, create_cli

from server.adapter import Genie3Adapter
from server.configs import build_binder_config, build_motif_config, build_unconditional_config
from server.datasets import extract_dataset
from server.models import BinderRequest, MotifRequest, UnconditionalRequest
from server.settings import Genie3Settings


class _Off(Genie3Settings):
    model_config = SettingsConfigDict(
        env_prefix="GENIE3_TEST_", env_file=None, extra="ignore", case_sensitive=False,
    )


def _write_yaml(config, job_dir):
    path = job_dir / "input" / "experiment.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _genie3_argv(config_path, num_devices, settings):
    cmd = [settings.bin, "generate", "-c", str(config_path)]
    if num_devices is not None:
        cmd.extend(["--num-devices", str(num_devices)])
    return cmd


def _unconditional_build(req, inputs, job_dir, settings):
    config = build_unconditional_config(rootdir=job_dir / "output", req=req)
    config_path = _write_yaml(config, job_dir)
    return _genie3_argv(config_path, req.num_devices, settings)


def _motif_build(req, inputs, job_dir, settings):
    dataset_root = extract_dataset(inputs["dataset"], job_dir / "input" / "dataset")
    config = build_motif_config(rootdir=job_dir / "output", dataset_root=dataset_root, req=req)
    config_path = _write_yaml(config, job_dir)
    return _genie3_argv(config_path, req.num_devices, settings)


def _binder_build(req, inputs, job_dir, settings):
    dataset_root = extract_dataset(inputs["dataset"], job_dir / "input" / "dataset")
    config = build_binder_config(rootdir=job_dir / "output", dataset_root=dataset_root, req=req)
    config_path = _write_yaml(config, job_dir)
    return _genie3_argv(config_path, req.num_devices, settings)


ENDPOINTS = {
    "unconditional": CLIEndpoint(
        name="unconditional",
        help="Unconditional protein backbone generation",
        request_model=UnconditionalRequest,
        build_argv=_unconditional_build,
    ),
    "motif": CLIEndpoint(
        name="motif",
        help="Motif scaffolding (dataset zip with problems/ + motifs/)",
        request_model=MotifRequest,
        build_argv=_motif_build,
        inputs={"dataset": ("Dataset zip file (problems/ + motifs/)", True)},
    ),
    "binder": CLIEndpoint(
        name="binder",
        help="Binder design (dataset zip with problems/ + targets/)",
        request_model=BinderRequest,
        build_argv=_binder_build,
        inputs={"dataset": ("Dataset zip file (problems/ + targets/)", True)},
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"unconditional", "motif", "binder"}


def test_unconditional_has_no_inputs():
    assert ENDPOINTS["unconditional"].inputs == {}
    assert ENDPOINTS["unconditional"].request_model is UnconditionalRequest


def test_motif_input_required():
    assert ENDPOINTS["motif"].inputs["dataset"][1] is True
    assert ENDPOINTS["motif"].request_model is MotifRequest


def test_binder_input_required():
    assert ENDPOINTS["binder"].inputs["dataset"][1] is True
    assert ENDPOINTS["binder"].request_model is BinderRequest


# ---- Build_argv callbacks ----


def test_unconditional_build_argv(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs", root=tmp_path)
    job_dir = tmp_path / "j"
    argv = _unconditional_build(
        UnconditionalRequest(n_sample=5, min_length=80, max_length=120),
        {},
        job_dir,
        s,
    )
    assert argv[0] == s.bin
    assert "generate" in argv
    assert "-c" in argv
    config_path = Path(argv[argv.index("-c") + 1])
    assert config_path.exists()
    config = yaml.safe_load(config_path.read_text())
    assert config["generation"]["dataset"]["n_sample"] == 5


def test_binder_build_argv_with_dataset(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs", root=tmp_path)
    job_dir = tmp_path / "j"

    dataset_zip = tmp_path / "dataset.zip"
    with zipfile.ZipFile(dataset_zip, "w") as zf:
        zf.writestr("problems/p1.yaml", "target_chain: A\n")
        zf.writestr("targets/target.pdb", "ATOM\n")

    argv = _binder_build(
        BinderRequest(n_sample=3),
        {"dataset": dataset_zip},
        job_dir,
        s,
    )
    assert argv[0] == s.bin
    config_path = Path(argv[argv.index("-c") + 1])
    config = yaml.safe_load(config_path.read_text())
    assert config["generation"]["dataset"]["n_sample"] == 3


# ---- End-to-end create_cli ----


def test_cli_unconditional_success(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs", root=tmp_path)
    adapter = Genie3Adapter(settings=s)
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "unconditional",
        "--n-sample", "3",
        "--output-dir", str(output_dir),
    ]):
        with patch("bioagent_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.2.0")
                assert exc_info.value.code == 0


def test_cli_no_subcommand_exits_2(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs", root=tmp_path)
    adapter = Genie3Adapter(settings=s)

    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)
