"""argv builders must produce the same flags as the legacy `tasks.py` did.

These tests don't run any subprocess — they just lock in the CLI contract so
the migration is provably non-behavioral.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from server.models import ProteinMPNNRequest, RF2Request, RFdiffusionRequest
from server.settings import RFantibodySettings
from server.tools import (
    PROTEINMPNN_OUTPUT,
    RF2_OUTPUT,
    RFDIFFUSION_OUTPUT,
    proteinmpnn_argv,
    rf2_argv,
    rfdiffusion_argv,
)


@pytest.fixture
def settings(tmp_path: Path) -> RFantibodySettings:
    # Use tmp_path as the project root so the weight-check branch (which inspects
    # the filesystem) deterministically takes the "missing" path.
    return RFantibodySettings(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path / "rfantibody",
        weights_dir=tmp_path / "weights",
        scripts_dir=tmp_path / "scripts",
    )


def test_rfdiffusion_argv_includes_required_overrides(
    settings: RFantibodySettings, tmp_path: Path
) -> None:
    job_dir = tmp_path / "j1"
    (job_dir / "output").mkdir(parents=True)
    req = RFdiffusionRequest(num_designs=20, hotspots="B1,B2", deterministic=True)
    argv = rfdiffusion_argv(
        req,
        target_pdb=tmp_path / "target.pdb",
        framework_pdb=tmp_path / "framework.pdb",
        job_dir=job_dir,
        settings=settings,
    )
    assert argv[1].endswith("rfdiffusion_inference.py")
    assert "--config-name" in argv and "antibody" in argv
    joined = " ".join(argv)
    assert "antibody.target_pdb=" in joined
    assert "inference.num_designs=20" in joined
    assert "ppi.hotspot_res=[B1,B2]" in joined
    assert f"inference.quiver={job_dir / 'output' / RFDIFFUSION_OUTPUT}" in joined
    assert "inference.deterministic=True" in argv
    assert "inference.write_trajectory=False" in argv  # no_trajectory default True


def test_rfdiffusion_argv_omits_hotspots_when_unset(
    settings: RFantibodySettings, tmp_path: Path
) -> None:
    job_dir = tmp_path / "j2"
    (job_dir / "output").mkdir(parents=True)
    req = RFdiffusionRequest()  # hotspots=None
    argv = rfdiffusion_argv(
        req,
        target_pdb=tmp_path / "t.pdb",
        framework_pdb=tmp_path / "f.pdb",
        job_dir=job_dir,
        settings=settings,
    )
    joined = " ".join(argv)
    assert "ppi.hotspot_res" not in joined


def test_proteinmpnn_argv_flag_style(settings: RFantibodySettings, tmp_path: Path) -> None:
    job_dir = tmp_path / "j3"
    (job_dir / "output").mkdir(parents=True)
    req = ProteinMPNNRequest(temperature=0.5, deterministic=True)
    argv = proteinmpnn_argv(
        req, input_quiver=tmp_path / "in.qv", job_dir=job_dir, settings=settings,
    )
    assert "-quiver" in argv
    # Output path is in the expected place.
    out_idx = argv.index("-outquiver")
    assert argv[out_idx + 1] == str(job_dir / "output" / PROTEINMPNN_OUTPUT)
    assert "-temperature" in argv and "0.5" in argv
    assert "-deterministic" in argv


def test_rf2_argv_hydra_style(settings: RFantibodySettings, tmp_path: Path) -> None:
    job_dir = tmp_path / "j4"
    (job_dir / "output").mkdir(parents=True)
    req = RF2Request(seed=42, num_recycles=5)
    argv = rf2_argv(req, input_quiver=tmp_path / "in.qv", job_dir=job_dir, settings=settings)
    joined = " ".join(argv)
    assert f"output.quiver={job_dir / 'output' / RF2_OUTPUT}" in joined
    assert "inference.num_recycles=5" in joined
    assert "inference.cautious=False" in argv  # always-on safety flag
    assert "+inference.seed=42" in argv


def test_argv_appends_weights_when_present(
    settings: RFantibodySettings, tmp_path: Path
) -> None:
    """If a checkpoint file exists, the argv carries an override pointing at it."""
    settings.weights_dir.mkdir(parents=True, exist_ok=True)
    (settings.weights_dir / "RFdiffusion_Ab.pt").write_bytes(b"\x00")
    job_dir = tmp_path / "j5"
    (job_dir / "output").mkdir(parents=True)
    argv = rfdiffusion_argv(
        RFdiffusionRequest(),
        target_pdb=tmp_path / "t.pdb",
        framework_pdb=tmp_path / "f.pdb",
        job_dir=job_dir,
        settings=settings,
    )
    assert any("inference.ckpt_override_path=" in tok for tok in argv)
