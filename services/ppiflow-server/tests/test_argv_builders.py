"""argv builders for the five PPIFlow samplers.

These run without PPIFlow / its weights installed — they just check the CLI
shape we produce matches what `sample_*.py --help` would accept. Catches
typos in flag names, missing --hotspots when set, JSON-encoding of list args.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic_settings import SettingsConfigDict

from server.models import (
    AntibodyRequest,
    BinderRequest,
    MonomerRequest,
    NanobodyRequest,
    ScaffoldingRequest,
)
from server.settings import PPIFlowSettings
from server.tools import (
    antibody_argv,
    binder_argv,
    monomer_argv,
    nanobody_argv,
    scaffolding_argv,
)


class _OfflineSettings(PPIFlowSettings):
    model_config = SettingsConfigDict(
        env_prefix="PPIFLOW_TEST_", env_file=None, extra="ignore",
    )


@pytest.fixture
def settings(tmp_path: Path) -> PPIFlowSettings:
    s = _OfflineSettings(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path / "ppiflow",
        ckpt_dir=tmp_path / "ppiflow" / "checkpoint",
        config_dir=tmp_path / "ppiflow" / "configs",
    )
    s.ckpt_dir.mkdir(parents=True, exist_ok=True)
    s.config_dir.mkdir(parents=True, exist_ok=True)
    return s


def test_binder_argv_includes_required_flags(settings, tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    req = BinderRequest(
        target_chain="B",
        binder_chain="A",
        specified_hotspots="B119,B141",
        samples_min_length=75,
        samples_max_length=120,
        samples_per_target=8,
        name="IL7Ra",
    )
    target_pdb = tmp_path / "target.pdb"
    target_pdb.write_text("HEADER")
    argv = binder_argv(req, target_pdb, job_dir, settings)
    assert argv[1].endswith("sample_binder.py")
    assert "--input_pdb" in argv
    assert "--target_chain" in argv and "B" in argv
    assert "--specified_hotspots" in argv and "B119,B141" in argv
    assert "--samples_per_target" in argv and "8" in argv
    assert "--name" in argv and "IL7Ra" in argv


def test_binder_argv_omits_hotspots_when_unset(settings, tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    req = BinderRequest(target_chain="B")
    argv = binder_argv(req, tmp_path / "target.pdb", job_dir, settings)
    assert "--specified_hotspots" not in argv


def test_antibody_argv_keeps_light_chain(settings, tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    req = AntibodyRequest(
        antigen_chain="C",
        heavy_chain="A",
        light_chain="B",
        cdr_length="CDRH1,8-8,CDRH2,8-8,CDRH3,10-20,CDRL1,6-9,CDRL2,3-3,CDRL3,9-11",
    )
    argv = antibody_argv(
        req,
        tmp_path / "antigen.pdb",
        tmp_path / "framework.pdb",
        job_dir,
        settings,
    )
    assert argv[1].endswith("sample_antibody_nanobody.py")
    assert "--light_chain" in argv and "B" in argv
    assert any("antibody.ckpt" in a for a in argv)


def test_nanobody_argv_drops_light_chain_and_uses_nb_ckpt(settings, tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    req = NanobodyRequest(antigen_chain="C", heavy_chain="A")
    argv = nanobody_argv(
        req,
        tmp_path / "antigen.pdb",
        tmp_path / "framework.pdb",
        job_dir,
        settings,
    )
    assert "--light_chain" not in argv
    assert any("nanobody.ckpt" in a for a in argv)


def test_monomer_argv_serializes_length_subset_as_json(settings, tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    req = MonomerRequest(length_subset=[50, 100], samples_per_target=3, name="mono")
    argv = monomer_argv(req, job_dir, settings)
    assert "--length_subset" in argv
    idx = argv.index("--length_subset")
    assert json.loads(argv[idx + 1]) == [50, 100]
    assert any("monomer.ckpt" in a for a in argv)


def test_scaffolding_argv_uses_monomer_ckpt_and_motif_names_json(
    settings, tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    req = ScaffoldingRequest(motif_names=["01_1LDB"], samples_per_target=5, name="scaf")
    motif_csv = tmp_path / "motif.csv"
    motif_csv.write_text("target,length,contig,motif_path\n01_1LDB,125,0-100,01_1LDB.pdb\n")
    argv = scaffolding_argv(req, motif_csv, job_dir, settings)
    assert any("monomer.ckpt" in a for a in argv)
    idx = argv.index("--motif_names")
    assert json.loads(argv[idx + 1]) == ["01_1LDB"]
