"""campaign JSON 拼装、三组 argv、权重探针。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic_settings import SettingsConfigDict

from server.models import DesignRequest, FilterRequest, RankRequest
from server.settings import Bindcraft2Settings
from server.tools import (
    CAMPAIGN_MODELS,
    build_campaign_json,
    design_argv,
    filter_argv,
    missing_alphafold_params,
    missing_proteinmpnn_weights,
    prepare_design,
    rank_argv,
    write_campaign_file,
)


class _Off(Bindcraft2Settings):
    model_config = SettingsConfigDict(
        env_prefix="BINDCRAFT2_TEST_", env_file=None, extra="ignore", case_sensitive=False
    )


@pytest.fixture
def settings(tmp_path: Path) -> _Off:
    return _Off(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path / "root",
        python="/usr/bin/python3",
        alphafold_params_dir=tmp_path / "af",
    )


@pytest.fixture
def job_dir(tmp_path: Path) -> Path:
    d = tmp_path / "jobs" / "j1"
    (d / "output").mkdir(parents=True)
    return d


def test_campaign_models_match_upstream_order():
    assert CAMPAIGN_MODELS == (
        "model_1_multimer_v3",
        "model_2_multimer_v3",
        "model_3_multimer_v3",
        "model_4_multimer_v3",
        "model_5_multimer_v3",
        "model_1_ptm",
        "model_2_ptm",
    )


def test_shipped_target_campaign(job_dir):
    campaign = build_campaign_json(
        DesignRequest(target_name="hPDL1", number_of_final_designs=5),
        job_dir=job_dir,
        target_path=None,
        max_trajectories=50,
    )
    assert campaign["target"] == "hPDL1"
    assert "targets" not in campaign
    assert campaign["number_of_final_designs"] == 5
    assert campaign["max_trajectories"] == 50
    assert campaign["modality"] == "binder"
    assert campaign["project_folder"] == str((job_dir / "output").resolve())


def test_uploaded_target_campaign_uses_service_generated_name(job_dir):
    target = job_dir / "input" / "target.pdb"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("ATOM\n", encoding="utf-8")

    campaign = build_campaign_json(
        DesignRequest(target_chains="A", hotspots="54,56", coldspots="90-95"),
        job_dir=job_dir,
        target_path=target,
        max_trajectories=10,
    )
    entry = campaign["targets"][0]
    assert entry["name"] == "target"  # 不取用户文件名
    assert entry["target_path"] == str(target.resolve())
    assert entry["chains"] == "A"
    assert entry["hotspots"] == "54,56"
    assert entry["coldspots"] == "90-95"


def test_campaign_omits_unset_optionals(job_dir):
    campaign = build_campaign_json(
        DesignRequest(target_name="hPDL1"), job_dir=job_dir, target_path=None, max_trajectories=1
    )
    assert "binder_lengths" not in campaign
    assert "core" not in campaign
    assert "humanize" not in campaign
    assert campaign["campaign_name"] == "j1"  # 未给则用 job_id


def test_campaign_writes_enabled_properties(job_dir):
    campaign = build_campaign_json(
        DesignRequest(target_name="hPDL1", humanize=True, bigbang=True, binder_lengths=[60, 60]),
        job_dir=job_dir,
        target_path=None,
        max_trajectories=1,
    )
    assert campaign["humanize"] is True
    assert campaign["bigbang"] is True
    assert "protease_stable" not in campaign
    assert campaign["binder_lengths"] == [60, 60]


def test_write_campaign_file_roundtrips(job_dir):
    path = write_campaign_file({"target": "hPDL1"}, job_dir)
    assert path == job_dir / "input" / "campaign.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"target": "hPDL1"}


def test_design_argv(settings, job_dir):
    argv = design_argv(job_dir / "input" / "campaign.json", job_dir, settings)
    assert argv == [
        "/usr/bin/python3",
        "-m",
        "bindcraft.cli",
        "design",
        str((job_dir / "input" / "campaign.json").resolve()),
    ]


def test_runner_prefix_omits_module_when_blank(tmp_path, job_dir):
    s = _Off(python="/bin/true", module="")
    argv = design_argv(job_dir / "input" / "campaign.json", job_dir, s)
    assert argv[0] == "/bin/true"
    assert "-m" not in argv


def test_prepare_design_returns_written_campaign(settings, job_dir):
    path = prepare_design(
        DesignRequest(target_name="hPDL1"), job_dir=job_dir, target_path=None, settings=settings
    )
    campaign = json.loads(path.read_text(encoding="utf-8"))
    assert campaign["target"] == "hPDL1"
    # 未指定 max_trajectories 时用 settings 兜底
    assert campaign["max_trajectories"] == settings.default_max_trajectories


def test_rank_argv_maps_every_field(settings, job_dir, tmp_path):
    campaign = tmp_path / "src"
    campaign.mkdir()
    argv = rank_argv(
        RankRequest(campaign_uri=str(campaign), on=["i_pTM", "i_pDAE"], lowest_first=True, top=20),
        campaign,
        job_dir,
        settings,
    )
    joined = " ".join(argv)
    assert "rank" in argv
    assert "--on i_pTM --on i_pDAE" in joined
    assert "--table accepted" in joined
    assert "--lowest-first" in joined
    assert "--top 20" in joined
    assert argv[argv.index("--output") + 1] == str((job_dir / "output" / "ranked_by_i_pTM.csv").resolve())


def test_rank_argv_omits_optional_flags(settings, job_dir, tmp_path):
    campaign = tmp_path / "src"
    campaign.mkdir()
    argv = rank_argv(RankRequest(campaign_uri=str(campaign)), campaign, job_dir, settings)
    assert "--lowest-first" not in argv
    assert "--highest-first" not in argv
    assert "--top" not in argv


def test_rank_argv_uses_highest_first_when_false(settings, job_dir, tmp_path):
    campaign = tmp_path / "src"
    campaign.mkdir()
    argv = rank_argv(
        RankRequest(campaign_uri=str(campaign), lowest_first=False), campaign, job_dir, settings
    )
    assert "--highest-first" in argv


def test_filter_argv_maps_every_field(settings, job_dir, tmp_path):
    campaign = tmp_path / "src"
    campaign.mkdir()
    argv = filter_argv(
        FilterRequest(campaign_uri=str(campaign), where=["i_pAE=0.45", "Interface_Residues>=7"], top=5),
        campaign,
        job_dir,
        settings,
    )
    joined = " ".join(argv)
    assert "filter" in argv
    assert "--where i_pAE=0.45 --where Interface_Residues>=7" in joined
    assert "--table candidates" in joined
    assert "--top 5" in joined
    assert argv[argv.index("--output") + 1] == str((job_dir / "output" / "filtered.csv").resolve())


def test_filter_argv_without_where(settings, job_dir, tmp_path):
    campaign = tmp_path / "src"
    campaign.mkdir()
    argv = filter_argv(FilterRequest(campaign_uri=str(campaign)), campaign, job_dir, settings)
    assert "--where" not in argv


def test_missing_alphafold_params_reports_all_seven(tmp_path):
    missing = missing_alphafold_params(tmp_path / "af")
    assert len(missing) == 7
    assert "params_model_1_multimer_v3.npz" in missing


def test_missing_alphafold_params_accepts_both_layouts(tmp_path):
    af = tmp_path / "af" / "params"
    af.mkdir(parents=True)
    for name in CAMPAIGN_MODELS:
        (af / f"params_{name}.npz").write_bytes(b"x" * (101 << 20))
    assert missing_alphafold_params(tmp_path / "af") == []


def test_missing_alphafold_params_flags_truncated(tmp_path):
    af = tmp_path / "af" / "params"
    af.mkdir(parents=True)
    for name in CAMPAIGN_MODELS:
        (af / f"params_{name}.npz").write_bytes(b"x" * (101 << 20))
    (af / "params_model_1_multimer_v3.npz").write_bytes(b"x")  # 中断的下载
    assert missing_alphafold_params(tmp_path / "af") == ["params_model_1_multimer_v3.npz"]


def test_missing_proteinmpnn_weights(tmp_path):
    root = tmp_path / "root"
    for variant in ("neutral", "negative", "positive"):
        d = root / "bindcraft" / "weights" / "proteinmpnn" / f"weights_{variant}"
        d.mkdir(parents=True)
        (d / "v_48_020.npz").write_bytes(b"x" * (2 << 20))
    assert missing_proteinmpnn_weights(root) == []

    (root / "bindcraft" / "weights" / "proteinmpnn" / "weights_positive" / "v_48_020.npz").unlink()
    assert missing_proteinmpnn_weights(root) == ["weights_positive/v_48_020.npz"]
