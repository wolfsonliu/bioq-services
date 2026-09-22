"""CLI 批处理模式：endpoint 注册、argv 回调、create_cli 端到端。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from bioq_service.cli import create_cli
from server.adapter import Bindcraft2Adapter

import server.__main__ as cli_main


@pytest.fixture
def offline(tmp_path: Path, offline_settings):
    return offline_settings


def test_endpoint_registry_covers_three_subcommands():
    assert set(cli_main.endpoints.keys()) == {"design", "rank", "filter"}


def test_design_endpoint_declares_optional_target_input():
    ep = cli_main.endpoints["design"]
    assert "target" in ep.inputs
    help_text, required = ep.inputs["target"]
    assert required is False, "target 可选：也可以走 --target-name"
    assert "target-name" not in ep.inputs


def test_rank_and_filter_have_no_file_inputs():
    assert cli_main.endpoints["rank"].inputs == {}
    assert cli_main.endpoints["filter"].inputs == {}


def test_design_build_writes_campaign(tmp_path, offline):
    job_dir = tmp_path / "run"
    (job_dir / "output").mkdir(parents=True)
    req = cli_main.endpoints["design"].request_model(target_name="hPDL1", max_trajectories=3)
    argv = cli_main.endpoints["design"].build_argv(req, {}, job_dir, offline)
    assert argv[-2] == "design"
    campaign = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
    assert campaign["target"] == "hPDL1"
    assert campaign["max_trajectories"] == 3


def test_design_build_uses_target_input(tmp_path, offline):
    job_dir = tmp_path / "run"
    (job_dir / "output").mkdir(parents=True)
    target = tmp_path / "t.pdb"
    target.write_text("ATOM\n", encoding="utf-8")
    req = cli_main.endpoints["design"].request_model(max_trajectories=1)
    argv = cli_main.endpoints["design"].build_argv(req, {"target": target}, job_dir, offline)
    campaign = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
    assert campaign["targets"][0]["target_path"] == str(target.resolve())


def test_rank_build_resolves_campaign_uri(tmp_path, offline):
    campaign = tmp_path / "jobs" / "abc" / "output"
    campaign.mkdir(parents=True)
    job_dir = tmp_path / "run"
    (job_dir / "output").mkdir(parents=True)
    req = cli_main.endpoints["rank"].request_model(campaign_uri="job://abc", on="i_pTM")
    argv = cli_main.endpoints["rank"].build_argv(req, {}, job_dir, offline)
    assert str(campaign) in argv
    assert argv[-1] == str((job_dir / "output" / "ranked_by_i_pTM.csv").resolve())


def test_cli_design_success(tmp_path, offline):
    adapter = Bindcraft2Adapter(settings=offline)
    output_dir = tmp_path / "run" / "output"
    with patch.object(sys, "argv", [
        "prog", "design", "--target-name", "hPDL1", "--max-trajectories", "1",
        "--output-dir", str(output_dir),
    ]):
        with pytest.raises(SystemExit) as exc:
            create_cli(adapter, offline, cli_main.endpoints, version="0.0.1")
    assert exc.value.code == 0
    assert (output_dir / "3_Ranked" / "!_Ranked.csv").is_file()


def test_cli_rank_json_output(tmp_path, offline, capsys):
    adapter = Bindcraft2Adapter(settings=offline)
    campaign = tmp_path / "jobs" / "abc" / "output"
    campaign.mkdir(parents=True)
    output_dir = tmp_path / "run" / "output"
    with patch.object(sys, "argv", [
        "prog", "rank", "--campaign-uri", "job://abc", "--on", "i_pTM",
        "--json", "--output-dir", str(output_dir),
    ]):
        with pytest.raises(SystemExit) as exc:
            create_cli(adapter, offline, cli_main.endpoints, version="0.0.1")
    assert exc.value.code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"


def test_cli_no_subcommand_exits_2(tmp_path, offline):
    adapter = Bindcraft2Adapter(settings=offline)
    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, offline, cli_main.endpoints)


def test_cli_requires_target_or_target_name(tmp_path, offline):
    adapter = Bindcraft2Adapter(settings=offline)
    output_dir = tmp_path / "run" / "output"
    with patch.object(sys, "argv", ["prog", "design", "--output-dir", str(output_dir)]):
        with pytest.raises(SystemExit) as exc:
            create_cli(adapter, offline, cli_main.endpoints)
    assert exc.value.code != 0
