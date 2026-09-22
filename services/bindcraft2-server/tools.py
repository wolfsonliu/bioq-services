"""campaign 文件拼装与 argv 构造。

本模块只构造 argv，从不 spawn 进程——子进程生命周期归框架
（`bioq_service.runner.SubprocessRunner`）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .models import DesignRequest, FilterRequest, RankRequest
from .settings import Bindcraft2Settings

__all__ = [
    "CAMPAIGN_MODELS",
    "DESIGN_RANKED_CSV",
    "DESIGN_SUMMARY_CSV",
    "FILTER_OUTPUT",
    "MPNN_VARIANTS",
    "build_campaign_json",
    "design_argv",
    "filter_argv",
    "missing_alphafold_params",
    "missing_proteinmpnn_weights",
    "prepare_design",
    "rank_argv",
    "rank_output_name",
    "write_campaign_file",
]

# 上游 bindcraft/model_weights.py: CAMPAIGN_MODELS —— 5 个 multimer + 2 个 monomer。
CAMPAIGN_MODELS: tuple[str, ...] = tuple(
    f"model_{index}_multimer_v3" for index in range(1, 6)
) + ("model_1_ptm", "model_2_ptm")

MPNN_VARIANTS: tuple[str, ...] = ("neutral", "negative", "positive")
MPNN_CHECKPOINT = "v_48_020.npz"

# 上游输出约定（docs/outputs.md）。
DESIGN_RANKED_CSV = "3_Ranked/!_Ranked.csv"
DESIGN_SUMMARY_CSV = "summary.csv"
TRAJECTORIES_CSV = "1_Trajectories/!_Trajectories.csv"
CAMPAIGN_METADATA_JSON = "campaign_metadata.json"
FILTER_OUTPUT = "filtered.csv"


def runner_prefix(settings: Bindcraft2Settings) -> list[str]:
    """`[python, -m, module]`；module 为空时退化成 `[python]`（离线 stub / 上游改名）。"""
    if settings.module:
        return [settings.python, "-m", settings.module]
    return [settings.python]


def alphafold_parameter_path(parameters_dir: Path, model_name: str) -> Path | None:
    """按上游 `alphafold_parameter_file` 的候选顺序查找，返回第一个存在的。"""
    for candidate in (
        parameters_dir / "params" / f"params_{model_name}.npz",
        parameters_dir / f"params_{model_name}.npz",
        parameters_dir / "params" / f"{model_name}.npz",
        parameters_dir / f"{model_name}.npz",
    ):
        if candidate.is_file():
            return candidate
    return None


def missing_alphafold_params(
    parameters_dir: Path, *, floor_bytes: int = 100 << 20
) -> list[str]:
    """返回缺失或体积不足（下载中断）的检查点名。

    上游对 AlphaFold 检查点用 100 MB 下限判"未完成"（
    `ALPHAFOLD_CHECKPOINT_FLOOR = 100 << 20`）。
    """
    missing: list[str] = []
    for name in CAMPAIGN_MODELS:
        path = alphafold_parameter_path(parameters_dir, name)
        if path is None or path.stat().st_size < floor_bytes:
            missing.append(f"params_{name}.npz")
    return missing


def missing_proteinmpnn_weights(root: Path, *, floor_bytes: int = 1 << 20) -> list[str]:
    """ProteinMPNN 三变体随包（~6.6 MB each，上游下限 1 MB）。"""
    missing: list[str] = []
    for variant in MPNN_VARIANTS:
        path = (
            root
            / "bindcraft"
            / "weights"
            / "proteinmpnn"
            / f"weights_{variant}"
            / MPNN_CHECKPOINT
        )
        if not path.is_file() or path.stat().st_size < floor_bytes:
            missing.append(f"weights_{variant}/{MPNN_CHECKPOINT}")
    return missing


def build_campaign_json(
    req: DesignRequest,
    *,
    job_dir: Path,
    target_path: Path | None,
    max_trajectories: int,
) -> dict:
    """把请求编译成上游 campaign JSON。

    上游 campaign JSON 的 key 与请求字段一一对应；服务额外写死
    `project_folder`（指向本 job 的 output/），并注入 `max_trajectories`。
    """
    campaign: dict = {
        "campaign_name": req.campaign_name or job_dir.name,
        "modality": req.modality,
        "number_of_final_designs": req.number_of_final_designs,
        "max_trajectories": max_trajectories,
        "project_folder": str((job_dir / "output").resolve()),
    }

    if req.target_name:
        campaign["target"] = req.target_name
    else:
        if target_path is None:
            raise ValueError("target_path is required when target_name is unset")
        entry: dict = {
            # 服务生成的名字：用户文件名可能含空格 / 非 ASCII / 长度不定，
            # 而上游用它拼结果文件名与 hash。
            "name": "target",
            "target_path": str(target_path.resolve()),
        }
        if req.target_chains:
            entry["chains"] = req.target_chains
        if req.hotspots:
            entry["hotspots"] = req.hotspots
        if req.coldspots:
            entry["coldspots"] = req.coldspots
        campaign["targets"] = [entry]

    if req.binder_lengths:
        campaign["binder_lengths"] = list(req.binder_lengths)
    if req.core:
        campaign["core"] = req.core
    for name in req.properties:
        campaign[name] = True

    return campaign


def write_campaign_file(campaign: dict, job_dir: Path) -> Path:
    """把 campaign 写到 `<job_dir>/input/campaign.json` 并返回该路径。"""
    path = job_dir / "input" / "campaign.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def prepare_design(
    req: DesignRequest,
    *,
    job_dir: Path,
    target_path: Path | None,
    settings: Bindcraft2Settings,
) -> Path:
    """HTTP 与 CLI 共用的 design 前置：拼装 + 落盘 campaign 文件。"""
    max_trajectories = req.max_trajectories or settings.default_max_trajectories
    campaign = build_campaign_json(
        req, job_dir=job_dir, target_path=target_path, max_trajectories=max_trajectories
    )
    return write_campaign_file(campaign, job_dir)


def design_argv(
    campaign_file: Path, job_dir: Path, settings: Bindcraft2Settings
) -> list[str]:
    """`python -m bindcraft.cli design <campaign.json>`。"""
    return runner_prefix(settings) + ["design", str(campaign_file.resolve())]


def rank_output_name(metrics: list[str]) -> str:
    """rank 产物文件名，与上游 `rank.ranked_filename()` 同口径。

    所有指标按顺序拼进去、非字母数字折成 `_`：`on=["i_pTM","i_pAE"]` →
    `ranked_by_i_pTM_i_pAE.csv`。只用 `metrics[0]` 会让不同 tie-break 的产物撞名，
    事后无法分辨用的是哪个排序。
    """
    named = re.sub(r"[^0-9A-Za-z]+", "_", "_".join(metrics)).strip("_")
    return f"ranked_by_{named}.csv"


def rank_argv(
    req: RankRequest,
    campaign_dir: Path,
    job_dir: Path,
    settings: Bindcraft2Settings,
) -> list[str]:
    """`... rank <campaign_dir> --on M [--on M2] --table T --output <job>/output/ranked_by_<M...>.csv`。"""
    output = job_dir / "output" / rank_output_name(req.on)
    cmd = runner_prefix(settings) + ["rank", str(campaign_dir)]
    for metric in req.on:
        cmd += ["--on", metric]
    cmd += ["--table", req.table.value, "--output", str(output.resolve())]
    if req.lowest_first is not None:
        cmd.append("--lowest-first" if req.lowest_first else "--highest-first")
    if req.top is not None:
        cmd += ["--top", str(req.top)]
    return cmd


def filter_argv(
    req: FilterRequest,
    campaign_dir: Path,
    job_dir: Path,
    settings: Bindcraft2Settings,
) -> list[str]:
    """`... filter <campaign_dir> [--where EXPR]... --table T --output <job>/output/filtered.csv`。"""
    output = job_dir / "output" / FILTER_OUTPUT
    cmd = runner_prefix(settings) + ["filter", str(campaign_dir)]
    for expression in req.where or []:
        cmd += ["--where", expression]
    cmd += ["--table", req.table.value, "--output", str(output.resolve())]
    if req.top is not None:
        cmd += ["--top", str(req.top)]
    return cmd
