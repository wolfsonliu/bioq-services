"""bindcraft2-server 的 CLI 批处理入口。

用法::

    python -m server design --target-name hPDL1 --max-trajectories 200 \\
        --output-dir /scratch/$SLURM_JOB_ID/
    python -m server design --target /data/target.pdb --target-chains A \\
        --output-dir /scratch/$SLURM_JOB_ID/
    python -m server rank --campaign-uri file:///scratch/run1/output --on i_pTM \\
        --output-dir /scratch/run2/
    python -m server filter --campaign-uri file:///scratch/run1/output \\
        --where 'i_pAE=0.45' --output-dir /scratch/run3/
"""

from __future__ import annotations

import sys
from pathlib import Path

from bioq_service.cli import CLIEndpoint, create_cli

from .adapter import Bindcraft2Adapter
from .campaigns import resolve_campaign_dir
from .models import DesignRequest, FilterRequest, RankRequest
from .settings import Bindcraft2Settings
from .tools import design_argv, filter_argv, prepare_design, rank_argv

settings = Bindcraft2Settings()
adapter = Bindcraft2Adapter(settings=settings)


def _design_build(req: DesignRequest, inputs: dict[str, Path], job_dir: Path, settings) -> list[str]:
    target_path = inputs.get("target")
    # "二选一"这种条件必填只能在 build 回调里查：create_cli 只强制 argparse 层的
    # required，表达不了跨字段条件（`inputs["target"]` 声明为可选，见 CLIEndpoint）。
    # 不查的话会一路走到 prepare_design 抛 ValueError，用户看到的是一整坨
    # traceback 而不是用法提示。仓库既有先例：services/diffdock-server/cli_impl.py。
    if target_path is None and not req.target_name:
        print("error: one of --target or --target-name is required", file=sys.stderr)
        raise SystemExit(2)
    campaign_file = prepare_design(
        req, job_dir=job_dir, target_path=target_path, settings=settings
    )
    return design_argv(campaign_file, job_dir, settings)


def _rank_build(req: RankRequest, _inputs: dict[str, Path], job_dir: Path, settings) -> list[str]:
    return rank_argv(req, resolve_campaign_dir(req.campaign_uri, settings), job_dir, settings)


def _filter_build(req: FilterRequest, _inputs: dict[str, Path], job_dir: Path, settings) -> list[str]:
    return filter_argv(req, resolve_campaign_dir(req.campaign_uri, settings), job_dir, settings)


endpoints = {
    "design": CLIEndpoint(
        name="design",
        help="Run a full BindCraft2 design campaign",
        request_model=DesignRequest,
        build_argv=_design_build,
        inputs={
            "target": (
                "Target structure (PDB/mmCIF/FASTA); omit when using --target-name",
                False,
            )
        },
    ),
    "rank": CLIEndpoint(
        name="rank",
        help="Re-rank an existing campaign on other measurements",
        request_model=RankRequest,
        build_argv=_rank_build,
    ),
    "filter": CLIEndpoint(
        name="filter",
        help="Re-apply acceptance thresholds to an existing campaign",
        request_model=FilterRequest,
        build_argv=_filter_build,
    ),
}

if __name__ == "__main__":
    # 守卫是必需的，不是风格问题：tests/test_cli.py 直接 import 本模块来复用
    # endpoints 注册表（避免像 rfantibody 那样在测试里重复一份）。无守卫时 import
    # 会立刻执行 create_cli，按 pytest 的 sys.argv 解析 → argparse SystemExit(2)。
    # `python -m server <endpoint>` 路径下 __name__ == "__main__"，行为不变。
    create_cli(adapter, settings, endpoints, version="0.0.1")
