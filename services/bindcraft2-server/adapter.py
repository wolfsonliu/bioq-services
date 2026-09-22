"""bindcraft2-server 的服务级策略。

三处偏离框架默认：

  * `detect_outputs` 接受 design / rank / filter 三类产物中的任意一种——一个
    adapter 服务三组 endpoint。
  * `subprocess_env` 注入 BC2 必需的环境变量（AF2 参数目录、XLA 编译缓存、
    worker 上限），否则上游会尝试下载 5.3 GB 参数或按节点内存过度 pack worker。
  * `subprocess_cwd` 返回上游源码树（`settings.root`）——**但理由不是**"上游按
    相对路径解析 `settings/` / `scaffolds/`"：上游用 `Path(__file__).parent.parent`
    绝对定位 preset 树（`upstream/bindcraft/settings.py:41,353`），与 cwd 无关。
    这里返回它只是因为 `Popen(cwd=...)` 要求一个**已存在**的目录（不存在直接
    ENOENT）；指到一个"存在但无关"的目录在功能上无害，保留 `/opt/bindcraft` 只是
    与镜像布局一致。
"""

from __future__ import annotations

import logging
from pathlib import Path

from bioq_service import EndpointExample, JobAdapter, JobInfo, JobStatus

from .settings import Bindcraft2Settings
from .tools import (
    CAMPAIGN_METADATA_JSON,
    CAMPAIGN_MODELS,
    DESIGN_RANKED_CSV,
    DESIGN_SUMMARY_CSV,
    FILTER_OUTPUT,
    TRAJECTORIES_CSV,
)

logger = logging.getLogger(__name__)


class Bindcraft2Adapter(JobAdapter):
    name = "bindcraft2"

    settings: Bindcraft2Settings  # 收窄类型，便于 IDE 提示

    def __init__(self, settings: Bindcraft2Settings) -> None:
        super().__init__(settings)

    # ---- 子进程环境 / 工作目录 ----

    def subprocess_cwd(self) -> Path | None:
        # 只因为 Popen(cwd=...) 需要一个存在的目录；上游的资源定位是绝对路径
        # （见模块 docstring），cwd 不参与解析。
        return self.settings.root

    def subprocess_env(self) -> dict[str, str]:
        cache_dir = self.settings.compile_cache_dir
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # NAS 不可写时不能让整个 job 失败——只是每次冷启动都要重编译。
            logger.warning(
                "compile cache dir %s is not writable (%s); JAX will recompile every run",
                cache_dir,
                exc,
            )
        return {
            # 必须显式设置：缺省时上游会去下载 alphafold_params_2022-12-06.tar。
            "BINDCRAFT_AF2_PARAMS": str(self.settings.alphafold_params_dir),
            "JAX_COMPILATION_CACHE_DIR": str(cache_dir),
            # BC2 默认吃掉所有可见卡并按显存 pack worker；FC 一实例一卡，锁死 1。
            "BINDCRAFT_WORKERS_PER_GPU": str(self.settings.workers_per_gpu),
            "BINDCRAFT_MAX_WORKERS_PER_GPU": str(self.settings.max_workers_per_gpu),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }

    # ---- 产物判定 ----

    def detect_outputs(self, job_dir: Path) -> bool:
        """design / rank / filter 任一类产物存在且非空即算成功。

        必须包含 `summary.csv`：一场合法完成但 accepted=0 的 campaign 不会写
        `3_Ranked/!_Ranked.csv`。放宽是安全的——rc != 0 仍由框架 `finalize_job`
        判失败，这些文件只在 rc == 0 时才是成功证据。
        """
        out = self.output_dir(job_dir)
        for candidate in (
            out / DESIGN_RANKED_CSV,
            out / DESIGN_SUMMARY_CSV,
            out / FILTER_OUTPUT,
        ):
            if candidate.is_file() and candidate.stat().st_size > 0:
                return True
        return any(
            path.is_file() and path.stat().st_size > 0
            for path in out.glob("ranked_by_*.csv")
        )

    def infer_job_from_dir(self, job_dir: Path) -> JobInfo:
        """从磁盘恢复 job；progress 报告 accepted 数量（rank/filter 报 finished）。"""
        job_id = job_dir.name
        if not self.detect_outputs(job_dir):
            return JobInfo(
                job_id=job_id,
                status=JobStatus.FAILED,
                service=self.name,
                message="Recovered from disk (no outputs)",
            )

        ranked = self.output_dir(job_dir) / DESIGN_RANKED_CSV
        if ranked.is_file():
            rows = ranked.read_text(encoding="utf-8", errors="replace").splitlines()
            accepted = max(0, len(rows) - 1)  # 减表头
            return JobInfo(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                service=self.name,
                progress=f"{accepted} accepted",
                message=f"Recovered from disk ({accepted} accepted designs)",
            )
        return JobInfo(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            service=self.name,
            progress="finished",
            message="Recovered from disk (no accepted designs)",
        )

    # ---- Agent 面向的元数据 ----

    def manifest_extras(self) -> dict:
        return {
            "tool_outputs": {
                "ranked": DESIGN_RANKED_CSV,
                "summary": DESIGN_SUMMARY_CSV,
                "trajectories": TRAJECTORIES_CSV,
                "metadata": CAMPAIGN_METADATA_JSON,
                "rank": "ranked_by_<m1>[_<m2>...].csv",
                "filter": FILTER_OUTPUT,
            },
            # 按输入面分层：target（单文件）与 campaign_uri（目录）接受的 scheme 不同。
            # 合成一个平铺 dict 会让人以为 campaign_uri 也吃 oss:// / http(s)://
            # （它不吃，会 422）。
            "input_uri_schemes": {
                "target": {
                    "upload": "multipart/form-data UploadFile",
                    "job://<job_id>/<file>": "上一 job 的 output/ 里的单个文件",
                    "file:///abs/path": "NAS 绝对路径（网关把 oss:// 改写成 /mnt/oss/... 后走这条）",
                    "oss://<bucket>/<key>": "Alibaba Cloud OSS 对象",
                    "http(s)://...": "通用 URL",
                },
                "campaign_uri": {
                    "job://<job_id>[/<subdir>]": "上一 job 的 output/ 目录（通常零拷贝，共享 NAS）",
                    "file:///abs/path": "NAS 绝对目录",
                    "/abs/path": "裸绝对目录",
                    "不支持": "oss:// 与 http(s):// 一律 422；网关把 oss:// 改写成 /mnt/oss/... 后走裸路径",
                },
            },
            "chaining_tip": (
                "design 完成后用 campaign_uri=job://<design_job_id> 调 /api/rank 或 "
                "/api/filter，无需重新上传或重跑设计。源表已有记录行且指标可派生时 rank "
                "只读源目录；源表无记录行但有结构文件（或 --rescore）时上游会把 scored.csv "
                "写进源目录。"
            ),
            "campaign_knobs": {
                "modality": (
                    "binder / large_binder / peptide / cyclic_peptide / homo_oligomer / "
                    "multidomain / VHH / ARP / scFv / Fab / induced_fit / fold_switch"
                ),
                "properties": [
                    "forced_targeting",
                    "humanize",
                    "protease_stable",
                    "disulfide_staple",
                    "mixed_topology",
                    "termini_together",
                    "termini_accessible",
                    "initial_guess",
                    "bigbang",
                ],
                "note": "modality 可逗号组合；兼容性由上游 preflight 校验，失败原因见 job 日志。",
            },
            "weights": {
                "alphafold_params_dir": str(self.settings.alphafold_params_dir),
                "expected_alphafold_models": [f"params_{m}.npz" for m in CAMPAIGN_MODELS],
                # 上游 alphafold_parameter_file 的四种候选布局（顺序敏感）。
                "expected_files": [
                    "params/params_<model>.npz",
                    "params_<model>.npz",
                    "params/<model>.npz",
                    "<model>.npz",
                ],
                "proteinmpnn": "随包（<root>/bindcraft/weights/proteinmpnn/weights_{neutral,negative,positive}/v_48_020.npz）",
            },
            "gpu": (
                "无 CPU 模式。cuda13 extra 需 compute capability >= 7.5；"
                "单 worker 显存预算 2.0*(3.4GB + 38kB*N^2)，其中 N 是 padding(bucket 32) 后的"
                "复合物残基数（不是 binder_lengths），另留 4 GB headroom。"
            ),
            "license_notice": (
                "BindCraft2 采用 Source-Available License (Hosting-Restricted)。"
                "本服务仅限组织内部使用；不得作为 hosted/API/agent action 提供给第三方。"
            ),
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/design": [
                EndpointExample(
                    title="shipped target，小 campaign",
                    curl=(
                        "curl -X POST $URL/api/design "
                        "-F target_name=hPDL1 "
                        "-F 'binder_lengths=[80,80]' "
                        "-F number_of_final_designs=10 "
                        "-F max_trajectories=200"
                    ),
                    notes=(
                        "shipped target 见 `bindcraft design --list-targets`。"
                        "HTTP 上复杂字段（binder_lengths / on / where）必须写成合法 JSON "
                        "字符串；逗号写法只对 CLI 有效，表单里发 80,80 会 422 json_invalid。"
                    ),
                ),
                EndpointExample(
                    title="上传自己的 target + hotspot",
                    curl=(
                        "curl -X POST $URL/api/design "
                        "-F target=@target.pdb "
                        "-F target_chains=A "
                        "-F 'hotspots=54,56,66-70' "
                        "-F modality=VHH "
                        "-F humanize=true "
                        "-F number_of_final_designs=10"
                    ),
                    python=(
                        "import httpx\n"
                        "with open('target.pdb', 'rb') as fh:\n"
                        "    r = httpx.post(\n"
                        "        f'{base_url}/api/design',\n"
                        "        files={'target': fh},\n"
                        "        data={\n"
                        "            'target_chains': 'A',\n"
                        "            'hotspots': '54,56,66-70',\n"
                        "            'modality': 'VHH',\n"
                        "            'humanize': 'true',\n"
                        "            'number_of_final_designs': '10',\n"
                        "        },\n"
                        "        timeout=120,\n"
                        "    )\n"
                        "job_id = r.json()['job_id']"
                    ),
                    notes="target / target_name / target_uri 三选一，多给或多不给都是 422。",
                ),
            ],
            "/api/tasks/design": [
                EndpointExample(
                    title="FC 异步任务模式（推荐入口）",
                    curl=(
                        "curl -X POST $URL/api/tasks/design "
                        "-H 'X-Fc-Invocation-Type: Async' "
                        "-H 'bioagent-session-id: demo-session' "
                        "-F target_uri=oss://bioagent-inputs/target.pdb "
                        "-F target_chains=A "
                        "-F max_trajectories=200"
                    ),
                    notes=(
                        "campaign 是小时级，必须走异步任务模式。上传体积超过 FC 128 KiB "
                        "事件上限时只能用 *_uri。"
                    ),
                ),
            ],
            "/api/rank": [
                EndpointExample(
                    title="接续上一场 campaign，换指标重排",
                    curl=(
                        "curl -X POST $URL/api/rank "
                        "-F campaign_uri=job://<design_job_id> "
                        "-F 'on=[\"i_pTM\",\"i_pDAE\"]' "
                        "-F table=accepted"
                    ),
                    notes=(
                        "产物写到新 job 的 output/ranked_by_i_pTM_i_pDAE.csv"
                        "（文件名按 on 里全部指标拼接），源目录只读。"
                    ),
                ),
            ],
            "/api/tasks/rank": [
                EndpointExample(
                    title="task 模式重排",
                    curl=(
                        "curl -X POST $URL/api/tasks/rank "
                        "-H 'X-Fc-Invocation-Type: Async' "
                        "-F campaign_uri=job://<design_job_id> "
                        "-F 'on=[\"i_pTM\"]'"
                    ),
                ),
            ],
            "/api/filter": [
                EndpointExample(
                    title="用新阈值重放候选集",
                    curl=(
                        "curl -X POST $URL/api/filter "
                        "-F campaign_uri=job://<design_job_id> "
                        "-F 'where=[\"i_pAE=0.45\",\"Interface_Residues>=7\"]'"
                    ),
                    notes="where 为空等价于用 campaign 自身阈值重放。",
                ),
            ],
            "/api/tasks/filter": [
                EndpointExample(
                    title="task 模式重放阈值",
                    curl=(
                        "curl -X POST $URL/api/tasks/filter "
                        "-H 'X-Fc-Invocation-Type: Async' "
                        "-F campaign_uri=job://<design_job_id>"
                    ),
                ),
            ],
        }
