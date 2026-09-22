"""bindcraft2-server 的运行时配置。

所有环境变量驱动的配置集中在这里；其余代码读 `settings.foo`，不调用
`os.getenv`。默认值对齐 Docker 镜像布局（`/opt/bindcraft`、`/opt/venv`）。
"""

from __future__ import annotations

from pathlib import Path

from bioq_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class Bindcraft2Settings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="BINDCRAFT2_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # 覆盖框架默认（/data/jobs）。
    jobs_base_dir: Path = Field(default=Path("/data/bindcraft2_jobs"))

    # 框架默认 disk_limit_mb=8000（jobs.py 在每次 submit / execute_task 前，一旦
    # jobs_base_dir 超过阈值就删掉整个 completed/failed job 目录）。一场
    # 500-trajectory campaign 的 output/ 很容易超过 8 GB，于是下一次请求会静默删掉
    # `rank` / `filter` 文档里承诺可 chain 的源 campaign（`/api/tasks/rank` 甚至可能
    # 先删掉它正要读取的目录、再 404）。这里按 NAS 容量显式放宽到 1 TiB；仍可用
    # BINDCRAFT2_DISK_LIMIT_MB 覆盖。
    disk_limit_mb: int = Field(default=1_048_576, ge=100)

    # 上游源码树，同时是子进程 cwd。**必须存在**只因为 Popen(cwd=...) 要求它是目录；
    # 上游的 settings/ 与 preset 树用 Path(__file__).parent.parent 绝对定位
    # （upstream/bindcraft/settings.py:41,353），不参与 cwd 解析——所以指到另一个
    # 存在目录在功能上无害，保留 /opt/bindcraft 只是与镜像布局一致。
    root: Path = Field(default=Path("/opt/bindcraft"))

    # 解释器。离线测试指向 tests/data/fake_bindcraft.sh。
    python: str = Field(default="/opt/venv/bin/python")

    # 上游 CLI 模块。置空则只执行 `python <args>`（离线 stub 用；上游改名时也可用）。
    module: str = Field(default="bindcraft.cli")

    # AlphaFold 参数目录：其下 params/params_<model>.npz 或 params_<model>.npz。
    # 必须显式设置——否则上游会去下载 5.3 GB。见 fetch_weights.sh / 设计文档 P1。
    alphafold_params_dir: Path = Field(default=Path("/data/models/bindcraft2/alphafold"))

    # 上游源码中的 ProteinMPNN 权重根（随包，26 MB/变体）。
    shipped_weights_dir: Path = Field(default=Path("/opt/bindcraft"))

    # JAX 编译缓存。镜像内没有 nvidia-smi，不设它上游会把图缓存丢进 /tmp，
    # 每次冷启动重付编译成本。
    compile_cache_dir: Path = Field(default=Path("/data/models/bindcraft2/xla_cache"))

    # BC2 会吃掉所有可见卡并按显存 pack worker；FC 一实例一卡，必须独占。
    max_concurrent_jobs: int = Field(default=1, ge=1, le=4)
    workers_per_gpu: int = Field(default=1, ge=1, le=8)
    max_workers_per_gpu: int = Field(default=1, ge=1, le=8)

    # max_trajectories 未由调用方指定时的兜底（上游对尝试次数无上限）。
    default_max_trajectories: int = Field(default=500, ge=1)

    # /healthz/detail 的 GPU 探针缓存秒数。0 = 每次探测。
    gpu_probe_ttl_seconds: int = Field(default=300, ge=0)
