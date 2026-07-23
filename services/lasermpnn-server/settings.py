"""Env-driven config for lasermpnn-server.

All values via pydantic-settings; no `os.getenv` anywhere else in this package.
Env vars use the `LASERMPNN_` prefix (e.g. `LASERMPNN_ROOT`).
"""

from __future__ import annotations

from pathlib import Path

from bioq_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class LASErMPNNSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="LASERMPNN_", env_file=".env", extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/lasermpnn_jobs"))

    # subprocess cwd + PYTHONPATH root. The upstream tree is copied here as the
    # package `LASErMPNN`, invoked with `python -m LASErMPNN.run_batch_inference`
    # (upstream uses absolute `from LASErMPNN.utils...` imports, so its parent
    # dir must be on sys.path).
    root: Path = Field(default=Path("/opt/lasermpnn"))

    # NAS weights mount. 3 LASErMPNN checkpoints (~82 MB each) + ligand encoder
    # (~12 MB) live here; not baked into the image.
    weights_dir: Path = Field(default=Path("/data/models/lasermpnn"))

    # PyTorch device string. Override with LASERMPNN_DEVICE=cpu for CPU smoke
    # tests on a GPU-less host.
    device: str = Field(default="cuda:0")

    # Single-GPU serial by default.
    max_concurrent_jobs: int = Field(default=1, ge=1)
