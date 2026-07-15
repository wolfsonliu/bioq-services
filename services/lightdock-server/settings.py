"""Env-driven config for lightdock-server.

All values via pydantic-settings; env_prefix=`LIGHTDOCK_`. No `os.getenv`.

LightDock ships no NN weights — its scoring parameters travel inside the pip
package. There is therefore no NAS-mounted weight and no fetch_weights.sh; the
`/healthz/detail` probe reports the installed version + available scoring
functions rather than `weights_loaded`.
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class LightdockSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="LIGHTDOCK_",
        env_file=".env",
        extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/lightdock_jobs"))

    root: Path = Field(
        default=Path("/opt/lightdock"),
        description="Service root (parent of upstream/ + server/).",
    )

    python: str = Field(
        default="/opt/lightdock/.venv/bin/python",
        description="venv interpreter that has lightdock + its lgd_* console scripts installed.",
    )

    driver_script: str = Field(
        default="/opt/lightdock/server/docking.py",
        description="Wrapper we subprocess into; orchestrates the lgd_* console scripts.",
    )

    bin_dir: Path = Field(
        default=Path("/opt/lightdock/.venv/bin"),
        description="Directory holding the lgd_* console scripts (installed with the .py suffix).",
    )

    default_scoring: str = Field(
        default="fastdfire",
        description="Default LightDock scoring function when the request omits one.",
    )

    default_glowworms: int = Field(default=200, ge=1, le=500)
    default_steps: int = Field(default=100, ge=1, le=1000)
    default_top: int = Field(default=10, ge=1, le=200)

    default_cores: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Default multiprocessing cores for the GSO run (matches an 8 vCPU FC instance).",
    )

    # LightDock is CPU-only and a single docking run's multiprocessing already
    # saturates the box, so serialize jobs by default.
    max_concurrent_jobs: int = Field(default=1, ge=1, le=8)
