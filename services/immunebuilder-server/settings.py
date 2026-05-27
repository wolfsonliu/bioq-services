"""Env-driven config for immunebuilder-server.

All values via pydantic-settings; no `os.getenv` anywhere else in this package.
Env vars use the `IMMUNEBUILDER_` prefix (e.g. `IMMUNEBUILDER_VENV_BIN`).
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class ImmuneBuilderSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="IMMUNEBUILDER_", env_file=".env", extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/immunebuilder_jobs"))

    venv_bin: Path = Field(
        default=Path("/opt/conda/envs/immunebuilder/bin"),
        description="conda env bin/ where ABodyBuilder2/NanoBodyBuilder2/TCRBuilder2 live",
    )

    oss_region: str = Field(default="cn-hangzhou")
