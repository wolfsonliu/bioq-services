"""Env-driven config for flowmol-server.

All values via pydantic-settings (no `os.getenv`).  Env vars use the
`FLOWMOL_` prefix.
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class FlowMolSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="FLOWMOL_",
        env_file=".env",
        extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/flowmol_jobs"))

    root: Path = Field(
        default=Path("/opt/flowmol"),
        description="Service root (subprocess cwd — upstream module lookups "
        "resolve `flowmol/trained_models/` paths relative to package __file__, "
        "but wrapper passes an absolute --model-dir so cwd is safe anywhere).",
    )

    python: str = Field(
        default="/opt/conda/envs/flowmol/bin/python",
        description="Python interpreter inside the conda env.",
    )

    inference_script: str = Field(
        default="/opt/flowmol/server/inference.py",
        description="Service wrapper that imports flowmol.* and exposes a "
        "clean CLI with explicit --model-dir / --output-file — bypassing "
        "upstream `flowmol.load_pretrained` which does a `wget` subprocess "
        "at runtime and cannot be used behind FC egress restrictions.",
    )

    # Pretrained checkpoints — externalized to NAS at
    # /data/models/flowmol/trained_models/<variant>/{checkpoints/last.ckpt, config.yaml}
    # (FC mount; SIF / HPC bind via apptainer).  Each variant is a directory
    # per upstream `flowmol/trained_models/readme.md`.
    # See engineering/decisions/2026-06-26-service-weights-externalization.md.
    weights_dir: Path = Field(
        default=Path("/data/models/flowmol"),
        description="Root of NAS-mounted weights. Expected layout: "
        "<weights_dir>/trained_models/<variant>/{checkpoints/last.ckpt, config.yaml}.",
    )

    default_variant: str = Field(
        default="flowmol3",
        description="Fallback model variant if request does not specify one; "
        "matches the pydantic default in models.GenerateRequest.",
    )

    # Single-GPU FC instances run jobs serially.
    max_concurrent_jobs: int = Field(default=1, ge=1, le=4)
