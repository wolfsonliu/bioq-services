"""Argv assembly for drughive-server.

Upstream ``generate_molecules.py`` and ``generate_optimize.py`` each take
a single YAML config path as their positional arg.  ``configs.py`` builds
the dict, the endpoint writes it under ``<job_dir>/input/config.yml``, and
these builders wire the path into the subprocess argv.
"""

from __future__ import annotations

from pathlib import Path

from .settings import DrughiveSettings


def _script_argv(script: Path, cfg_path: Path, settings: DrughiveSettings) -> list[str]:
    return [settings.python, str(script), str(cfg_path)]


def generate_argv(
    *, cfg_path: Path, settings: DrughiveSettings
) -> list[str]:
    """`/api/generate` and `/api/generate_spatial` — same upstream script;
    upstream routes on presence of ``substruct_modify_*`` keys in the YAML."""
    return _script_argv(settings.generate_script, cfg_path, settings)


def optimize_argv(
    *, cfg_path: Path, settings: DrughiveSettings
) -> list[str]:
    """`/api/optimize` — multi-cycle QVina2 optimization."""
    return _script_argv(settings.optimize_script, cfg_path, settings)


__all__ = ["generate_argv", "optimize_argv"]
