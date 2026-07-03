"""Build the YAML config dicts consumed by upstream
``generate_molecules.py`` and ``generate_optimize.py``.

Each builder is a pure function of the request + job-local paths + settings
so it can be unit-tested without launching a subprocess.  The endpoint
writes the resulting dict to ``<job_dir>/input/config.yml`` before the
argv build in ``tools.py``.

Upstream reads these keys via ``Hparams.load_yaml()`` + ``.get(key, default)``;
``None``-valued keys are omitted so the upstream default kicks in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import (
    GenerateRequest,
    GenerateSpatialRequest,
    MolFilterParams,
    OptimizeRequest,
)
from .settings import DrughiveSettings


def _filter_dict(mf: MolFilterParams) -> dict[str, Any]:
    """Only emit non-None mol-filter keys so upstream default handling wins."""
    out: dict[str, Any] = {}
    if mf.ring_sizes is not None:
        out["ring_sizes"] = mf.ring_sizes
    if mf.ring_system_max is not None:
        out["ring_system_max"] = mf.ring_system_max
    if mf.ring_loops_max is not None:
        out["ring_loops_max"] = mf.ring_loops_max
    if mf.dbl_bond_pairs is not None:
        out["dbl_bond_pairs"] = mf.dbl_bond_pairs
    if mf.n_atoms_min is not None:
        out["n_atoms_min"] = mf.n_atoms_min
    return out


def _base_generate_dict(
    *,
    req: GenerateRequest,
    target_path: Path,
    ligand_path: Path,
    output_dir: Path,
    settings: DrughiveSettings,
) -> dict[str, Any]:
    """Shared skeleton for /api/generate and /api/generate_spatial."""
    cfg: dict[str, Any] = {
        "target_path": str(target_path),
        "ligand_path": str(ligand_path),
        "pdb_id": req.pdb_id,
        "output": str(output_dir),
        "n_samples": req.n_samples,
        "random_rotate": req.random_rotate,
        "random_translate": req.random_translate,
        "zbetas": req.zbetas,
        "temps": req.temps,
        "checkpoint": str(settings.checkpoint_path),
        "model_id": settings.model_id,
        "ffopt_mols": req.ffopt_mols,
    }
    cfg.update(_filter_dict(req.mol_filter))
    return cfg


def build_generate_config(
    *,
    req: GenerateRequest,
    target_path: Path,
    ligand_path: Path,
    output_dir: Path,
    settings: DrughiveSettings,
) -> dict[str, Any]:
    """De novo mode — upstream picks MolGenerator when no substruct_modify_*."""
    return _base_generate_dict(
        req=req,
        target_path=target_path,
        ligand_path=ligand_path,
        output_dir=output_dir,
        settings=settings,
    )


def build_generate_spatial_config(
    *,
    req: GenerateSpatialRequest,
    target_path: Path,
    ligand_path: Path,
    output_dir: Path,
    settings: DrughiveSettings,
    substruct_modify_path: Path | None = None,
) -> dict[str, Any]:
    """Scaffold hopping mode — upstream picks MolGeneratorSpatial when
    ``substruct_modify_path`` OR ``substruct_modify_pattern`` is set.

    Exactly one of ``substruct_modify_path`` (from an uploaded / URI-fetched
    SDF) or ``req.substruct_modify_pattern`` (SMILES/SMARTS) must be non-null;
    the endpoint / model_validator enforces this before we get here.
    """
    cfg = _base_generate_dict(
        req=req,
        target_path=target_path,
        ligand_path=ligand_path,
        output_dir=output_dir,
        settings=settings,
    )
    if substruct_modify_path is not None:
        cfg["substruct_modify_path"] = str(substruct_modify_path)
    if req.substruct_modify_pattern is not None:
        cfg["substruct_modify_pattern"] = req.substruct_modify_pattern
    return cfg


def build_optimize_config(
    *,
    req: OptimizeRequest,
    target_path: Path,
    ligand_path: Path,
    target_pdbqt_path: Path | None,
    output_dir: Path,
    settings: DrughiveSettings,
) -> dict[str, Any]:
    """Multi-cycle QVina2 optimization config.

    Upstream will expand scalar zbetas/temps to ``[value] * n_cycles``
    internally, so we can pass either form.
    """
    cfg: dict[str, Any] = {
        "target_path": str(target_path),
        "ligand_path": str(ligand_path),
        "pdb_id": req.pdb_id,
        "output": str(output_dir),
        "save_name": req.save_name,
        "key_opt": req.key_opt,
        "opt_increase": req.opt_increase,
        "n_cycles": req.n_cycles,
        "n_samples_initial": req.n_samples_initial,
        "n_samples": req.n_samples,
        "n_best_parents": req.n_best_parents,
        "affinity_quantile_thresh": req.affinity_quantile_thresh,
        "cluster_parents": req.cluster_parents,
        "random_rotate": req.random_rotate,
        "random_translate": req.random_translate,
        "zbetas_initial": req.zbetas_initial,
        "temps_initial": req.temps_initial,
        "zbetas": req.zbetas,
        "temps": req.temps,
        "checkpoint": str(settings.checkpoint_path),
        "model_id": settings.model_id,
        "docking_cmd": settings.docking_cmd,
        "protonate": req.protonate,
    }
    if target_pdbqt_path is not None:
        cfg["target_path_pdbqt"] = str(target_pdbqt_path)
    cfg.update(_filter_dict(req.mol_filter))
    return cfg


__all__ = [
    "build_generate_config",
    "build_generate_spatial_config",
    "build_optimize_config",
]
