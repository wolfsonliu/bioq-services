"""Build the 5-block YAML config that upstream ``scripts/sample_use.py``
consumes.

Upstream reads the config via ``make_config()`` (utils/misc.py) and expects
these top-level blocks::

    sample:      # runtime controls (seed, batch_size, num_mols, save_traj_prob)
    data:        # input paths + pocket_args + pocmol_args
    transforms:  # optional feature overrides
    task:        # task definition & mode
    noise:       # diffusion noise config

Each ``build_<endpoint>_config`` is a pure function of the request + paths
+ settings so it can be unit-tested without launching a subprocess.  The
endpoint writes the resulting dict to
``<job_dir>/input/task_config.yml`` before assembling argv.

``build_model_config`` overrides the ``model.checkpoint`` field so we can
put the ckpt on NAS instead of the vendored ``data/trained_models/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import (
    ConfidenceRequest,
    ConfidenceVariant,
    DockRequest,
    LinkingRequest,
    NoiseMode,
    OptimizeRequest,
    PepDesignMode,
    PepDesignRequest,
    SbddMode,
    SbddRequest,
)
from .settings import PocketXMolSettings


# ---------------------------------------------------------------------------
# Reusable schedule primitives — the ``advance`` sigmoid schedule from
# MolDiff is what every example config uses.
# ---------------------------------------------------------------------------
def _advance_level(min_val: float = 0.0, max_val: float = 1.0) -> dict[str, Any]:
    return {
        "name": "advance",
        "min": min_val,
        "max": max_val,
        "step2level": {
            "scale_start": 0.99999,
            "scale_end": 0.00001,
            "width": 3,
        },
    }


def _sample_block(*, seed: int | None, batch_size: int, num_samples: int,
                  save_traj_prob: float = 0.0) -> dict[str, Any]:
    return {
        "seed": seed if seed is not None else 2024,
        "batch_size": batch_size,
        "num_mols": num_samples,
        "save_traj_prob": save_traj_prob,
    }


def _variable_mol_size(*, mean: int, std: int,
                       not_remove: list[int] | None = None) -> dict[str, Any]:
    """Compact ``variable_mol_size`` transform used by sbdd / linking / optimize."""
    block: dict[str, Any] = {
        "name": "variable_mol_size",
        "num_atoms_distri": {
            "strategy": "mol_atoms_based",
            "mean": {"coef": 0, "bias": mean},
            "std": {"coef": 0, "bias": std},
            "min": 5,
        },
    }
    if not_remove is not None:
        block["not_remove"] = not_remove
    return block


# ===========================================================================
# 1. Dock
# ===========================================================================
def build_dock_config(
    *,
    req: DockRequest,
    protein_path: Path,
    ligand_path: Path | None,
    ref_ligand_path: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    """Build config for /api/dock.

    Priority for the `input_ligand` field on the upstream data block:
      1. `smiles` (SMILES string) — small-molecule dock from a SMILES.
      2. `pep_sequence` (peptide sequence) — peptide dock from sequence.
      3. Uploaded `ligand` file (SDF or PDB).
    Exactly one of these must be supplied — enforced at the endpoint layer.

    ``pocket_args`` picks between explicit coord (`pocket_coord`) and
    reference-ligand-derived (`ref_ligand_path` or fall back to the input
    ligand itself).
    """
    if req.smiles is not None:
        input_ligand: str | None = req.smiles
    elif req.pep_sequence is not None:
        input_ligand = f"pepseq_{req.pep_sequence}"
    elif ligand_path is not None:
        input_ligand = str(ligand_path)
    else:
        input_ligand = None

    pocket_args: dict[str, Any] = {
        "radius": req.pocket_radius,
        "criterion": req.pocket_criterion.value,
    }
    if req.pocket_coord is not None:
        pocket_args["pocket_coord"] = req.pocket_coord
    elif ref_ligand_path is not None:
        pocket_args["ref_ligand_path"] = str(ref_ligand_path)

    settings_block = (
        {"free": 1, "flexible": 0}
        if req.noise_mode == NoiseMode.gaussian
        else {"free": 0, "flexible": 1}
    )

    return {
        "sample": _sample_block(
            seed=req.seed,
            batch_size=req.batch_size,
            num_samples=req.num_samples,
        ),
        "data": {
            "protein_path": str(protein_path),
            "input_ligand": input_ligand,
            "is_pep": req.is_pep,
            "pocket_args": pocket_args,
            "pocmol_args": {"data_id": "dock_svc"},
        },
        "transforms": {},
        "task": {
            "name": "dock",
            "transform": {
                "name": "dock",
                "settings": settings_block,
            },
        },
        "noise": {
            "name": "dock",
            "num_steps": 100,
            "prior": "from_train",
            "level": _advance_level(),
        },
    }


# ===========================================================================
# 2. SBDD
# ===========================================================================
def build_sbdd_config(
    *,
    req: SbddRequest,
    protein_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build config for /api/sbdd (de novo SBDD).

    Two internal modes:
      - ``ar`` — task.transform.name=ar, noise.name=maskfill (2 groups,
        part1/part2) with `ar_config` refinement loop; matches
        upstream ``sbdd.yml``.
      - ``simple`` — task.transform.name=sbdd, noise.name=sbdd (single
        group); matches ``sbdd_simple.yml``.  Faster.
    """
    center = req.pocket_coord

    transforms: dict[str, Any] = {
        "featurizer_pocket": {"center": center},
        "variable_mol_size": _variable_mol_size(
            mean=req.mol_size_mean, std=req.mol_size_std,
        ),
    }

    if req.mode == SbddMode.ar:
        task = {
            "name": "sbdd",
            "transform": {"name": "ar", "part1_pert": "small"},
        }
        noise = {
            "name": "maskfill",
            "num_steps": 100,
            "ar_config": {
                "strategy": "refine",
                "r": 3,
                "threshold_node": 0.98,
                "threshold_pos": 0.91,
                "threshold_bond": 0.98,
                "max_ar_step": 10,
                "change_init_step": 1,
            },
            "prior": {"part1": "from_train", "part2": "from_train"},
            "level": {
                "part1": {"name": "uniform", "min": 0.6, "max": 1.0},
                "part2": _advance_level(),
            },
        }
    else:  # simple
        task = {
            "name": "sbdd",
            "transform": {"name": "sbdd"},
        }
        noise = {
            "name": "sbdd",
            "num_steps": 100,
            "prior": "from_train",
            "level": _advance_level(),
        }

    return {
        "sample": _sample_block(
            seed=req.seed, batch_size=req.batch_size, num_samples=req.num_samples,
        ),
        "data": {
            "protein_path": str(protein_path),
            "is_pep": False,
            "pocket_args": {
                "pocket_coord": center,
                "radius": req.pocket_radius,
            },
            "pocmol_args": {"data_id": "sbdd_svc"},
        },
        "transforms": transforms,
        "task": task,
        "noise": noise,
    }


# ===========================================================================
# 3. Linking / growing
# ===========================================================================
def _linking_maskfill_noise() -> dict[str, Any]:
    """Fully explicit prior + advance level for maskfill task.

    Matches upstream ``linking_fixed_frags.yml`` / ``growing_fixed_frag.yml``
    prior blocks — the two share this exact structure.
    """
    return {
        "name": "maskfill",
        "num_steps": 100,
        "prior": {
            "part1": {
                "pos_only": True,
                "pos": {
                    "name": "allpos",
                    "pos": {"name": "gaussian_simple", "sigma_max": 1},
                    "translation": {
                        "name": "translation", "ve": False, "mean": 0, "std": 1,
                    },
                    "rotation": {"name": "rotation", "sigma_max": 0.0002},
                    "torsional": {"name": "torsional", "sigma_max": 0.2},
                },
            },
            "part2": {
                "node": {
                    "name": "categorical",
                    "prior_type": "predefined",
                    "prior_probs": [3, 2, 2, 2, 1, 1, 1, 0.3, 0.3, 0.3, 0.3, 13.2],
                },
                "pos": {
                    "name": "allpos",
                    "pos": {"name": "gaussian_simple", "sigma_max": 1},
                },
                "edge": {"name": "categorical", "prior_type": "tomask_half"},
            },
        },
        "level": {
            "part1": _advance_level(),
            "part2": _advance_level(),
        },
    }


def build_linking_config(
    *,
    req: LinkingRequest,
    protein_path: Path,
    input_ligand_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build config for /api/linking (linking / growing / PROTAC).

    Fragment atoms (union of all `fragments` groups) go into
    ``variable_mol_size.not_remove`` so upstream's variable-size sampler
    won't strip them.  The ``preset_partition.grouped_node_p1`` field
    takes the list-of-lists directly.
    """
    all_frag_atoms = sorted({a for group in req.fragments for a in group})

    transforms: dict[str, Any] = {}
    if req.use_input_center:
        transforms["featurizer"] = {"mol_as_pocket_center": True}
    transforms["variable_mol_size"] = _variable_mol_size(
        mean=req.mol_size_mean,
        std=req.mol_size_std,
        not_remove=all_frag_atoms,
    )

    settings_block: dict[str, dict[str, int]] = {
        "part1_pert": {req.part1_pert.value: 1},
        "known_anchor": {"none": 1},
    }

    return {
        "sample": _sample_block(
            seed=req.seed, batch_size=req.batch_size, num_samples=req.num_samples,
        ),
        "data": {
            "protein_path": str(protein_path),
            "input_ligand": str(input_ligand_path),
            "is_pep": False,
            "pocket_args": {"radius": req.pocket_radius},
            "pocmol_args": {"data_id": "linking_svc"},
        },
        "transforms": transforms,
        "task": {
            "name": "maskfill",
            "transform": {
                "name": "maskfill",
                "preset_partition": {"grouped_node_p1": req.fragments},
                "settings": settings_block,
            },
        },
        "noise": _linking_maskfill_noise(),
    }


# ===========================================================================
# 4. Optimize
# ===========================================================================
def build_optimize_config(
    *,
    req: OptimizeRequest,
    protein_path: Path,
    input_ligand_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build config for /api/optimize (full-molecule optimization).

    Mirrors upstream ``opt_mol.yml``: task=sbdd/simple with
    ``init_step<1`` so denoising starts from the perturbed input pose.
    """
    return {
        "sample": _sample_block(
            seed=req.seed, batch_size=req.batch_size, num_samples=req.num_samples,
        ),
        "data": {
            "protein_path": str(protein_path),
            "input_ligand": str(input_ligand_path),
            "is_pep": False,
            "pocket_args": {"radius": req.pocket_radius},
            "pocmol_args": {"data_id": "optimize_svc"},
        },
        "transforms": {
            "featurizer": {"mol_as_pocket_center": True},
            "variable_mol_size": _variable_mol_size(
                mean=req.mol_size_mean, std=req.mol_size_std,
            ),
        },
        "task": {
            "name": "sbdd",
            "transform": {"name": "sbdd"},
        },
        "noise": {
            "name": "sbdd",
            "num_steps": req.num_steps,
            "init_step": req.init_step,
            "prior": "from_train",
            "level": _advance_level(),
        },
    }


# ===========================================================================
# 5. Peptide design
# ===========================================================================
def _pepdesign_settings_block(mode: PepDesignMode) -> dict[str, Any]:
    if mode in (PepDesignMode.denovo_linear, PepDesignMode.denovo_cyclic,
                PepDesignMode.inverse_fold):
        # Both denovo variants and inverse-fold use `sc: 1` when input
        # backbone is provided, but the upstream `pepdesign_denovo.yml`
        # ships with `full: 1` and the mode is toggled via input_ligand
        # form (peplen_N vs .pdb).  Keep consistent with upstream examples.
        if mode == PepDesignMode.inverse_fold:
            return {"mode": {"full": 0, "sc": 1, "packing": 0}}
        return {"mode": {"full": 1, "sc": 0, "packing": 0}}
    # sc_pack
    return {"mode": {"full": 0, "sc": 0, "packing": 1}}


def build_pepdesign_config(
    *,
    req: PepDesignRequest,
    protein_path: Path,
    input_peptide_path: Path | None,
    ref_ligand_path: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    """Build config for /api/pepdesign.

    Peptide input encoding:
      - ``denovo_linear`` → ``input_ligand: peplen_<N>``
      - ``denovo_cyclic`` → ``input_ligand: cycpeplen_<N>``
      - ``inverse_fold`` / ``sc_pack`` → ``input_ligand: <peptide.pdb>``
    """
    if req.mode == PepDesignMode.denovo_linear:
        input_ligand: str = f"peplen_{req.pep_length}"
    elif req.mode == PepDesignMode.denovo_cyclic:
        input_ligand = f"cycpeplen_{req.pep_length}"
    else:
        # inverse_fold / sc_pack: use input peptide PDB
        if input_peptide_path is None:
            raise ValueError(
                f"input peptide PDB required for mode={req.mode.value}"
            )
        input_ligand = str(input_peptide_path)

    pocket_args: dict[str, Any] = {"radius": req.pocket_radius}
    if req.pocket_coord is not None:
        pocket_args["pocket_coord"] = req.pocket_coord
    if ref_ligand_path is not None:
        pocket_args["ref_ligand_path"] = str(ref_ligand_path)

    # `variable_sc_size` — only meaningful for full / sc modes (not packing).
    # Include `not_remove` union of any fix_pos / fix_type side-chain res.
    transforms: dict[str, Any] = {}
    if req.mode != PepDesignMode.sc_pack:
        vsc: dict[str, Any] = {
            "name": "variable_sc_size",
            "applicable_tasks": ["pepdesign"],
            "num_atoms_distri": {
                "mean": 8,
                "std": {"coef": 0.3817, "bias": 1.8727},
            },
        }
        # Note: upstream `not_remove` is atom indices, not residue.  For
        # v0.0.1 we only fill this when the caller has computed atom
        # indices themselves (not exposed yet) — leave empty.
        transforms["variable_sc_size"] = vsc

    task_transform: dict[str, Any] = {
        "name": "pepdesign",
        "settings": _pepdesign_settings_block(req.mode),
    }
    if req.fix_pos_res_bb or req.fix_pos_res_sc:
        task_transform["fix_pos"] = {
            "res_bb": req.fix_pos_res_bb,
            "res_sc": req.fix_pos_res_sc,
            "atom": [],
        }
    if req.fix_type_res_bb or req.fix_type_res_sc:
        task_transform["fix_type_only"] = {
            "res_bb": req.fix_type_res_bb,
            "res_sc": req.fix_type_res_sc,
            "atom": [],
        }

    return {
        "sample": _sample_block(
            seed=req.seed, batch_size=req.batch_size, num_samples=req.num_samples,
        ),
        "data": {
            "protein_path": str(protein_path),
            "input_ligand": input_ligand,
            "is_pep": True,
            "pocket_args": pocket_args,
            "pocmol_args": {"data_id": "pepdesign_svc"},
        },
        "transforms": transforms,
        "task": {"name": "pepdesign", "transform": task_transform},
        "noise": {
            "name": "pepdesign",
            "num_steps": 100,
            "prior": {"bb": "from_train", "sc": "from_train"},
            "level": {"bb": _advance_level(), "sc": _advance_level()},
        },
    }


# ===========================================================================
# Model config (checkpoint path override for NAS-hosted weights)
# ===========================================================================
def build_model_config(settings: PocketXMolSettings) -> dict[str, Any]:
    """Override upstream's ``configs/sample/pxm.yml`` model.checkpoint field.

    Upstream default is ``data/trained_models/pxm/checkpoints/pocketxmol.ckpt``
    (relative to CWD).  We put the ckpt on NAS at settings.pxm_checkpoint
    and pass this via ``--config_model`` — upstream's ``make_config()``
    merges task/model YAMLs so we don't need to modify the vendored
    ``pxm.yml`` file itself.
    """
    return {"model": {"checkpoint": str(settings.pxm_checkpoint)}}


def confidence_yaml_path(
    variant: ConfidenceVariant,
    settings: PocketXMolSettings,
) -> Path:
    """Return the upstream confidence YAML path for the requested variant.

    The upstream ``believe_use_pdb.py --config`` flag wants a path to one
    of ``configs/sample/confidence/{tuned_cfd,flex_cfd}.yml``.
    """
    return settings.confidence_yaml_dir / f"{variant.value}.yml"


__all__ = [
    "build_dock_config",
    "build_sbdd_config",
    "build_linking_config",
    "build_optimize_config",
    "build_pepdesign_config",
    "build_model_config",
    "confidence_yaml_path",
]


# Silence unused-import lints for the confidence request module (used at
# type-check time even though this file doesn't call it at runtime).
_ = ConfidenceRequest
