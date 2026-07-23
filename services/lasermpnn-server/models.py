"""Per-endpoint pydantic request models for lasermpnn-server.

`_DesignCommon` captures the fields shared by both endpoints (batch sizes,
sampling temperatures, first-shell knobs, boolean toggles). `DesignRequest`
adds the LASErMPNN weight selector + ALA/GLY budget; `DesignLigandMPNNRequest`
is the retrained-LigandMPNN variant (fixed weights, no `-c` budget).
"""

from __future__ import annotations

from typing import Literal, Optional

from bioq_service import FailureKind, JobInfo, JobStatus  # noqa: F401 (re-exports)
from pydantic import BaseModel, Field

__all__ = [
    "DesignRequest",
    "DesignLigandMPNNRequest",
    "FailureKind",
    "JobInfo",
    "JobStatus",
]

# LASErMPNN checkpoint filenames on the NAS weights mount.
MODEL_VARIANT_WEIGHTS: dict[str, str] = {
    "nothing_heldout": "laser_weights_0p1A_nothing_heldout.pt",
    "ligandmpnn_split": "laser_weights_0p1A_noise_ligandmpnn_split.pt",
    "soluble": "soluble_weights_no_heldout_drop_clusters_optstep_65000.pt",
}

# Retrained-LigandMPNN endpoint always uses the ligandmpnn-split checkpoint.
LIGANDMPNN_WEIGHT = "laser_weights_0p1A_noise_ligandmpnn_split.pt"


class _DesignCommon(BaseModel):
    """Fields shared across /api/design and /api/design_ligandmpnn."""

    designs_per_input: int = Field(
        default=4, ge=1, le=1000,
        description="Number of designs to generate per input structure.",
    )
    designs_per_batch: int = Field(
        default=30, ge=1, le=200,
        description="Designs per GPU pass; lower it if you hit OOM.",
    )
    sequence_temp: Optional[float] = Field(
        default=None, ge=0.0, le=10.0,
        description="Sequence sampling temperature.",
    )
    first_shell_sequence_temp: Optional[float] = Field(
        default=None, ge=0.0, le=10.0,
        description="Separate temperature for ligand first-shell residues.",
    )
    chi_temp: Optional[float] = Field(
        default=None, ge=0.0, le=10.0,
        description="Side-chain chi-angle sampling temperature.",
    )
    disabled_residues: str = Field(
        default="X,C",
        description="Comma-separated residues never sampled (e.g. 'X,C').",
    )
    fix_beta: bool = Field(
        default=False,
        description="Residues with B-factor=1.0 keep their sequence + rotamer fixed.",
    )
    repack_only_input_sequence: bool = Field(
        default=False,
        description="Repack side chains without changing the sequence.",
    )
    ignore_ligand: bool = Field(
        default=False,
        description="Ignore the ligand (falls back to unconditioned design).",
    )
    use_water: bool = Field(
        default=False,
        description="Parse waters (resname HOH) as part of the ligand.",
    )
    noncanonical_aa_ligand: bool = Field(
        default=False,
        description="Featurize a non-canonical amino acid as a ligand.",
    )
    output_fasta: bool = Field(
        default=True,
        description="Also write designs.fasta alongside the PDB files.",
    )
    fs_calc_ca_distance: float = Field(
        default=10.0, ge=0.0, le=100.0,
        description="CA-to-ligand distance (A) for first-shell selection.",
    )
    fs_no_calc_burial: bool = Field(
        default=False,
        description="Use only distance (skip burial) for first-shell selection.",
    )
    disable_charged_fs: bool = Field(
        default=False,
        description="Never sample D,K,R,E in the ligand first shell.",
    )


class DesignRequest(_DesignCommon):
    """`POST /api/design` — LASErMPNN batch design (run_batch_inference)."""

    model_variant: Literal["nothing_heldout", "ligandmpnn_split", "soluble"] = Field(
        default="nothing_heldout",
        description="Which LASErMPNN checkpoint to use.",
    )
    constrain_ala_gly: bool = Field(
        default=False,
        description=(
            "Constrain ALA/GLY counts in exposed non-secondary-structure residues "
            "(upstream -c flag)."
        ),
    )
    ala_budget: int = Field(
        default=4, ge=0,
        description="Max ALA residues in the budget region (constrain_ala_gly only).",
    )
    gly_budget: int = Field(
        default=0, ge=0,
        description="Max GLY residues in the budget region (constrain_ala_gly only).",
    )


class DesignLigandMPNNRequest(_DesignCommon):
    """`POST /api/design_ligandmpnn` — retrained LigandMPNN variant.

    Uses run_batch_inference_ligandmpnn with the ligandmpnn-split checkpoint.
    Upstream's default disabled_residues for this script is 'X' (not 'X,C').
    """

    disabled_residues: str = Field(default="X")
