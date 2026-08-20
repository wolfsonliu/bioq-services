"""Per-endpoint pydantic request models for rfdiffusion2-server.

RFdiffusion2 (Ahern et al. 2025) is a single Hydra-driven script
(`rf_diffusion/run_inference.py`) that handles atomic-motif scaffolding and
small-molecule binder design with several conditioning flavours. We expose
the two flavours that the upstream `open_source_demo.json` documents, plus a
freeform escape hatch:

  * `/api/generate/active_site` — enzyme active-site scaffolding (atomic
    motif + ligand, with indexed or unindexed positions)
  * `/api/generate/small_molecule_binder` — protein binder around a small
    molecule, optionally RASA-conditioned
  * `/api/generate` — raw contig + extra Hydra overrides

Each typed endpoint exposes the fields that a typical user of that flavour
needs. Power users fall through to `/api/generate` and set anything in
`config/inference/base.yaml` (or any other config under `config/inference/`).
"""

from __future__ import annotations

from typing import Optional

from bioq_service import FailureKind, JobInfo, JobStatus  # re-exports
from bioq_service import default_semantics
from pydantic import BaseModel, Field

__all__ = [
    "ActiveSiteRequest",
    "CustomRequest",
    "FailureKind",
    "JobInfo",
    "JobStatus",
    "MODEL_CHOICES",
    "SmallMoleculeBinderRequest",
]


# Known checkpoint friendly-name → filename mapping. RFdiffusion2 ships only
# two diffusion checkpoints today; the default config (`aa.yaml`) points at
# RFD_140.pt. RFD_173.pt is the newer benchmark model used by
# `open_source_demo.yaml`. The LigandMPNN weights live in
# `rf_diffusion/third_party_model_weights/ligand_mpnn/` and are consumed by
# downstream pipelines, not by run_inference.py itself.
MODEL_CHOICES: dict[str, str] = {
    "rfd_140": "RFD_140.pt",
    "rfd_173": "RFD_173.pt",
}

_MODEL_DESC = (
    "Override the diffusion checkpoint via `inference.ckpt_path`. Leave unset to "
    "use the value baked into the chosen config (typically RFD_140.pt). "
    "Choices: " + ", ".join(sorted(MODEL_CHOICES))
)


class _GenerationCommon(BaseModel):
    """Fields every generation endpoint accepts."""

    num_designs: int = Field(
        default=4, ge=1, le=1000, description="Number of designs to sample."
    )
    diffuser_t: int = Field(
        default=100,
        ge=1,
        le=400,
        description=(
            "Reverse-diffusion timesteps. aa.yaml default is 100; lower values "
            "speed up sampling at the cost of quality."
        ),
    )
    final_step: int = Field(
        default=1,
        ge=1,
        description="Stop the trajectory at step N (default 1 = run to completion).",
    )
    write_trajectory: bool = Field(
        default=False,
        description="Write the full reverse-flow trajectory to disk. Default off — saves disk.",
    )
    deterministic: bool = Field(
        default=False, description="Seed every RNG so reruns reproduce exactly."
    )
    model: Optional[str] = Field(default=None, description=_MODEL_DESC, json_schema_extra=default_semantics("auto", "auto-select by the tool from the request inputs"))


class ActiveSiteRequest(_GenerationCommon):
    """Params for `POST /api/generate/active_site`.

    Reproduces the four `active_site_*` demo cases in
    `open_source_demo.json`. The motif residues whose sidechain atoms are
    anchored are passed via `contig_atoms`; whether their positions are
    fixed in the design (indexed) or chosen by the network (unindexed,
    guidepost) is controlled by `contig_as_guidepost`.
    """

    contigs: str = Field(
        description=(
            "Hydra contig string referencing residues in `input_pdb` plus stretches "
            "of length-only design, e.g. "
            "`46,A106-106,59,A166-166,2,A169-169,23,A193-193,46`. Bare integers "
            "are residues to be designed; `<chain><resnum>-<resnum>` are motif anchors."
        ),
        examples=["46,A106-106,59,A166-166,2,A169-169,23,A193-193,46"],
    )
    contig_atoms: dict[str, str] = Field(
        description=(
            "Per-motif-residue list of anchored atom names, keyed by `<chain><resnum>`. "
            "Each value is a comma-separated atom-name string. "
            "Example: `{\"A106\": \"NE,CD,CZ\", \"A166\": \"OD1,CG\"}`."
        ),
        examples=[
            {
                "A106": "NE,CD,CZ",
                "A166": "OD1,CG",
                "A169": "NH2,CZ",
                "A193": "NE2,CD2,CE1",
            }
        ],
    )
    ligand: str = Field(
        description=(
            "Comma-separated ligand residue names present in `input_pdb`, e.g. "
            "`NAD,OXM`. These atoms are kept in the scene as small molecules."
        ),
        examples=["NAD,OXM"],
    )
    contig_as_guidepost: bool = Field(
        default=True,
        description=(
            "True = unindexed atomic mode (network picks motif residue indices). "
            "False = indexed mode (motif residue indices are fixed). Upstream demos "
            "default to True for the harder unindexed variant."
        ),
    )
    only_guidepost_positions: Optional[str] = Field(
        default=None,
        description=(
            "When `contig_as_guidepost=True`, restrict the guidepost set to these "
            "residues (others remain indexed). Format: `<chain><resnum>` or comma-"
            "separated list, e.g. `A106` or `A106,A193`."
        ),
        examples=["A106"],
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    partially_fixed_ligand: Optional[dict[str, list[str]]] = Field(
        default=None,
        description=(
            "Per-ligand list of atom names to KEEP FIXED; the unlisted atoms are "
            "diffused. Format: `{\"<resname>\": [\"<atom>\", ...]}`. Example: "
            "`{\"NAD\": [\"O7N\",\"C7N\"], \"OXM\": [\"O3\",\"C2\"]}`."
        ),
        examples=[
            {
                "NAD": ["O7N", "C7N", "C3N", "N7N", "C2N", "C4N", "N1N", "C5N", "C1D"],
                "OXM": ["O3", "C2", "C1", "O2", "N1"],
            }
        ],
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    inpaint_seq: Optional[str] = Field(
        default=None,
        description=(
            "Mask the sequence at these positions (auto-switches model to InpaintSeq "
            "where supported). Format: `<chain><resnum>` / `<chain><resnum>-<resnum>`."
        ),
        examples=["A1/A30-40"],
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )


class SmallMoleculeBinderRequest(_GenerationCommon):
    """Params for `POST /api/generate/small_molecule_binder`.

    Reproduces the `small_molecule_binder_rasa_buried` demo case: generate a
    protein around a small molecule, optionally conditioned on the
    target relative-SASA value so the binder buries the ligand to the
    requested degree.
    """

    contigs: str = Field(
        description=(
            "Single bare-length contig string describing the binder, e.g. `150`. "
            "Multi-segment binders (`50/0 30`) work too."
        ),
        examples=["150"],
    )
    length: Optional[str] = Field(
        default=None,
        description=(
            "Hard length constraint, e.g. `150-150`. Useful when `contigs` is a range."
        ),
        examples=["150-150"],
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    ligand: str = Field(
        description="Ligand residue name(s) present in `input_pdb`, e.g. `PH2`.",
        examples=["PH2"],
    )
    rasa_active: bool = Field(
        default=True,
        description=(
            "Enable RASA-v2 conditioning (`inference.conditions.relative_sasa_v2.active`). "
            "When True, `rasa_target` shapes the desired solvent-exposed fraction."
        ),
    )
    rasa_target: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Target relative-SASA value in [0, 1]. 0 = ligand fully buried, "
            "1 = ligand fully exposed."
        ),
    )


class CustomRequest(_GenerationCommon):
    """Params for `POST /api/generate` — raw contig + freeform Hydra overrides."""

    contigs: str = Field(
        description="Raw contig string, passed as `contigmap.contigs=[<value>]`.",
        examples=["150-150", "46,A106-106,59"],
    )
    config_name: str = Field(
        default="aa",
        description=(
            "Hydra `--config-name`. Defaults to `aa.yaml` (the standard inference "
            "config). Use any other config under `rf_diffusion/config/inference/` "
            "for specialised flows (e.g. `aa_ppi.yaml`, `unconditional.yaml`)."
        ),
    )
    input_pdb_required: bool = Field(
        default=False,
        description=(
            "Set True for endpoints/configs that need an input PDB (motif, partial "
            "diffusion). Defaults to False so unconditional configs work without one."
        ),
    )
    ligand: Optional[str] = Field(
        default=None,
        description="`inference.ligand=...` if set (comma-separated ligand resnames).",
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    extra_overrides: Optional[str] = Field(
        default=None,
        description=(
            "JSON object mapping dotted Hydra paths → values, e.g. "
            "`{\"diffuser.T\": 50, \"inference.conditions.relative_sasa_v2.active\": true}`. "
            "Each entry is appended to argv as `key=value`."
        ),
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
