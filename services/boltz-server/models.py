"""Per-endpoint pydantic request models for boltz-server.

Two endpoints share the `_BoltzCommon` base:
  * `PredictStructureRequest` — complex structure prediction (no affinity)
  * `PredictAffinityRequest`  — structure + binding affinity

The structured `sequences` / `constraints` / `templates` fields mirror the
upstream Boltz YAML schema; `tools.build_yaml` renders them into the YAML the
`boltz predict` CLI ingests. For advanced features not covered by the
structured schema, `raw_yaml` / `raw_yaml_uri` provide an escape hatch.

Per-chain MSA control lives on `SequenceEntry.msa_uri`; the special string
`"empty"` triggers single-sequence mode (matches Boltz YAML convention).
Multipart `msa_files` uploads are matched by filename stem == chain id in the
endpoint handler, *not* declared here (UploadFile-in-nested-BaseModel doesn't
flow through multipart).
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from bioq_service import FailureKind, JobInfo, JobStatus  # noqa: F401
from pydantic import BaseModel, Field, field_validator, model_validator

# ---- Constants / enums ----

MsaMode = Literal["auto", "provided", "empty"]
OutputFormat = Literal["pdb", "mmcif"]
SequenceType = Literal["protein", "dna", "rna", "ligand"]
MsaPairingStrategy = Literal["greedy", "complete"]


# ---- Nested entries ----

class Modification(BaseModel):
    """A modified residue inside a protein/dna/rna chain."""

    position: int = Field(ge=1, description="Residue index, 1-based.")
    ccd: str = Field(description="CCD code of the modified residue.")


class SequenceEntry(BaseModel):
    """One entry in the Boltz YAML `sequences:` list.

    Polymer chains (protein/dna/rna) require `sequence`. Ligand chains require
    exactly one of `smiles` or `ccd`. The `id` can be a single string or a list
    of strings for symmetric multimers.

    `msa_uri` controls MSA for protein chains:
      * `None` (default): pick up the global `msa_mode` from the request
      * `"empty"`: force single-sequence mode for this chain only
      * any other URI (file://, job://, oss://, http(s)://) or `"<chain_id>.a3m"`
        to match a multipart-uploaded `msa_files` entry by filename
    """

    type: SequenceType
    id: Union[str, list[str]]
    sequence: Optional[str] = None
    smiles: Optional[str] = None
    ccd: Optional[str] = None
    msa_uri: Optional[str] = None
    cyclic: bool = False
    modifications: list[Modification] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistent(self) -> "SequenceEntry":
        if self.type == "ligand":
            if bool(self.smiles) == bool(self.ccd):
                raise ValueError(
                    f"ligand entry id={self.id!r} requires exactly one of `smiles` or `ccd`"
                )
            if self.sequence is not None:
                raise ValueError(
                    f"ligand entry id={self.id!r} must not have `sequence`"
                )
            if self.modifications:
                raise ValueError(
                    f"ligand entry id={self.id!r} cannot have `modifications`"
                )
            if self.msa_uri is not None:
                raise ValueError(
                    f"ligand entry id={self.id!r} cannot have `msa_uri`"
                )
        else:
            if not self.sequence:
                raise ValueError(
                    f"{self.type} entry id={self.id!r} requires `sequence`"
                )
            if self.smiles or self.ccd:
                raise ValueError(
                    f"{self.type} entry id={self.id!r} cannot have `smiles` or `ccd`"
                )
            if self.type != "protein" and self.msa_uri is not None:
                raise ValueError(
                    f"only protein entries support `msa_uri`; got type={self.type!r}"
                )
        return self


# ---- Constraints (discriminated union) ----

class BondConstraint(BaseModel):
    """Covalent bond between two atoms (CCD ligands + canonical residues only)."""

    kind: Literal["bond"] = "bond"
    atom1: tuple[str, int, str] = Field(description="[chain_id, res_idx, atom_name]")
    atom2: tuple[str, int, str] = Field(description="[chain_id, res_idx, atom_name]")


class PocketConstraint(BaseModel):
    """Binding-pocket constraint between a binder chain and contact residues/atoms."""

    kind: Literal["pocket"] = "pocket"
    binder: str
    contacts: list[tuple[str, Union[int, str]]] = Field(
        description="List of [chain_id, res_idx_or_atom_name] tuples."
    )
    max_distance: float = Field(default=6.0, ge=4.0, le=20.0)
    force: bool = False


class ContactConstraint(BaseModel):
    """Pairwise contact constraint between two residues/atoms."""

    kind: Literal["contact"] = "contact"
    token1: tuple[str, Union[int, str]]
    token2: tuple[str, Union[int, str]]
    max_distance: float = Field(default=6.0, ge=4.0, le=20.0)
    force: bool = False


ConstraintEntry = Annotated[
    Union[BondConstraint, PocketConstraint, ContactConstraint],
    Field(discriminator="kind"),
]


# ---- Templates ----

class TemplateEntry(BaseModel):
    """Structural template (CIF or PDB) for one or more protein chains."""

    cif_uri: Optional[str] = None
    pdb_uri: Optional[str] = None
    chain_id: Optional[Union[str, list[str]]] = None
    template_id: Optional[Union[str, list[str]]] = None
    force: bool = False
    threshold: Optional[float] = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _check_exactly_one_source(self) -> "TemplateEntry":
        if bool(self.cif_uri) == bool(self.pdb_uri):
            raise ValueError("TemplateEntry requires exactly one of `cif_uri` or `pdb_uri`")
        if self.force and self.threshold is None:
            raise ValueError("`force=True` requires `threshold` (Å)")
        return self


# ---- Common base for both endpoints ----

class _BoltzCommon(BaseModel):
    """Fields shared by `/api/predict_structure` and `/api/predict_affinity`."""

    name: str = Field(
        default="run",
        pattern=r"^[A-Za-z0-9_\-]{1,64}$",
        description="Output subdirectory name. Stays inside the job dir.",
    )

    # Structured input. Mutually exclusive with raw_yaml.
    sequences: list[SequenceEntry] = Field(default_factory=list)
    constraints: list[ConstraintEntry] = Field(default_factory=list)
    templates: list[TemplateEntry] = Field(default_factory=list)

    # Escape hatch: caller-supplied raw YAML. Mutually exclusive with `sequences`.
    raw_yaml: Optional[str] = Field(
        default=None,
        description="Full upstream Boltz YAML document. When set, structured fields must be empty.",
    )
    raw_yaml_uri: Optional[str] = Field(
        default=None,
        description="URI pointing to a YAML file (job://, file://, oss://, http(s)://).",
    )

    # MSA strategy.
    msa_mode: MsaMode = "auto"
    msa_server_url: str = "https://api.colabfold.com"
    msa_pairing_strategy: MsaPairingStrategy = "greedy"

    # Inference knobs (boltz predict flags).
    seed: Optional[int] = None
    recycling_steps: int = Field(default=3, ge=1, le=20)
    sampling_steps: int = Field(default=200, ge=10, le=1000)
    diffusion_samples: int = Field(default=1, ge=1, le=100)
    step_scale: Optional[float] = Field(default=None, ge=0.5, le=5.0)
    output_format: OutputFormat = "mmcif"
    use_potentials: bool = False
    write_full_pae: bool = False
    write_full_pde: bool = False
    no_kernels: bool = False

    @model_validator(mode="after")
    def _check_input_mutex(self) -> "_BoltzCommon":
        has_raw = bool(self.raw_yaml) or bool(self.raw_yaml_uri)
        has_structured = bool(self.sequences)
        if has_raw and has_structured:
            raise ValueError(
                "raw_yaml/raw_yaml_uri and structured `sequences` are mutually exclusive"
            )
        if has_raw and (self.constraints or self.templates):
            raise ValueError(
                "raw_yaml/raw_yaml_uri cannot be combined with `constraints` or `templates`"
            )
        if not has_raw and not has_structured:
            raise ValueError(
                "must supply either `sequences` (structured) or `raw_yaml`/`raw_yaml_uri`"
            )
        if self.raw_yaml and self.raw_yaml_uri:
            raise ValueError("`raw_yaml` and `raw_yaml_uri` are mutually exclusive")
        return self

    @model_validator(mode="after")
    def _check_msa_consistency(self) -> "_BoltzCommon":
        """Surface obvious msa_mode/per-chain msa_uri mismatches early as 422."""
        if not self.sequences:
            return self
        proteins = [s for s in self.sequences if s.type == "protein"]
        # In `provided` mode each protein must have an msa_uri (server-side or
        # `empty`). In `empty` mode we still allow msa_uri=empty (idempotent)
        # but warn-via-validate if a real path slips in.
        if self.msa_mode == "provided":
            missing = [s.id for s in proteins if not s.msa_uri]
            if missing:
                raise ValueError(
                    f"msa_mode='provided' requires `msa_uri` on every protein entry; "
                    f"missing on: {missing}"
                )
        if self.msa_mode == "empty":
            non_empty = [s.id for s in proteins if s.msa_uri and s.msa_uri != "empty"]
            if non_empty:
                raise ValueError(
                    f"msa_mode='empty' forbids real msa_uri on protein entries; "
                    f"got non-empty on: {non_empty}"
                )
        return self


# ---- Endpoint-specific requests ----

class PredictStructureRequest(_BoltzCommon):
    """Request body for `/api/predict_structure`.

    Identical to `_BoltzCommon` — split out for FastAPI schema clarity and to
    make future divergence cheap (e.g., disallowing `properties` in raw_yaml).
    """


class PredictAffinityRequest(_BoltzCommon):
    """Request body for `/api/predict_affinity`."""

    binder_id: str = Field(
        description="Chain id of the ligand to compute affinity for. Must match a `type=ligand` SequenceEntry."
    )
    affinity_mw_correction: bool = False
    sampling_steps_affinity: int = Field(default=200, ge=10, le=1000)
    diffusion_samples_affinity: int = Field(default=5, ge=1, le=50)

    @field_validator("binder_id")
    @classmethod
    def _check_binder_id_shape(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("binder_id must be a non-empty chain id")
        return value

    @model_validator(mode="after")
    def _check_binder_refers_to_ligand(self) -> "PredictAffinityRequest":
        # raw_yaml path: caller is responsible for `properties.affinity.binder`
        # consistency; we can't validate without parsing their YAML.
        if not self.sequences:
            return self
        for entry in self.sequences:
            ids = entry.id if isinstance(entry.id, list) else [entry.id]
            if self.binder_id in ids:
                if entry.type != "ligand":
                    raise ValueError(
                        f"binder_id={self.binder_id!r} refers to a {entry.type} chain; "
                        "must be a ligand"
                    )
                return self
        raise ValueError(
            f"binder_id={self.binder_id!r} not found in any SequenceEntry"
        )


__all__ = [
    "BondConstraint",
    "ConstraintEntry",
    "ContactConstraint",
    "FailureKind",
    "JobInfo",
    "JobStatus",
    "Modification",
    "MsaMode",
    "MsaPairingStrategy",
    "OutputFormat",
    "PocketConstraint",
    "PredictAffinityRequest",
    "PredictStructureRequest",
    "SequenceEntry",
    "SequenceType",
    "TemplateEntry",
]
