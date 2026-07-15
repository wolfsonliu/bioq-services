"""Per-endpoint pydantic request model for plip-server.

`ProfileRequest` carries every `plip` CLI knob we expose. File inputs
(`input_pdb` / `input_pdb_uri`) are parsed at the route layer via
`File(...)` / `Form(...)`, not on this model.

Enum-like / cross-field constraints are validated here so bad values are
rejected at request parsing (HTTP 422), not deep inside the argv builder
(which would surface as a 500).
"""

from __future__ import annotations

from typing import List, Optional

from bioagent_service import FailureKind, JobInfo, JobStatus  # noqa: F401 (re-exports)
from pydantic import BaseModel, Field, field_validator, model_validator

__all__ = [
    "FailureKind",
    "JobInfo",
    "JobStatus",
    "ProfileRequest",
]

# PLIP detection modes. `default` = automatic ligand detection (no extra flag);
# the others map to mutually-exclusive upstream flags.
_MODES = ("default", "peptide", "intra", "dnareceptor")

# Report output formats we allow (subset of PLIP's; stdout/gzip out of scope).
_FORMATS = ("xml", "txt")

_NAME_PATTERN = r"^[A-Za-z0-9_.-]+$"
_CHAIN_PATTERN = r"^[A-Za-z0-9]{1,4}$"


class ProfileRequest(BaseModel):
    """`POST /api/profile` — profile interactions in one PDB complex."""

    mode: str = Field(
        default="default",
        description=(
            "Detection mode: default (auto ligand detection) / peptide "
            "(--peptides, inter-chain) / intra (--intra, intra-chain) / "
            "dnareceptor (treat nucleic acids as receptor)."
        ),
    )
    peptide_chains: List[str] = Field(
        default_factory=list,
        description="mode=peptide: chain IDs treated as peptide ligands (--peptides A B).",
    )
    intra_chain: Optional[str] = Field(
        default=None,
        description="mode=intra: single chain to analyze for intra-chain contacts (--intra A).",
    )
    report_formats: List[str] = Field(
        default_factory=lambda: ["xml", "txt"],
        description="Report formats to emit: any non-empty subset of {xml, txt}.",
    )
    pymol_session: bool = Field(
        default=False,
        description="Also write a PyMOL session (.pse) per binding site (-y).",
    )
    render_images: bool = Field(
        default=False,
        description="Also render ray-traced images (.png) per binding site (-p).",
    )
    name: str = Field(
        default="report", min_length=1, max_length=64, pattern=_NAME_PATTERN,
        description="Report basename stem (output/<name>.xml / .txt).",
    )
    breakcomposite: bool = Field(
        default=False,
        description="Do not combine covalently-bound ligand fragments (--breakcomposite).",
    )
    altlocation: bool = Field(
        default=False,
        description="Keep alternate atom locations (--altlocation).",
    )
    nofix: bool = Field(
        default=False,
        description="Turn off automatic PDB fixing (--nofix).",
    )
    keepmod: bool = Field(
        default=False,
        description="Keep modified residues as ligands (--keepmod).",
    )
    nohydro: bool = Field(
        default=False,
        description="Do not add polar hydrogens (structure already protonated) (--nohydro).",
    )
    model: int = Field(
        default=1, ge=1,
        description="Model number for multi-model structures (--model).",
    )
    maxthreads: Optional[int] = Field(
        default=None, ge=1, le=128,
        description="Override PLIP_THREADS for this call (--maxthreads).",
    )

    @field_validator("mode")
    @classmethod
    def _check_mode(cls, v: str) -> str:
        if v not in _MODES:
            raise ValueError(f"invalid mode {v!r}; allowed: {', '.join(_MODES)}")
        return v

    @field_validator("report_formats")
    @classmethod
    def _check_formats(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("report_formats must not be empty")
        bad = [f for f in v if f not in _FORMATS]
        if bad:
            raise ValueError(f"invalid report_formats {bad}; allowed: {', '.join(_FORMATS)}")
        # de-dupe, preserve order
        seen: dict[str, None] = {}
        for f in v:
            seen.setdefault(f, None)
        return list(seen)

    @field_validator("peptide_chains")
    @classmethod
    def _check_chains(cls, v: List[str]) -> List[str]:
        import re
        for c in v:
            if not re.match(_CHAIN_PATTERN, c):
                raise ValueError(f"invalid chain id {c!r}; expected 1-4 alphanumerics")
        return v

    @field_validator("intra_chain")
    @classmethod
    def _check_intra_chain(cls, v: Optional[str]) -> Optional[str]:
        import re
        if v is not None and not re.match(_CHAIN_PATTERN, v):
            raise ValueError(f"invalid intra_chain {v!r}; expected 1-4 alphanumerics")
        return v

    @model_validator(mode="after")
    def _check_mode_args(self) -> "ProfileRequest":
        if self.mode == "peptide" and not self.peptide_chains:
            raise ValueError("mode=peptide requires peptide_chains")
        if self.mode == "intra" and not self.intra_chain:
            raise ValueError("mode=intra requires intra_chain")
        return self
