"""Per-endpoint pydantic request models.  Re-export framework's JobInfo."""

from __future__ import annotations

import re
from typing import Optional

from bioq_service import FailureKind, JobInfo, JobStatus  # noqa: F401
from pydantic import BaseModel, Field, model_validator


_COMPLEX_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class DockRequest(BaseModel):
    """DiffDock-L single protein-ligand docking (/api/dock).

    Three-way input on the protein side (mutex): ``protein`` upload,
    ``protein_uri`` (job/oss/file/http scheme), or ``protein_sequence``
    (str, will be folded by ESMFold on the server).  Same on the ligand
    side: ``ligand`` upload, ``ligand_uri``, or ``ligand_description``
    (SMILES or SMARTS parsable by RDKit).

    The multipart ``protein`` / ``ligand`` uploads live on the FastAPI
    endpoint signature; this model captures everything else, plus URI /
    sequence / SMILES text fields.  The endpoint validates the file-vs-
    text mutex after resolving the multipart form.
    """

    protein_uri: Optional[str] = Field(
        default=None,
        description="URI reference to a PDB file (scheme in {job, oss, file, "
        "http(s)}).  Mutex with ``protein`` upload and ``protein_sequence``.",
    )
    protein_sequence: Optional[str] = Field(
        default=None,
        min_length=20,
        max_length=1500 * 6,  # allow multi-chain with `:` separator
        description="Protein amino-acid sequence (single chain, or multi-"
        "chain separated by ``:``).  When present, upstream folds it with "
        "ESMFold v1 before docking (adds ~5 GB weight + ~30 s cold GPU "
        "warmup).  Mutex with ``protein`` upload and ``protein_uri``.",
    )

    ligand_uri: Optional[str] = Field(
        default=None,
        description="URI reference to a ligand file (.sdf/.mol2).  Mutex "
        "with ``ligand`` upload and ``ligand_description``.",
    )
    ligand_description: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=2000,
        description="SMILES / SMARTS string parseable by RDKit.  Mutex "
        "with ``ligand`` upload and ``ligand_uri``.",
    )

    complex_name: str = Field(
        default="complex_0",
        min_length=1,
        max_length=64,
        description="Output subdirectory name under output/.  Must match "
        "``[A-Za-z0-9_-]+`` (no slashes / spaces / dots).",
    )

    samples_per_complex: int = Field(
        default=10, ge=1, le=100,
        description="Number of reverse diffusion samples (paper peak = 40; "
        "10 is a speed/precision trade).",
    )
    inference_steps: int = Field(
        default=20, ge=10, le=40,
        description="Total denoise steps.",
    )
    actual_steps: int = Field(
        default=19, ge=1, le=40,
        description="Actual denoise steps to run (early-stop; must be "
        "<= inference_steps).",
    )
    batch_size: int = Field(
        default=10, ge=1, le=20,
        description="GPU batch size.  Reduce to 4 on OOM.",
    )
    no_final_step_noise: bool = Field(
        default=True,
        description="Skip the final noise step.  NOTE: upstream is "
        "store_true default True — request value False is IGNORED in "
        "v0.0.1.  See design doc §Risks §4.",
    )
    save_visualisation: bool = Field(
        default=False,
        description="Dump reverse-diffusion trajectory as PDB per rank "
        "(large output volume; debug only).",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed; framework fills a stable value when None.",
    )

    @model_validator(mode="after")
    def _validate(self) -> "DockRequest":
        if self.actual_steps > self.inference_steps:
            raise ValueError(
                f"actual_steps ({self.actual_steps}) must be <= "
                f"inference_steps ({self.inference_steps})"
            )
        if not _COMPLEX_NAME_RE.fullmatch(self.complex_name):
            raise ValueError(
                f"complex_name must match [A-Za-z0-9_-]+ (got "
                f"{self.complex_name!r})"
            )
        # Note: protein/ligand mutex validation happens at the endpoint
        # layer (see app.py) because file uploads are not in this model.
        # This validator only checks that URI and text are not both set
        # on the same side.
        if self.protein_uri is not None and self.protein_sequence is not None:
            raise ValueError(
                "Provide either protein_uri or protein_sequence, not both."
            )
        if self.ligand_uri is not None and self.ligand_description is not None:
            raise ValueError(
                "Provide either ligand_uri or ligand_description, not both."
            )
        return self
