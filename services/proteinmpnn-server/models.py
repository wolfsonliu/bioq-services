"""Per-endpoint pydantic request models for proteinmpnn-server.

A `_ProteinMPNNCommon` base captures the fields every endpoint shares
(model_variant / model_name / seed / batch_size / backbone_noise / name) and
its `model_validator(mode="after")` runs the (variant, model_name) cross-field validation.
"""

from __future__ import annotations

from typing import Literal, Optional

from bioagent_service import FailureKind, JobInfo, JobStatus  # noqa: F401 (re-exports)
from pydantic import BaseModel, Field, model_validator

__all__ = [
    "DesignRequest",
    "FailureKind",
    "JobInfo",
    "JobStatus",
    "ProbsRequest",
    "ScoreRequest",
]


_VANILLA_MODEL_NAMES = {"v_48_002", "v_48_010", "v_48_020", "v_48_030"}
_CA_MODEL_NAMES = {"v_48_002", "v_48_010", "v_48_020"}
_ABMPNN_MODEL_NAMES = {"abmpnn"}


class _ProteinMPNNCommon(BaseModel):
    """Fields shared across /api/design, /api/score, /api/probs."""

    name: str = Field(
        default="run",
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$",
        description=(
            "Output basename. Restricted to [A-Za-z0-9_.-] (no slashes/spaces) "
            "because it becomes the input PDB filename and ProteinMPNN derives the "
            "output FASTA name from the PDB stem, so `name=foo` → `seqs/foo.fa`."
        ),
    )
    model_variant: Literal["vanilla", "soluble", "ca_only", "abmpnn"] = "vanilla"
    model_name: str = Field(default="v_48_020")
    seed: int = Field(default=0, ge=0)
    batch_size: int = Field(default=1, ge=1, le=64)
    backbone_noise: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_variant_combo(self) -> "_ProteinMPNNCommon":
        if self.model_variant == "abmpnn" and self.model_name not in _ABMPNN_MODEL_NAMES:
            raise ValueError(
                "model_variant='abmpnn' requires model_name='abmpnn'; "
                f"got {self.model_name!r}",
            )
        if self.model_variant == "ca_only" and self.model_name not in _CA_MODEL_NAMES:
            raise ValueError(
                "model_variant='ca_only' supports model_name in "
                f"{sorted(_CA_MODEL_NAMES)}; got {self.model_name!r}",
            )
        if self.model_variant in ("vanilla", "soluble") and self.model_name not in _VANILLA_MODEL_NAMES:
            raise ValueError(
                f"model_variant={self.model_variant!r} supports model_name in "
                f"{sorted(_VANILLA_MODEL_NAMES)}; got {self.model_name!r}",
            )
        return self


class DesignRequest(_ProteinMPNNCommon):
    """`POST /api/design` — sequence design (FASTA output)."""

    num_seq_per_target: int = Field(default=8, ge=1, le=10000)
    sampling_temp: str = Field(
        default="0.1",
        description="Space-separated list of temperatures, e.g. '0.1 0.2 0.3'.",
    )
    chains_to_design: Optional[str] = Field(
        default=None,
        description="Space-separated chain IDs, e.g. 'A C'. Omit to design all chains.",
    )
    fixed_positions: Optional[str] = Field(
        default=None,
        description="Per-chain residue indices to fix, e.g. '1 2 3, 10 11' (segments aligned to chains_to_design).",
    )
    tied_positions: Optional[str] = Field(default=None)
    homooligomer: bool = Field(default=False)
    bias_AA: Optional[dict[str, float]] = Field(default=None)
    bias_by_res: Optional[dict] = Field(default=None)
    omit_AAs: str = Field(default="X")
    omit_AA_per_chain: Optional[dict] = Field(default=None)

    @model_validator(mode="after")
    def _validate_design_fields(self) -> "DesignRequest":
        if self.bias_AA:
            for k in self.bias_AA:
                if len(k) != 1 or not k.isalpha():
                    raise ValueError(f"bias_AA keys must be single-letter AAs; got {k!r}")
        # fixed_positions / tied_positions require chains_to_design (without it
        # the upstream helper silently produces an empty dict and the user's
        # input is lost).
        if (self.fixed_positions or self.tied_positions) and not self.chains_to_design:
            raise ValueError(
                "fixed_positions / tied_positions require chains_to_design "
                "to be set; otherwise upstream helper silently drops the input",
            )
        # fixed_positions / tied_positions segment count must match chains_to_design.
        if self.chains_to_design:
            n_chains = len(self.chains_to_design.split())
            for label, val in (("fixed_positions", self.fixed_positions),
                                ("tied_positions", self.tied_positions)):
                if val:
                    n_segs = len(val.split(","))
                    if n_segs != n_chains:
                        raise ValueError(
                            f"{label} has {n_segs} segments but chains_to_design has "
                            f"{n_chains} chains; they must match",
                        )
        return self


class ScoreRequest(_ProteinMPNNCommon):
    """`POST /api/score` — score a (structure, sequence) pair via --score_only 1."""

    num_seq_per_target: int = Field(default=10, ge=1, le=10000)
    sampling_temp: str = Field(default="0.1")
    chains_to_design: Optional[str] = Field(default=None)
    save_score: bool = Field(default=True)


class ProbsRequest(_ProteinMPNNCommon):
    """`POST /api/probs` — output per-residue AA probabilities."""

    kind: Literal["conditional", "conditional_backbone", "unconditional"] = "conditional"
    save_probs: bool = Field(default=True)
    chains_to_design: Optional[str] = Field(default=None)
