"""Per-endpoint pydantic request models for openadmet-server."""

from __future__ import annotations

from typing import Literal, Optional

from bioq_service import FailureKind, JobInfo, JobStatus  # noqa: F401 (re-exported)
from bioq_service import default_semantics
from pydantic import BaseModel, Field, model_validator


AcceleratorChoice = Literal["cpu", "gpu", "auto", "mps", "tpu", "ipu"]
AcquisitionChoice = Literal["ucb", "ei", "pi"]
LabelTypeChoice = Literal["biotarget", "model", "feat", "tasks"]


class PredictRequest(BaseModel):
    """Inputs to `openadmet predict`.

    One of ``input_smiles`` / ``input_csv_uri`` / ``input_sdf_uri`` must be
    provided (or the ``input_csv``/``input_sdf`` file upload — those fields
    live outside the pydantic model on the FastAPI endpoint signature).
    """

    # ---- Input source (URI variants; UploadFile passed separately in app.py) ----
    input_smiles: Optional[str] = Field(
        default=None,
        max_length=100_000,
        description=(
            "Inline SMILES: comma- or newline-separated. Up to 200 molecules "
            "for quick probing. For larger batches use `input_csv` upload or URI."
        ),
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    input_csv_uri: Optional[str] = Field(
        default=None,
        description="URI (oss://, file://, http(s)://, job://) pointing to an input CSV.",
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    input_sdf_uri: Optional[str] = Field(
        default=None,
        description="URI pointing to an input SDF.",
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )

    input_col: Optional[str] = Field(
        default=None,
        max_length=256,
        description=(
            "Column name in the input CSV containing SMILES. If omitted, "
            "the server derives it from the first `model_names` entry's "
            "recipe_components/data.yaml::input_col (auto-derive)."
        ),
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )

    # ---- Model selection ----
    model_names: list[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description=(
            "Pre-registered model names on NAS. GET /api/models to enumerate. "
            "All requested models will be applied to the same input; the "
            "output CSV has OADMET_PRED_* columns per (model, task)."
        ),
    )

    # ---- Execution ----
    accelerator: AcceleratorChoice = Field(default="gpu")

    # ---- Active-learning acquisition (optional) ----
    aq_fxns: list[AcquisitionChoice] = Field(default_factory=list, json_schema_extra=default_semantics("unset", "empty when omitted"))
    beta: list[float] = Field(default_factory=list, json_schema_extra=default_semantics("unset", "empty when omitted"))
    best_y: list[float] = Field(default_factory=list, json_schema_extra=default_semantics("unset", "empty when omitted"))
    xi: list[float] = Field(default_factory=list, json_schema_extra=default_semantics("unset", "empty when omitted"))

    debug: bool = Field(default=False)

    @model_validator(mode="after")
    def _check_acquisition_alignment(self) -> "PredictRequest":
        # Mirror upstream openadmet.models.cli.predict._validate_aq_fxns.
        # Each aq function may only appear at most once.
        for aq in ("ucb", "ei", "pi"):
            if self.aq_fxns.count(aq) > 1:
                raise ValueError(f"Acquisition function '{aq}' can only be specified once.")

        ucb_count = self.aq_fxns.count("ucb")
        ei_pi_count = self.aq_fxns.count("ei") + self.aq_fxns.count("pi")

        if ucb_count != len(self.beta):
            raise ValueError(
                f"`beta` must be provided once per 'ucb' entry "
                f"(got {len(self.beta)} beta, {ucb_count} ucb)."
            )
        if ei_pi_count != len(self.best_y) or ei_pi_count != len(self.xi):
            raise ValueError(
                f"`best_y` and `xi` must be provided once per 'ei'+'pi' entry "
                f"(got {len(self.best_y)} best_y, {len(self.xi)} xi, "
                f"{ei_pi_count} ei+pi)."
            )
        return self


class CompareRequest(BaseModel):
    """Inputs to `openadmet compare`.

    Two mutually exclusive modes:

    * **Mode A (from model_dirs)**: give `model_names` + `label_types`.
      Server resolves each name to its NAS model_dir and passes to upstream.
    * **Mode B (from stats JSONs)**: give `labels` + `task_names`; the
      `model_stats_files` upload lives outside the pydantic model (on the
      FastAPI endpoint signature). At least one mode's required fields must
      be set.
    """

    # ---- Mode A ----
    model_names: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Pre-registered model names on NAS (Mode A).",
        json_schema_extra=default_semantics("unset", "empty when omitted"),
    )
    label_types: list[LabelTypeChoice] = Field(default_factory=list, json_schema_extra=default_semantics("unset", "empty when omitted"))

    # ---- Mode B ----
    labels: list[str] = Field(default_factory=list, json_schema_extra=default_semantics("unset", "empty when omitted"))
    task_names: list[str] = Field(default_factory=list, json_schema_extra=default_semantics("unset", "empty when omitted"))

    # ---- Common ----
    mt_id: Optional[str] = Field(
        default=None,
        description="Multitask identifier — required when comparing multitask models.",
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    report: bool = Field(
        default=False,
        description="Generate a PDF report in addition to the JSON stats.",
    )

    @model_validator(mode="after")
    def _check_mode_selection(self) -> "CompareRequest":
        mode_a = bool(self.model_names)
        mode_b = bool(self.labels or self.task_names)

        if not mode_a and not mode_b:
            raise ValueError(
                "compare: provide either `model_names` (Mode A) or "
                "`labels`+`task_names` (Mode B, with model_stats_files upload)."
            )
        if mode_a and mode_b:
            raise ValueError(
                "compare: `model_names` (Mode A) and `labels`/`task_names` "
                "(Mode B) are mutually exclusive."
            )

        if mode_a:
            if len(self.model_names) < 2:
                raise ValueError("compare Mode A: need ≥ 2 model_names.")
            if self.label_types and len(self.label_types) != len(self.model_names):
                raise ValueError(
                    "compare Mode A: `label_types` must be same length as `model_names`."
                )
        if mode_b:
            if len(self.labels) < 2:
                raise ValueError("compare Mode B: need ≥ 2 labels.")
            if len(self.task_names) != len(self.labels):
                raise ValueError(
                    "compare Mode B: `task_names` must be same length as `labels`."
                )
        return self
