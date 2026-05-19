"""YAML construction + argv assembly for boltz-server.

Two responsibilities:

  * `build_yaml(req, ...)` renders the structured pydantic request into a
    Boltz-compatible YAML file at `<job_dir>/input/input.yaml`. The stem is
    fixed to `input` so the output path (`output/predictions/input/`) is
    predictable for `JobAdapter.detect_outputs`.

  * `predict_argv(req, ...)` composes the argv for the `boltz predict`
    subprocess. `--model boltz2` is hard-coded (v0.0.1 supports Boltz-2 only).

`raw_yaml` and `raw_yaml_uri` bypass the structured renderer but still pass
through `validate_raw_yaml` to catch obvious schema mistakes early as 422.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import yaml
from fastapi import HTTPException

from .models import (
    BondConstraint,
    ContactConstraint,
    PocketConstraint,
    PredictAffinityRequest,
    PredictStructureRequest,
    SequenceEntry,
    TemplateEntry,
)
from .settings import BoltzSettings
from .uris import resolve_uri


RequestT = Union[PredictStructureRequest, PredictAffinityRequest]


# ----- YAML construction -----

def build_yaml(
    req: RequestT,
    *,
    job_dir: Path,
    settings: BoltzSettings,
    saved_msa_paths: dict[str, Path],
    saved_template_paths: dict[str, Path],
) -> Path:
    """Render the request into `<job_dir>/input/input.yaml`. Returns its path."""
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = input_dir / "input.yaml"

    if req.raw_yaml_uri:
        resolve_uri(req.raw_yaml_uri, yaml_path, settings)
        validate_raw_yaml(yaml_path.read_text(encoding="utf-8"))
        return yaml_path

    if req.raw_yaml:
        validate_raw_yaml(req.raw_yaml)
        yaml_path.write_text(req.raw_yaml, encoding="utf-8")
        return yaml_path

    doc: dict[str, Any] = {"version": 1, "sequences": []}
    for entry in req.sequences:
        doc["sequences"].append(
            _render_sequence(entry, req.msa_mode, saved_msa_paths)
        )
    if req.constraints:
        doc["constraints"] = [_render_constraint(c) for c in req.constraints]
    if req.templates:
        doc["templates"] = [
            _render_template(t, saved_template_paths) for t in req.templates
        ]
    if isinstance(req, PredictAffinityRequest):
        doc["properties"] = [{"affinity": {"binder": req.binder_id}}]

    yaml_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return yaml_path


def _render_sequence(
    entry: SequenceEntry,
    global_msa_mode: str,
    saved_msa_paths: dict[str, Path],
) -> dict[str, Any]:
    """Render one SequenceEntry as the upstream `{type: {fields}}` dict shape."""
    fields: dict[str, Any] = {"id": entry.id}

    if entry.type == "ligand":
        if entry.smiles:
            fields["smiles"] = entry.smiles
        else:
            fields["ccd"] = entry.ccd
        return {"ligand": fields}

    # protein / dna / rna
    fields["sequence"] = entry.sequence
    if entry.cyclic:
        fields["cyclic"] = True
    if entry.modifications:
        fields["modifications"] = [m.model_dump() for m in entry.modifications]

    if entry.type == "protein":
        msa_value = _resolve_msa_field(entry, global_msa_mode, saved_msa_paths)
        if msa_value is not None:
            fields["msa"] = msa_value

    return {entry.type: fields}


def _resolve_msa_field(
    entry: SequenceEntry,
    global_msa_mode: str,
    saved_msa_paths: dict[str, Path],
) -> Optional[str]:
    """Pick the YAML `msa` field value for a protein chain.

    Returns:
      * `None`  → omit `msa:` entirely (boltz will use --use_msa_server)
      * `"empty"` → single-sequence mode
      * relative path → server-side a3m file (relative to input/input.yaml)
    """
    chain_id = entry.id if isinstance(entry.id, str) else entry.id[0]

    # Per-chain explicit override wins.
    if entry.msa_uri is not None:
        if entry.msa_uri == "empty":
            return "empty"
        path = saved_msa_paths.get(chain_id)
        if path is not None:
            # Relative to the YAML's directory (input/) so boltz can find it.
            return f"msa/{path.name}"
        # URI not pre-resolved by endpoint; should not happen if endpoint did
        # its job, but degrade gracefully — treat as literal path.
        return entry.msa_uri

    # Fall back to global mode.
    if global_msa_mode == "auto":
        return None
    if global_msa_mode == "empty":
        return "empty"
    # provided: each protein must have a saved a3m (validator caught the missing case)
    path = saved_msa_paths.get(chain_id)
    if path is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"msa_mode='provided' but no a3m supplied for chain {chain_id!r}; "
                "set SequenceEntry.msa_uri or upload a matching `msa_files` (filename stem = chain id)."
            ),
        )
    return f"msa/{path.name}"


def _render_constraint(c: Union[BondConstraint, PocketConstraint, ContactConstraint]) -> dict[str, Any]:
    """Render one ConstraintEntry as `{<kind>: {fields}}`."""
    if isinstance(c, BondConstraint):
        return {"bond": {"atom1": list(c.atom1), "atom2": list(c.atom2)}}
    if isinstance(c, PocketConstraint):
        return {
            "pocket": {
                "binder": c.binder,
                "contacts": [list(pair) for pair in c.contacts],
                "max_distance": c.max_distance,
                "force": c.force,
            }
        }
    return {
        "contact": {
            "token1": list(c.token1),
            "token2": list(c.token2),
            "max_distance": c.max_distance,
            "force": c.force,
        }
    }


def _render_template(t: TemplateEntry, saved_template_paths: dict[str, Path]) -> dict[str, Any]:
    """Render one TemplateEntry. URIs were already resolved by the endpoint."""
    uri = t.cif_uri or t.pdb_uri
    assert uri is not None  # validator guarantees one
    path = saved_template_paths.get(uri)
    if path is None:
        # URI not resolved by endpoint (would only happen if endpoint forgot to
        # save it). Pass through literally — boltz may fail loudly later.
        ref = uri
    else:
        ref = f"templates/{path.name}"

    fields: dict[str, Any] = {}
    fields["cif" if t.cif_uri else "pdb"] = ref
    if t.chain_id is not None:
        fields["chain_id"] = t.chain_id
    if t.template_id is not None:
        fields["template_id"] = t.template_id
    if t.force:
        fields["force"] = True
        fields["threshold"] = t.threshold
    return fields


def validate_raw_yaml(text: str) -> None:
    """Cheap structural sanity check on caller-supplied YAML.

    Catches the most common mistakes (not YAML, not a dict, missing
    `sequences`) before handing off to `boltz predict`. The deep schema check
    happens inside boltz itself.
    """
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise HTTPException(
            status_code=422, detail=f"raw_yaml is not valid YAML: {exc}"
        ) from None

    if not isinstance(doc, dict):
        raise HTTPException(
            status_code=422,
            detail=f"raw_yaml top-level must be a mapping, got {type(doc).__name__}",
        )
    if "sequences" not in doc or not isinstance(doc["sequences"], list) or not doc["sequences"]:
        raise HTTPException(
            status_code=422,
            detail="raw_yaml must contain a non-empty `sequences:` list",
        )


# ----- argv assembly -----

def predict_argv(
    req: RequestT,
    *,
    job_dir: Path,
    yaml_path: Path,
    settings: BoltzSettings,
) -> list[str]:
    """Compose the `boltz predict` argv. Returns a list ready for subprocess.Popen."""
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    argv: list[str] = [
        settings.binary,
        "predict",
        str(yaml_path),
        "--out_dir", str(out_dir),
        "--cache", str(settings.cache_dir),
        "--model", "boltz2",  # hardcoded — v0.0.1 supports Boltz-2 only
        "--accelerator", "gpu",
        "--devices", "1",
        "--recycling_steps", str(req.recycling_steps),
        "--sampling_steps", str(req.sampling_steps),
        "--diffusion_samples", str(req.diffusion_samples),
        "--output_format", req.output_format,
        "--override",  # service guarantees fresh job_dir; never resume stale state
    ]

    if req.seed is not None:
        argv += ["--seed", str(req.seed)]
    if req.step_scale is not None:
        argv += ["--step_scale", str(req.step_scale)]
    if req.use_potentials:
        argv.append("--use_potentials")
    if req.write_full_pae:
        argv.append("--write_full_pae")
    if req.write_full_pde:
        argv.append("--write_full_pde")
    if req.no_kernels:
        argv.append("--no_kernels")

    if req.msa_mode == "auto":
        argv += [
            "--use_msa_server",
            "--msa_server_url", req.msa_server_url,
            "--msa_pairing_strategy", req.msa_pairing_strategy,
        ]

    if isinstance(req, PredictAffinityRequest):
        if req.affinity_mw_correction:
            argv.append("--affinity_mw_correction")
        argv += [
            "--sampling_steps_affinity", str(req.sampling_steps_affinity),
            "--diffusion_samples_affinity", str(req.diffusion_samples_affinity),
        ]

    return argv
