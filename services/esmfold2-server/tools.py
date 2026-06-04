"""Input JSON construction + argv assembly for esmfold2-server.

Two responsibilities:

  * `build_input_json(req, ...)` renders the pydantic request into a JSON file
    at `<job_dir>/input/input.json` that `inference.py` consumes.

  * `fold_argv(req, ...)` composes the argv for the `inference.py` subprocess.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import FoldRequest, SequenceEntry
from .settings import ESMFold2Settings


def build_input_json(
    req: FoldRequest,
    *,
    job_dir: Path,
    saved_msa_paths: dict[str, Path],
) -> Path:
    """Render the request into `<job_dir>/input/input.json`. Returns its path."""
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    json_path = input_dir / "input.json"

    doc: dict[str, Any] = {"sequences": []}
    for entry in req.sequences:
        doc["sequences"].append(_render_sequence(entry, saved_msa_paths))

    json_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return json_path


def _render_sequence(
    entry: SequenceEntry,
    saved_msa_paths: dict[str, Path],
) -> dict[str, Any]:
    chain_id = entry.id if isinstance(entry.id, str) else entry.id[0]
    fields: dict[str, Any] = {
        "type": entry.type,
        "id": entry.id,
    }

    if entry.type == "ligand":
        if entry.smiles:
            fields["smiles"] = entry.smiles
        else:
            fields["ccd"] = entry.ccd
    else:
        fields["sequence"] = entry.sequence
        if entry.modifications:
            fields["modifications"] = [m.model_dump() for m in entry.modifications]

    msa_path = saved_msa_paths.get(chain_id)
    if msa_path is not None:
        fields["msa_path"] = str(msa_path)

    return fields


def fold_argv(
    req: FoldRequest,
    *,
    job_dir: Path,
    input_json: Path,
    settings: ESMFold2Settings,
) -> list[str]:
    """Compose the inference.py argv. Returns a list ready for subprocess."""
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    argv: list[str] = [
        settings.python,
        settings.inference_script,
        "--input-json", str(input_json),
        "--output-dir", str(out_dir),
        "--model-dir", str(settings.model_dir),
        "--esmc-dir", str(settings.esmc_dir),
        "--ccd-path", str(settings.ccd_path),
        "--num-loops", str(req.num_loops),
        "--num-sampling-steps", str(req.num_sampling_steps),
        "--num-diffusion-samples", str(req.num_diffusion_samples),
    ]

    if req.seed is not None:
        argv += ["--seed", str(req.seed)]
    if req.noise_scale is not None:
        argv += ["--noise-scale", str(req.noise_scale)]
    if req.step_scale is not None:
        argv += ["--step-scale", str(req.step_scale)]

    return argv
