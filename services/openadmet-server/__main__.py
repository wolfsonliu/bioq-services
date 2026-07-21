"""CLI batch-mode entry point for openadmet-server.

Same Docker image runs the HTTP service (via uvicorn) and this CLI batch
mode (via ``python -m server predict ...`` / ``python -m server compare ...``).

Usage examples::

    # Predict from a CSV against one model
    python -m server predict \\
        --input-csv /data/compounds.csv \\
        --model-name herg-chemeleon-baseline \\
        --output-dir /scratch/results/

    # Predict inline SMILES against multiple models
    python -m server predict \\
        --params-json '{
            "input_smiles": "CCO,c1ccccc1",
            "model_names": ["herg-chemeleon-baseline", "pxr-chemeleon-baseline"]
        }' \\
        --output-dir /scratch/results/

    # Compare two model dirs
    python -m server compare \\
        --params-json '{
            "model_names": ["m1", "m2"],
            "label_types": ["biotarget", "biotarget"]
        }' \\
        --output-dir /scratch/compare/
"""

from __future__ import annotations

from pathlib import Path

from bioq_service.cli import CLIEndpoint, create_cli

from .adapter import OpenAdmetAdapter
from .models import CompareRequest, PredictRequest
from .settings import OpenAdmetSettings
from .tools import (
    archive_request,
    augment_csv_with_aliases,
    build_predict_shell,
    compare_argv_mode_a,
    compare_argv_mode_b,
    predict_composite_argv,
    sniff_smiles_column,
    split_inline_smiles,
    write_alias_csv,
)

settings = OpenAdmetSettings()
adapter = OpenAdmetAdapter(settings=settings)


def _predict_build(
    req: PredictRequest,
    inputs: dict[str, Path],
    job_dir: Path,
    settings: OpenAdmetSettings,
) -> list[str]:
    # Resolve models against NAS registry (raises if missing).
    available = {m.name: m for m in settings.list_models()}
    if not available:
        raise RuntimeError(
            f"No models registered under {settings.models_root}. "
            f"Run scripts/fetch_weights.sh + rsync to NAS."
        )
    missing = [n for n in req.model_names if n not in available]
    if missing:
        raise RuntimeError(
            f"Model names not on NAS: {missing}. Available: {sorted(available)}"
        )
    resolved = [available[n] for n in req.model_names]

    # Land input CSV in <job_dir>/input/input.csv.
    in_dir = job_dir / "input"
    in_dir.mkdir(parents=True, exist_ok=True)
    if "input_csv" in inputs and inputs["input_csv"] is not None:
        raw = inputs["input_csv"]
        detected = sniff_smiles_column(raw, settings.default_input_col_aliases)
        if detected is None:
            raise RuntimeError(
                f"Input CSV has no recognized SMILES column. "
                f"Expected one of: {settings.default_input_col_aliases}"
            )
        input_path = augment_csv_with_aliases(
            raw, in_dir / "input.csv", settings.default_input_col_aliases, detected,
        )
    elif "input_sdf" in inputs and inputs["input_sdf"] is not None:
        input_path = inputs["input_sdf"]
    elif req.input_smiles:
        smiles = split_inline_smiles(req.input_smiles, max_n=200)
        input_path = write_alias_csv(
            smiles, in_dir / "input.csv", settings.default_input_col_aliases,
        )
    else:
        raise RuntimeError(
            "predict CLI: provide --input-csv, --input-sdf, or `input_smiles` in --params-json"
        )

    archive_request(job_dir, "predict_request", req.model_dump(mode="json"))
    argvs = predict_composite_argv(
        req,
        input_csv=input_path,
        job_dir=job_dir,
        settings=settings,
        models=resolved,
    )
    return build_predict_shell(argvs, output_dir=job_dir / "output")


def _compare_build(
    req: CompareRequest,
    inputs: dict[str, Path],
    job_dir: Path,
    settings: OpenAdmetSettings,
) -> list[str]:
    archive_request(job_dir, "compare_request", req.model_dump(mode="json"))
    output_dir = job_dir / "output"

    if req.model_names:
        available = {m.name: m for m in settings.list_models()}
        missing = [n for n in req.model_names if n not in available]
        if missing:
            raise RuntimeError(f"Model names not on NAS: {missing}")
        model_dirs = [available[n].path for n in req.model_names]
        return compare_argv_mode_a(
            req, output_dir=output_dir, model_dirs=model_dirs, settings=settings,
        )

    # Mode B — expects one or more stats_file_N inputs.
    stats_paths: list[Path] = []
    idx = 0
    while True:
        key = f"stats_file_{idx}"
        if key not in inputs:
            break
        stats_paths.append(inputs[key])
        idx += 1
    if not stats_paths:
        raise RuntimeError(
            "compare Mode B: provide --stats-file-0 [--stats-file-1 ...] "
            "or use Mode A with model_names in --params-json"
        )
    return compare_argv_mode_b(
        req, output_dir=output_dir, stats_files=stats_paths, settings=settings,
    )


endpoints = {
    "predict": CLIEndpoint(
        name="predict",
        help="Predict ADMET properties with one or more NAS-registered models",
        request_model=PredictRequest,
        build_argv=_predict_build,
        inputs={
            "input_csv": ("Input CSV with a SMILES column (any of the recognized aliases)", False),
            "input_sdf": ("Input SDF file", False),
        },
    ),
    "compare": CLIEndpoint(
        name="compare",
        help="Post-hoc compare 2+ trained models (Mode A) or stats JSONs (Mode B)",
        request_model=CompareRequest,
        build_argv=_compare_build,
        # Stats-file inputs are dynamic; up to 4 slots exposed as CLI flags.
        # (More can go via `--params-json` but the CLI knows only these 4.)
        inputs={
            "stats_file_0": ("Model stats JSON file (Mode B slot 0)", False),
            "stats_file_1": ("Model stats JSON file (Mode B slot 1)", False),
            "stats_file_2": ("Model stats JSON file (Mode B slot 2)", False),
            "stats_file_3": ("Model stats JSON file (Mode B slot 3)", False),
        },
    ),
}


create_cli(adapter, settings, endpoints, version="0.0.1")
