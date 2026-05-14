"""argv builders + helper-script orchestration for proteinmpnn-server.

Each endpoint flow:
  1. prepare_inputs() runs helper_scripts/*.py synchronously to convert
     structured request fields into the JSONLs ProteinMPNN expects, writing
     them into <job_dir>/intermediates/.
  2. <endpoint>_argv() composes the final `python protein_mpnn_run.py ...`
     argv that the framework runner will execute (cwd = settings.root).
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from .models import DesignRequest, ProbsRequest, ScoreRequest
from .settings import ProteinMPNNSettings

logger = logging.getLogger(__name__)


_WEIGHT_DIRS = {
    "vanilla": "vanilla_model_weights",
    "soluble": "soluble_model_weights",
    "ca_only": "ca_model_weights",
    "abmpnn": "AbMPNN_model_weights",
}


def weight_flags(model_variant: str, model_name: str, settings: ProteinMPNNSettings) -> list[str]:
    """Build the `--path_to_model_weights`/`--model_name`/`--ca_only`/`--use_soluble_model`
    flag fragment for a given (variant, model_name) pair."""
    weights_subdir = settings.weights_dir / _WEIGHT_DIRS[model_variant]
    flags = [
        "--path_to_model_weights", str(weights_subdir) + "/",
        "--model_name", model_name,
    ]
    if model_variant == "ca_only":
        flags.append("--ca_only")
    elif model_variant == "soluble":
        flags.append("--use_soluble_model")
    return flags


def run_helper(script: str, args: list[str], settings: ProteinMPNNSettings) -> None:
    """Run helper_scripts/<script> with given args; 422 on non-zero exit.

    Used for the pre-flight JSONL preparation pipeline (parse_multiple_chains,
    assign_fixed_chains, make_*). Failures here mean the request was malformed
    (bad chain IDs, bad position spec, etc.), so they surface as 422 rather than
    a job-level failure.
    """
    script_path = settings.root / "helper_scripts" / script
    if not script_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Helper script not found: {script_path}",
        )
    cmd = [sys.executable, str(script_path), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(settings.root))
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-1000:]
        logger.warning("helper %s failed (rc=%d): %s", script, proc.returncode, tail)
        raise HTTPException(
            status_code=422,
            detail=f"helper {script} failed: {tail}",
        )


def prepare_inputs(
    job_dir: Path,
    *,
    settings: ProteinMPNNSettings,
    ca_only: bool,
    chains_to_design: Optional[str],
    fixed_positions: Optional[str],
    tied_positions: Optional[str],
    homooligomer: bool,
    bias_AA: Optional[dict],
    bias_by_res: Optional[dict],
    omit_AA_per_chain: Optional[dict],
) -> dict[str, Path]:
    """Run the (up to seven) helper scripts that produce ProteinMPNN's JSONL inputs.

    Returns a dict of artifact name → path. Keys that are not in the dict mean the
    corresponding helper was skipped (because the request did not supply that field).

    Failures are raised as HTTP 422 by `run_helper`, so callers don't catch.
    """
    inter = job_dir / "intermediates"
    inter.mkdir(parents=True, exist_ok=True)
    input_dir = job_dir / "input"

    paths: dict[str, Path] = {}

    parsed = inter / "parsed.jsonl"
    parse_args = [
        "--input_path", str(input_dir),
        "--output_path", str(parsed),
    ]
    if ca_only:
        parse_args.append("--ca_only")
    run_helper("parse_multiple_chains.py", parse_args, settings)
    paths["parsed"] = parsed

    if chains_to_design:
        assigned = inter / "assigned.jsonl"
        run_helper(
            "assign_fixed_chains.py",
            [
                "--input_path", str(parsed),
                "--output_path", str(assigned),
                "--chain_list", chains_to_design,
            ],
            settings,
        )
        paths["assigned"] = assigned

    if fixed_positions:
        fixed = inter / "fixed.jsonl"
        run_helper(
            "make_fixed_positions_dict.py",
            [
                "--input_path", str(parsed),
                "--output_path", str(fixed),
                "--chain_list", chains_to_design or "",
                "--position_list", fixed_positions,
            ],
            settings,
        )
        paths["fixed"] = fixed

    if tied_positions:
        tied = inter / "tied.jsonl"
        tied_args = [
            "--input_path", str(parsed),
            "--output_path", str(tied),
            "--chain_list", chains_to_design or "",
            "--position_list", tied_positions,
            "--homooligomer", "1" if homooligomer else "0",
        ]
        run_helper("make_tied_positions_dict.py", tied_args, settings)
        paths["tied"] = tied

    if bias_AA:
        bias = inter / "bias_AA.jsonl"
        aa_list = " ".join(bias_AA.keys())
        bias_list = " ".join(str(v) for v in bias_AA.values())
        run_helper(
            "make_bias_AA.py",
            [
                "--output_path", str(bias),
                "--AA_list", aa_list,
                "--bias_list", bias_list,
            ],
            settings,
        )
        paths["bias_AA"] = bias

    if bias_by_res:
        per_res = inter / "bias_by_res.jsonl"
        # Upstream's make_bias_per_res_dict.py needs custom invocation; for the
        # v0.0.1 surface we accept the pre-built dict directly. The script is
        # only useful when the dict needs to be built from a recipe; we skip
        # that case (callers either supply the dict or omit the field).
        per_res.write_text(json.dumps(bias_by_res))
        paths["bias_by_res"] = per_res

    if omit_AA_per_chain:
        omit = inter / "omit_AA.jsonl"
        omit.write_text(json.dumps(omit_AA_per_chain))
        paths["omit_AA"] = omit

    return paths


def design_argv(
    req: DesignRequest,
    *,
    job_dir: Path,
    paths: dict[str, Path],
    settings: ProteinMPNNSettings,
) -> list[str]:
    """Compose argv for `python protein_mpnn_run.py` in sequence-design mode."""
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    argv: list[str] = [
        sys.executable, "protein_mpnn_run.py",
        "--jsonl_path", str(paths["parsed"]),
        "--out_folder", str(out_dir),
        "--num_seq_per_target", str(req.num_seq_per_target),
        "--sampling_temp", req.sampling_temp,
        "--batch_size", str(req.batch_size),
        "--seed", str(req.seed),
        "--backbone_noise", str(req.backbone_noise),
        "--omit_AAs", req.omit_AAs,
        *weight_flags(req.model_variant, req.model_name, settings),
    ]
    if "assigned" in paths:
        argv += ["--chain_id_jsonl", str(paths["assigned"])]
    if "fixed" in paths:
        argv += ["--fixed_positions_jsonl", str(paths["fixed"])]
    if "tied" in paths:
        argv += ["--tied_positions_jsonl", str(paths["tied"])]
    if "bias_AA" in paths:
        argv += ["--bias_AA_jsonl", str(paths["bias_AA"])]
    if "bias_by_res" in paths:
        argv += ["--bias_by_res_jsonl", str(paths["bias_by_res"])]
    if "omit_AA" in paths:
        argv += ["--omit_AA_jsonl", str(paths["omit_AA"])]
    return argv


def score_argv(
    req: ScoreRequest,
    *,
    job_dir: Path,
    paths: dict[str, Path],
    settings: ProteinMPNNSettings,
) -> list[str]:
    """Compose argv for `python protein_mpnn_run.py --score_only 1`."""
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    argv: list[str] = [
        sys.executable, "protein_mpnn_run.py",
        "--jsonl_path", str(paths["parsed"]),
        "--out_folder", str(out_dir),
        "--num_seq_per_target", str(req.num_seq_per_target),
        "--sampling_temp", req.sampling_temp,
        "--batch_size", str(req.batch_size),
        "--seed", str(req.seed),
        "--backbone_noise", str(req.backbone_noise),
        "--score_only", "1",
        "--save_score", "1" if req.save_score else "0",
        *weight_flags(req.model_variant, req.model_name, settings),
    ]
    if "assigned" in paths:
        argv += ["--chain_id_jsonl", str(paths["assigned"])]
    return argv


_PROBS_FLAGS = {
    "conditional": "--conditional_probs_only",
    "conditional_backbone": "--conditional_probs_only_backbone",
    "unconditional": "--unconditional_probs_only",
}


def probs_argv(
    req: ProbsRequest,
    *,
    job_dir: Path,
    paths: dict[str, Path],
    settings: ProteinMPNNSettings,
) -> list[str]:
    """Compose argv for `python protein_mpnn_run.py --<probs_flag> 1`."""
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    argv: list[str] = [
        sys.executable, "protein_mpnn_run.py",
        "--jsonl_path", str(paths["parsed"]),
        "--out_folder", str(out_dir),
        "--batch_size", str(req.batch_size),
        "--seed", str(req.seed),
        "--backbone_noise", str(req.backbone_noise),
        _PROBS_FLAGS[req.kind], "1",
        "--save_probs", "1" if req.save_probs else "0",
        *weight_flags(req.model_variant, req.model_name, settings),
    ]
    if "assigned" in paths:
        argv += ["--chain_id_jsonl", str(paths["assigned"])]
    return argv
