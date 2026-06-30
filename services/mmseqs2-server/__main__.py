"""Dual-mode entry for mmseqs2-server.

Without arguments, runs the FastAPI HTTP service (ColabFold-protocol +
``/api/tasks/*`` task endpoints). With a subcommand, runs CLI batch mode for
sbatch / one-shot use::

    # HTTP mode (default, used in Docker / FC):
    python -m server

    # CLI batch mode (one-shot, used in sbatch / Apptainer):
    python -m server msa \\
        --input-fasta /data/query.fasta \\
        --mode env \\
        --output-dir /scratch/$SLURM_JOB_ID/

    python -m server pair \\
        --input-fasta /data/multimer.fasta \\
        --mode pairgreedy \\
        --output-dir /scratch/$SLURM_JOB_ID/

CLI mode shares ``colabfold_search_argv`` with the HTTP endpoints; the only
difference is execution model (synchronous one-shot vs async HTTP
submit/poll). See
``engineering/decisions/2026-05-29-cli-batch-mode.md`` for the rationale.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from bioagent_service.cli import CLIEndpoint, create_cli

from .adapter import MMseqs2JobAdapter
from .models import MSARequest
from .settings import MMseqs2Settings
from .tools import colabfold_search_argv, parse_mode_flags, parse_query_fasta

settings = MMseqs2Settings()
adapter = MMseqs2JobAdapter(settings=settings)


def _build_search_argv(
    req: MSARequest,
    inputs: dict,
    job_dir: Path,
    settings: MMseqs2Settings,
    *,
    require_paired: bool,
) -> list[str]:
    """Stage the input FASTA into ``<job_dir>/input/`` and build argv.

    CLI vs HTTP difference: the CLI receives a file path
    (``inputs["input_fasta"]``) and reads its content; the HTTP path
    constructs the FASTA on-the-fly from the ColabFold-protocol ``q`` form
    field. Validation rules are identical.
    """
    src = Path(inputs["input_fasta"])
    fasta_text = src.read_text(encoding="utf-8")

    parsed = parse_query_fasta(fasta_text)
    mode_config = parse_mode_flags(req.mode)

    is_paired_mode = mode_config.pair_mode == "paired"
    if require_paired and not is_paired_mode:
        raise ValueError(f"mode {req.mode!r} is not a paired mode (use msa subcommand)")
    if not require_paired and is_paired_mode:
        raise ValueError(f"mode {req.mode!r} is paired (use pair subcommand)")
    if require_paired and len(parsed) < 2:
        raise ValueError(f"paired mode needs >=2 sequences, got {len(parsed)}")

    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = input_dir / "query.fasta"
    fasta_path.write_text(fasta_text, encoding="utf-8")

    return colabfold_search_argv(
        query_path=fasta_path,
        output_dir=output_dir,
        mode_config=mode_config,
        settings=settings,
    )


def _msa_build(req, inputs, job_dir, settings):
    return _build_search_argv(req, inputs, job_dir, settings, require_paired=False)


def _pair_build(req, inputs, job_dir, settings):
    return _build_search_argv(req, inputs, job_dir, settings, require_paired=True)


endpoints = {
    "msa": CLIEndpoint(
        name="msa",
        help="Run a monomer (unpaired) ColabFold MSA search",
        request_model=MSARequest,
        build_argv=_msa_build,
        inputs={"input_fasta": ("Input FASTA file (single chain)", True)},
    ),
    "pair": CLIEndpoint(
        name="pair",
        help="Run a multimer (paired) ColabFold MSA search",
        request_model=MSARequest,
        build_argv=_pair_build,
        inputs={"input_fasta": ("Input FASTA file (>=2 chains)", True)},
    ),
}


def _has_cli_subcommand() -> bool:
    """Return True when argv looks like ``python -m server <endpoint> ...``."""
    if len(sys.argv) < 2:
        return False
    return sys.argv[1] in endpoints or sys.argv[1] in {"-h", "--help"}


def main() -> None:
    if _has_cli_subcommand():
        create_cli(adapter, settings, endpoints, version="0.0.1")
        return

    import uvicorn

    from .app import app

    port = int(os.environ.get("PORT", 9000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
