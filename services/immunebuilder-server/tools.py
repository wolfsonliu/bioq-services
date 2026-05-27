"""FASTA construction + argv builders for immunebuilder-server.

Each endpoint flow:
  1. write_fasta() creates a FASTA file from sequence fields into job_dir/input/
  2. predict_*_argv() composes the CLI entry point command with the FASTA path
"""

from __future__ import annotations

from pathlib import Path

from .models import AntibodyRequest, NanobodyRequest, TCRRequest
from .settings import ImmuneBuilderSettings


def write_fasta(sequences: dict[str, str], dest: Path) -> Path:
    """Write sequences dict to FASTA file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        for chain_id, seq in sequences.items():
            f.write(f">{chain_id}\n{seq}\n")
    return dest


def _common_flags(
    req,
    *,
    job_dir: Path,
    fasta_path: Path,
    settings: ImmuneBuilderSettings,
) -> list[str]:
    """Build flag fragment shared by all three predictors."""
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    argv = ["-f", str(fasta_path), "-n", req.numbering_scheme]

    if req.save_all_models:
        argv += ["--to_directory", "-o", str(output_dir)]
    else:
        argv += ["-o", str(output_dir / "final_model.pdb")]

    if req.no_sidechain_bond_check:
        argv.append("-u")

    if req.n_threads > 0:
        argv += ["--n_threads", str(req.n_threads)]

    return argv


def predict_antibody_argv(
    req: AntibodyRequest,
    *,
    job_dir: Path,
    fasta_path: Path,
    settings: ImmuneBuilderSettings,
) -> list[str]:
    return [
        str(settings.venv_bin / "ABodyBuilder2"),
        *_common_flags(req, job_dir=job_dir, fasta_path=fasta_path, settings=settings),
    ]


def predict_nanobody_argv(
    req: NanobodyRequest,
    *,
    job_dir: Path,
    fasta_path: Path,
    settings: ImmuneBuilderSettings,
) -> list[str]:
    return [
        str(settings.venv_bin / "NanoBodyBuilder2"),
        *_common_flags(req, job_dir=job_dir, fasta_path=fasta_path, settings=settings),
    ]


def predict_tcr_argv(
    req: TCRRequest,
    *,
    job_dir: Path,
    fasta_path: Path,
    settings: ImmuneBuilderSettings,
) -> list[str]:
    return [
        str(settings.venv_bin / "TCRBuilder2"),
        *_common_flags(req, job_dir=job_dir, fasta_path=fasta_path, settings=settings),
    ]
