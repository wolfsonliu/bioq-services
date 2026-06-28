"""Argv assembly for chembounce-server.

Wraps upstream's `chembounce.py` CLI.  We subprocess into the upstream
CLI directly (rather than importing `chembounce`) because upstream packs
its whole pipeline into `main()` and has no clean library API.
"""

from __future__ import annotations

from pathlib import Path

from .models import ScaffoldHopRequest
from .settings import ChemBounceSettings


def scaffold_hop_argv(
    req: ScaffoldHopRequest,
    *,
    job_dir: Path,
    settings: ChemBounceSettings,
) -> list[str]:
    """Compose upstream `chembounce.py` argv.

    Output directory is `<job_dir>/output/`.  Upstream writes
    `overall_result.txt` + per-fragment intermediates into this dir.
    """
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if req.database == "full":
        scaffold_db = settings.scaffold_db_full
        fingerprint_db = settings.fingerprint_full
    else:  # 250mw
        scaffold_db = settings.scaffold_db_250mw
        fingerprint_db = settings.fingerprint_250mw

    argv: list[str] = [
        settings.python,
        settings.entrypoint,
        "-o", str(output_dir),
        "-i", req.input_smiles,
        "-n", str(req.frag_max_n),
        "-t", str(req.tanimoto_threshold),
        "--cand_max_n__rplc", str(req.cand_max_n__rplc),
        "--scaffold-db", str(scaffold_db),
        "--fingerprint-db", str(fingerprint_db),
    ]

    if req.core_smiles is not None:
        argv += ["--core_smiles", req.core_smiles]
    if req.overall_max_n is not None:
        argv += ["--overall_max_n", str(req.overall_max_n)]
    if req.scaffold_top_n is not None:
        argv += ["--scaffold_top_n", str(req.scaffold_top_n)]

    # Property thresholds — only emit if non-None to preserve upstream defaults.
    _emit_opt(argv, "--qed_min", req.qed_min)
    _emit_opt(argv, "--qed_max", req.qed_max)
    _emit_opt(argv, "--sa_min", req.sa_min)
    _emit_opt(argv, "--sa_max", req.sa_max)
    _emit_opt(argv, "--logp_min", req.logp_min)
    _emit_opt(argv, "--logp_max", req.logp_max)
    _emit_opt(argv, "--mw_min", req.mw_min)
    _emit_opt(argv, "--mw_max", req.mw_max)
    _emit_opt(argv, "--h_donor_min", req.h_donor_min)
    _emit_opt(argv, "--h_donor_max", req.h_donor_max)
    _emit_opt(argv, "--h_acceptor_min", req.h_acceptor_min)
    _emit_opt(argv, "--h_acceptor_max", req.h_acceptor_max)

    if req.wo_lipinski:
        argv.append("--wo_lipinski")
    if req.low_mem:
        argv.append("-l")

    return argv


def _emit_opt(argv: list[str], flag: str, val) -> None:  # noqa: ANN001
    if val is not None:
        argv += [flag, str(val)]


__all__ = ["scaffold_hop_argv"]
