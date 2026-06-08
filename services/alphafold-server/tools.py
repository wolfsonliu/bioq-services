"""Argv assembly for alphafold-server.

Builds the `run_alphafold.py` subprocess command. Database paths are derived
from `settings.data_dir` and vary by `db_preset` and `model_preset`.
"""

from __future__ import annotations

from pathlib import Path

from .models import FoldRequest
from .settings import AlphaFoldSettings


def fold_argv(
    req: FoldRequest,
    *,
    job_dir: Path,
    fasta_path: Path,
    settings: AlphaFoldSettings,
) -> list[str]:
    """Compose the run_alphafold.py argv. Returns a list ready for subprocess."""
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = settings.data_dir

    argv: list[str] = [
        settings.python,
        str(settings.root / "run_alphafold.py"),
        "--fasta_paths", str(fasta_path),
        "--output_dir", str(out_dir),
        "--data_dir", str(data),
        "--model_preset", req.model_preset,
        "--db_preset", req.db_preset,
        "--max_template_date", req.max_template_date,
        "--models_to_relax", req.models_to_relax,
        "--use_gpu_relax=" + str(req.use_gpu_relax).lower(),
        # Common databases
        "--uniref90_database_path", str(data / "uniref90" / "uniref90.fasta"),
        "--mgnify_database_path", str(data / "mgnify" / "mgy_clusters_2022_05.fa"),
        "--template_mmcif_dir", str(data / "pdb_mmcif" / "mmcif_files"),
        "--obsolete_pdbs_path", str(data / "pdb_mmcif" / "obsolete.dat"),
    ]

    # db_preset-dependent databases
    if req.db_preset == "full_dbs":
        argv += [
            "--bfd_database_path",
            str(data / "bfd" / "bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"),
            "--uniref30_database_path",
            str(data / "uniref30" / "UniRef30_2021_03"),
        ]
    else:
        argv += [
            "--small_bfd_database_path",
            str(data / "small_bfd" / "bfd-first_non_consensus_sequences.fasta"),
        ]

    # model_preset-dependent databases
    if req.model_preset == "multimer":
        argv += [
            "--pdb_seqres_database_path",
            str(data / "pdb_seqres" / "pdb_seqres.txt"),
            "--uniprot_database_path",
            str(data / "uniprot" / "uniprot.fasta"),
        ]
    else:
        argv += [
            "--pdb70_database_path",
            str(data / "pdb70" / "pdb70"),
        ]

    if req.model_preset == "multimer":
        argv += [
            "--num_multimer_predictions_per_model",
            str(req.num_multimer_predictions_per_model),
        ]

    if req.use_precomputed_msas:
        argv.append("--use_precomputed_msas=true")

    if req.random_seed is not None:
        argv += ["--random_seed", str(req.random_seed)]

    return argv
