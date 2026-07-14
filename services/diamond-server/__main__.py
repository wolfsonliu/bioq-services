"""CLI batch-mode entry point for diamond-server (SIF / sbatch).

Exposes all commands, including the CLI-only ``makedb``::

    python -m server makedb --sequences ref.faa --output-dir /scratch/out/
    python -m server blastp --query q.faa --db ref.dmnd --output-dir /scratch/out/
    python -m server blastp --query q.faa --subject subj.faa --output-dir out/
    python -m server blastx --query reads.fna --db ref.dmnd --output-dir out/
    python -m server cluster --sequences lib.faa --algorithm cluster --output-dir out/
    python -m server msa --query q.faa --db uniref.dmnd --output-dir out/
"""

from __future__ import annotations

from bioagent_service.cli import CLIEndpoint, create_cli

from .adapter import DiamondAdapter
from .models import BlastpRequest, BlastxRequest, ClusterRequest, MakedbRequest, MsaRequest
from .settings import DiamondSettings
from .tools import blastp_argv, blastx_argv, cluster_argv, makedb_argv, msa_argv

settings = DiamondSettings()
adapter = DiamondAdapter(settings=settings)


def _makedb_build(req, inputs, job_dir, settings):
    return makedb_argv(req, job_dir=job_dir, sequences_path=inputs["sequences"], settings=settings)


def _blastp_build(req, inputs, job_dir, settings):
    return blastp_argv(
        req, job_dir=job_dir, query_path=inputs["query"],
        db_path=inputs.get("db"), subject_path=inputs.get("subject"), settings=settings,
    )


def _blastx_build(req, inputs, job_dir, settings):
    return blastx_argv(
        req, job_dir=job_dir, query_path=inputs["query"],
        db_path=inputs.get("db"), subject_path=inputs.get("subject"), settings=settings,
    )


def _cluster_build(req, inputs, job_dir, settings):
    return cluster_argv(req, job_dir=job_dir, sequences_path=inputs["sequences"], settings=settings)


def _msa_build(req, inputs, job_dir, settings):
    return msa_argv(req, job_dir=job_dir, query_path=inputs["query"], db_path=inputs["db"], settings=settings)


endpoints = {
    "makedb": CLIEndpoint(
        name="makedb",
        help="Build a DIAMOND .dmnd database from a protein FASTA (CLI/SIF only)",
        request_model=MakedbRequest,
        build_argv=_makedb_build,
        inputs={"sequences": ("Protein FASTA to index", True)},
    ),
    "blastp": CLIEndpoint(
        name="blastp",
        help="Align a protein query against a protein DB",
        request_model=BlastpRequest,
        build_argv=_blastp_build,
        inputs={
            "query": ("Query protein FASTA", True),
            "db": ("Prebuilt .dmnd database", False),
            "subject": ("Subject protein FASTA (build DB inline)", False),
        },
    ),
    "blastx": CLIEndpoint(
        name="blastx",
        help="Align a translated-DNA query against a protein DB",
        request_model=BlastxRequest,
        build_argv=_blastx_build,
        inputs={
            "query": ("Query DNA FASTA", True),
            "db": ("Prebuilt .dmnd database", False),
            "subject": ("Subject protein FASTA (build DB inline)", False),
        },
    ),
    "cluster": CLIEndpoint(
        name="cluster",
        help="Cluster a protein FASTA (cluster / deepclust / linclust)",
        request_model=ClusterRequest,
        build_argv=_cluster_build,
        inputs={"sequences": ("Protein FASTA to cluster", True)},
    ),
    "msa": CLIEndpoint(
        name="msa",
        help="Build a query-anchored a3m via blastp against a reference DB",
        request_model=MsaRequest,
        build_argv=_msa_build,
        inputs={
            "query": ("Query protein FASTA", True),
            "db": ("Reference .dmnd database", True),
        },
    ),
}

create_cli(adapter, settings, endpoints, version="0.0.1")
