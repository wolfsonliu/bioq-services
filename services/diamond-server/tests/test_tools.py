"""Unit tests for diamond-server argv builders + a3m reconstruction.

Pure logic — no DIAMOND binary, no subprocess.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic_settings import SettingsConfigDict

from server.a3m import (
    Hit,
    _align_hit_to_query,
    parse_blast_tab,
    read_first_fasta,
    reconstruct_a3m,
)
from server.models import (
    BlastpRequest,
    BlastxRequest,
    ClusterRequest,
    MakedbRequest,
    MsaRequest,
)
from server.settings import DiamondSettings
from server.tools import (
    blastp_argv,
    blastx_argv,
    cluster_argv,
    makedb_argv,
    msa_argv,
    outfmt_ext,
)


class _Off(DiamondSettings):
    model_config = SettingsConfigDict(env_prefix="DIAMOND_TEST_", env_file=None, extra="ignore")


def _settings(tmp_path: Path) -> _Off:
    return _Off(binary="/usr/local/bin/diamond", threads=4, jobs_base_dir=tmp_path)


def _flag_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


# ---- outfmt_ext ----

def test_outfmt_ext():
    assert outfmt_ext("6") == "tsv"
    assert outfmt_ext("0") == "txt"
    assert outfmt_ext("5") == "xml"
    assert outfmt_ext("101") == "sam"
    assert outfmt_ext("103") == "paf"
    assert outfmt_ext("104") == "tsv"


# ---- makedb ----

def test_makedb_argv(tmp_path):
    s = _settings(tmp_path)
    argv = makedb_argv(MakedbRequest(name="ref"), job_dir=tmp_path / "j", sequences_path=tmp_path / "in.faa", settings=s)
    assert argv[0] == "/usr/local/bin/diamond"
    assert argv[1] == "makedb"
    assert "--in" in argv and "--db" in argv
    assert _flag_value(argv, "--db").endswith("output/ref")
    assert _flag_value(argv, "-p") == "4"


# ---- blastp / blastx (driver invocation) ----

def test_blastp_argv_with_subject(tmp_path):
    s = _settings(tmp_path)
    argv = blastp_argv(
        BlastpRequest(name="hits", sensitivity="very-sensitive"),
        job_dir=tmp_path / "j", query_path=tmp_path / "q.faa",
        subject_path=tmp_path / "subj.faa", settings=s,
    )
    assert argv[0] == sys.executable
    assert argv[1:3] == ["-m", "server.diamond_driver"]
    assert argv[3] == "search"
    assert _flag_value(argv, "--command") == "blastp"
    assert "--subject" in argv
    assert "--db" not in argv
    assert "--db-work" in argv
    assert _flag_value(argv, "--outfmt") == "6"
    assert _flag_value(argv, "--sensitivity") == "very-sensitive"
    assert _flag_value(argv, "--max-target-seqs") == "25"


def test_blastp_argv_with_db(tmp_path):
    s = _settings(tmp_path)
    argv = blastp_argv(
        BlastpRequest(name="hits", outfmt="103"),
        job_dir=tmp_path / "j", query_path=tmp_path / "q.faa",
        db_path=tmp_path / "ref.dmnd", settings=s,
    )
    assert "--db" in argv
    assert "--subject" not in argv
    assert _flag_value(argv, "--output").endswith("hits.paf")


def test_blastx_argv_command(tmp_path):
    s = _settings(tmp_path)
    argv = blastx_argv(
        BlastxRequest(name="hits"),
        job_dir=tmp_path / "j", query_path=tmp_path / "q.fna",
        subject_path=tmp_path / "subj.faa", settings=s,
    )
    assert _flag_value(argv, "--command") == "blastx"


def test_blastp_default_sensitivity_from_settings(tmp_path):
    s = _Off(binary="diamond", threads=2, jobs_base_dir=tmp_path, default_sensitivity="sensitive")
    argv = blastp_argv(
        BlastpRequest(name="h"), job_dir=tmp_path / "j",
        query_path=tmp_path / "q.faa", subject_path=tmp_path / "s.faa", settings=s,
    )
    assert _flag_value(argv, "--sensitivity") == "sensitive"


# ---- cluster ----

def test_cluster_argv(tmp_path):
    s = _settings(tmp_path)
    argv = cluster_argv(
        ClusterRequest(name="lib", algorithm="deepclust", approx_id=90, member_cover=80),
        job_dir=tmp_path / "j", sequences_path=tmp_path / "seqs.faa", settings=s,
    )
    assert argv[3] == "cluster"
    assert _flag_value(argv, "--algorithm") == "deepclust"
    assert _flag_value(argv, "--approx-id") == "90.0"
    assert _flag_value(argv, "--member-cover") == "80.0"
    assert _flag_value(argv, "--output").endswith("lib.clusters.tsv")


# ---- msa ----

def test_msa_argv(tmp_path):
    s = _settings(tmp_path)
    argv = msa_argv(
        MsaRequest(name="query"),
        job_dir=tmp_path / "j", query_path=tmp_path / "q.faa",
        db_path=tmp_path / "uniref.dmnd", settings=s,
    )
    assert argv[3] == "msa"
    assert "--db" in argv
    assert _flag_value(argv, "--output").endswith("query.a3m")
    assert _flag_value(argv, "--max-target-seqs") == "2000"  # msa depth default


# ---- a3m reconstruction ----

def test_align_hit_basic_match(tmp_path):
    # Full-length exact match: line == subject, all uppercase, no insertions.
    hit = Hit(sseqid="s", qstart=1, qend=5, qseq="MKHKG", sseq="MKHKG", bitscore=10)
    line = _align_hit_to_query(5, hit)
    assert line == "MKHKG"


def test_align_hit_insertion_lowercase():
    # Column 3 is an insertion relative to the query (qseq gap) → lowercase.
    hit = Hit(sseqid="s", qstart=1, qend=4, qseq="MK-HK", sseq="MKXHK", bitscore=10)
    line = _align_hit_to_query(5, hit)
    assert line == "MKxHK-"
    # Column conservation: non-lowercase chars == query length.
    assert sum(1 for c in line if not c.islower()) == 5


def test_align_hit_boundary_gaps():
    # Hit covers query cols 3..5 only → leading '--', no trailing.
    hit = Hit(sseqid="s", qstart=3, qend=5, qseq="HKG", sseq="HKG", bitscore=10)
    line = _align_hit_to_query(5, hit)
    assert line == "--HKG"


def test_align_hit_subject_gap_becomes_dash():
    # Subject gap at a query column → '-' (deletion in the homolog).
    hit = Hit(sseqid="s", qstart=1, qend=5, qseq="MKHKG", sseq="MK-KG", bitscore=10)
    line = _align_hit_to_query(5, hit)
    assert line == "MK-KG"


def test_reconstruct_a3m_shape():
    hits = [
        Hit(sseqid="h1", qstart=1, qend=5, qseq="MKHKG", sseq="MKHKG", bitscore=20),
        Hit(sseqid="h2", qstart=1, qend=4, qseq="MK-HK", sseq="MKXHK", bitscore=10),
        Hit(sseqid="h1", qstart=1, qend=5, qseq="MKHKG", sseq="MKHKG", bitscore=5),  # dup id → dropped
    ]
    text = reconstruct_a3m("query1", "mkhkg", hits)
    lines = text.strip().split("\n")
    assert lines[0] == ">query1"
    assert lines[1] == "MKHKG"  # query uppercased, full length
    assert lines[2] == ">h1"
    assert lines[4] == ">h2"
    assert ">h1" in text and text.count(">h1") == 1  # dedup


# ---- parse_blast_tab / read_first_fasta ----

def test_parse_blast_tab(tmp_path):
    tsv = tmp_path / "hits.tsv"
    tsv.write_text(
        "query1\tsubjA\t95.0\t50\t1\t50\t1\t50\t1e-30\t120.5\tMKHKG\tMKHKG\n"
        "# comment\n"
        "short\tline\n"
    )
    hits = parse_blast_tab(tsv)
    assert len(hits) == 1
    assert hits[0].sseqid == "subjA"
    assert hits[0].qstart == 1
    assert hits[0].qend == 50
    assert hits[0].bitscore == 120.5


def test_read_first_fasta(tmp_path):
    faa = tmp_path / "q.faa"
    faa.write_text(">q1 desc\nmkhk\ngss\n>q2\nAAAA\n")
    qid, seq = read_first_fasta(faa)
    assert qid == "q1"
    assert seq == "MKHKGSS"
