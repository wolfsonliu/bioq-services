"""DIAMOND service adapter: output detection + manifest + examples.

DIAMOND is CPU-only and resolves all paths from argv, so `subprocess_cwd()`
stays None. `detect_outputs` accepts any non-empty file under `output/`
(hits.<ext> / *.clusters.tsv / *.a3m / *.dmnd) — label-agnostic, no manifest.
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import EndpointExample, JobAdapter

from .settings import DiamondSettings


class DiamondAdapter(JobAdapter):
    name = "diamond"

    settings: DiamondSettings  # narrow for IDEs

    def __init__(self, settings: DiamondSettings) -> None:
        super().__init__(settings)

    def detect_outputs(self, job_dir: Path) -> bool:
        out = self.output_dir(job_dir)
        if not out.is_dir():
            return False
        for path in out.rglob("*"):
            try:
                if path.is_file() and path.stat().st_size > 0:
                    return True
            except OSError:
                continue
        return False

    def manifest_extras(self) -> dict:
        return {
            "protocol": (
                "DIAMOND is CPU-only (no GPU). blastp/blastx align a query FASTA "
                "against a protein DB; cluster de-duplicates a FASTA; msa builds a "
                "query-anchored a3m from blastp hits. makedb is CLI/SIF-only."
            ),
            "tool_outputs": {
                "blastp": "output/<name>.<ext> — alignments (ext by outfmt: tsv/txt/xml/sam/paf).",
                "blastx": "output/<name>.<ext> — translated-DNA alignments (same formats).",
                "cluster": "output/<name>.clusters.tsv — two columns: <representative>\\t<member>.",
                "msa": (
                    "output/<name>.a3m — query is record 0 (full, uppercase); each "
                    "homolog aligned to query columns (gaps '-', insertions lowercase). "
                    "output/<name>.hits.tsv holds the raw blastp hits."
                ),
                "note": (
                    "Enumerate via GET /api/jobs/{id}/files; single file via "
                    "/api/jobs/{id}/file/{relpath}; whole dir via /download."
                ),
            },
            "db_handling": {
                "blastp/blastx": (
                    "Provide EITHER `db_uri` (a prebuilt .dmnd) OR a `subject` FASTA "
                    "(upload/uri) which is built inline via makedb. Exactly one."
                ),
                "msa": (
                    "Provide `db_uri` (a prebuilt .dmnd) or rely on the server's "
                    "default reference DB (DIAMOND_MSA_DB under db_dir)."
                ),
                "makedb": "CLI/SIF only: `python -m server makedb --sequences ref.faa`.",
            },
            "input_uri_schemes": {
                "upload": "multipart/form-data (query / subject / sequences).",
                "job://<id>/<file>": "Re-use a file (e.g. a .dmnd) from a prior job.",
                "file:///abs/path": "Direct NAS path on the shared mount.",
                "oss://<bucket>/<key>": "Alibaba Cloud OSS object.",
                "http(s)://...": "Generic URL, including OSS signed URLs.",
            },
            "config_tips": {
                "sensitivity": (
                    "fast (default) → ultra-sensitive. Higher = slower, more remote "
                    "homologs. Use very-sensitive+ for MSA depth on divergent queries."
                ),
                "max_target_seqs": "blastp/blastx default 25; msa default 2000 (MSA depth).",
                "algorithm": "cluster (cascaded, sensitive) / deepclust / linclust (fastest).",
            },
            "endpoints_summary": {
                "/api/blastp": "Protein query vs protein DB.",
                "/api/blastx": "Translated-DNA query vs protein DB.",
                "/api/cluster": "Small-scale protein clustering.",
                "/api/msa": "DIAMOND→a3m homolog MSA.",
                "/api/tasks/*": "FC async task-mode variants of the four endpoints.",
                "/healthz/detail": "Extended health (db_dir + msa_db presence).",
            },
            "not_in_scope_v0_0_1": (
                "makedb over HTTP; realign/recluster/reassign/greedy-vertex-cover/"
                "view/getseq/dbinfo; blastn (DNA-DNA); taxonomy output (102); DAA "
                "(100); paired multimer MSA (use mmseqs2-server); MSA caching."
            ),
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/blastp": [
                EndpointExample(
                    title="protein search against a subject FASTA (inline DB build)",
                    curl=(
                        "curl -X POST $URL/api/blastp "
                        "-F query=@query.faa "
                        "-F subject=@reference.faa "
                        "-F sensitivity=very-sensitive "
                        "-F name=hits"
                    ),
                    notes="Results in output/hits.tsv (outfmt 6). Provide a prebuilt DB via `db_uri` instead of `subject` for large references.",
                ),
                EndpointExample(
                    title="search against a prebuilt .dmnd from a prior job",
                    curl=(
                        "curl -X POST $URL/api/blastp "
                        "-F query=@query.faa "
                        "-F db_uri=job://<makedb_job_id>/output/ref.dmnd"
                    ),
                    notes="`db_uri` must point at a .dmnd built by `diamond makedb`.",
                ),
            ],
            "/api/blastx": [
                EndpointExample(
                    title="translated-DNA search",
                    curl=(
                        "curl -X POST $URL/api/blastx "
                        "-F query=@reads.fna "
                        "-F subject=@proteins.faa "
                        "-F name=hits"
                    ),
                    notes="Query is nucleotide FASTA; DIAMOND translates all 6 frames.",
                ),
            ],
            "/api/cluster": [
                EndpointExample(
                    title="deduplicate a protein set",
                    curl=(
                        "curl -X POST $URL/api/cluster "
                        "-F sequences=@library.faa "
                        "-F algorithm=cluster "
                        "-F approx_id=90 "
                        "-F name=lib"
                    ),
                    notes="output/lib.clusters.tsv lists <representative>\\t<member> pairs.",
                ),
            ],
            "/api/msa": [
                EndpointExample(
                    title="build a homolog a3m against the default reference DB",
                    curl=(
                        "curl -X POST $URL/api/msa "
                        "-F query=@query.faa "
                        "-F sensitivity=very-sensitive "
                        "-F max_target_seqs=2000 "
                        "-F name=query"
                    ),
                    notes="Needs DIAMOND_MSA_DB configured, or pass `db_uri` to a .dmnd. Output: output/query.a3m.",
                ),
            ],
        }


__all__ = ["DiamondAdapter"]
