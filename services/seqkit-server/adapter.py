"""SeqKit service adapter: output detection + manifest + examples.

SeqKit is CPU-only, a single static binary, and has NO model weights.
`detect_outputs` accepts either endpoint's artifact (`stats.tsv` or
`revcomp.fasta`) non-empty under `output/`. No `subprocess_env` override is
needed — there is no interpreter or PYTHONPATH to wire.
"""

from __future__ import annotations

from pathlib import Path

from bioq_service import EndpointExample, JobAdapter

from .settings import SeqkitSettings

# Endpoint artifacts, relative to the job `output/` dir.
_OUTPUTS = ("stats.tsv", "revcomp.fasta")


class SeqkitAdapter(JobAdapter):
    name = "seqkit"

    settings: SeqkitSettings  # narrow for IDEs

    def __init__(self, settings: SeqkitSettings) -> None:
        super().__init__(settings)

    def detect_outputs(self, job_dir: Path) -> bool:
        out = self.output_dir(job_dir)
        if not out.is_dir():
            return False
        for name in _OUTPUTS:
            path = out / name
            try:
                if path.is_file() and path.stat().st_size > 0:
                    return True
            except OSError:
                continue
        return False

    def manifest_extras(self) -> dict:
        return {
            "protocol": (
                "SeqKit is CPU-only (single static Go binary, no GPU, no model "
                "weights). It manipulates ONE FASTA/FASTQ file per call. This "
                "service wraps two deterministic operations: `stats` (summary "
                "statistics) and `revcomp` (reverse-complement). It does NOT "
                "align, assemble, or search sequences."
            ),
            "tool_outputs": {
                "stats": (
                    "output/stats.tsv — tab-separated `seqkit stats` table. "
                    "Core columns: file/format/type/num_seqs/sum_len/min_len/"
                    "avg_len/max_len; all_stats=true adds Q1/Q2/Q3/sum_gap/"
                    "N50/N50_num/Q20(%)/Q30(%)/AvgQual/GC(%)/sum_n."
                ),
                "revcomp": (
                    "output/revcomp.fasta — reverse-complement of every input "
                    "record; headers preserved, sequences upper-case."
                ),
                "note": (
                    "Enumerate via GET /api/jobs/{id}/files; single file via "
                    "/api/jobs/{id}/file/{relpath}; whole dir via /download."
                ),
            },
            "input_uri_schemes": {
                "upload": "multipart/form-data (input_fasta).",
                "job://<id>/<file>": "Re-use a FASTA/FASTQ from a prior job.",
                "file:///abs/path": "Direct path on the shared mount.",
                "oss://<bucket>/<key>": "Alibaba Cloud OSS object.",
                "http(s)://...": "Generic URL, including OSS signed URLs.",
            },
            "endpoints_summary": {
                "/api/stats": "Summary statistics for one FASTA/FASTQ.",
                "/api/tasks/stats": "FC async task-mode variant.",
                "/api/revcomp": "Reverse-complement every record.",
                "/api/tasks/revcomp": "FC async task-mode variant.",
                "/healthz/detail": "Extended health (seqkit binary present + runs).",
            },
            "not_in_scope_v0_0_1": (
                "Other seqkit subcommands (grep/split/subseq/mutate/concat/"
                "faidx/amplicon/...); FASTQ quality filtering beyond stats; "
                "multi-file batch input; gzip output. SeqKit does not align, "
                "assemble, or search."
            ),
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/stats": [
                EndpointExample(
                    title="full statistics table (TSV)",
                    curl=(
                        "curl -X POST $URL/api/stats "
                        "-F input_fasta=@reads.fasta "
                        "-F all_stats=true"
                    ),
                    notes="Result in output/stats.tsv.",
                ),
                EndpointExample(
                    title="core columns only",
                    curl=(
                        "curl -X POST $URL/api/stats "
                        "-F input_fasta=@reads.fasta "
                        "-F all_stats=false"
                    ),
                    notes="num_seqs/sum_len/min/avg/max only.",
                ),
                EndpointExample(
                    title="re-use a file from a prior job",
                    curl=(
                        "curl -X POST $URL/api/stats "
                        "-F input_fasta_uri=job://<job_id>/output/revcomp.fasta"
                    ),
                    notes="`input_fasta_uri` accepts job:// / oss:// / file:// / http(s)://.",
                ),
            ],
            "/api/revcomp": [
                EndpointExample(
                    title="reverse-complement (auto alphabet)",
                    curl=(
                        "curl -X POST $URL/api/revcomp "
                        "-F input_fasta=@reads.fasta"
                    ),
                    notes="Result in output/revcomp.fasta.",
                ),
                EndpointExample(
                    title="reverse-complement (explicit DNA)",
                    curl=(
                        "curl -X POST $URL/api/revcomp "
                        "-F input_fasta=@reads.fasta "
                        "-F seq_type=dna"
                    ),
                    notes="seq_type=dna silences seqkit's complement WARN.",
                ),
            ],
        }


__all__ = ["SeqkitAdapter"]
