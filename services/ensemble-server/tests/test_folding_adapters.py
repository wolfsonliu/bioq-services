"""Folding adapter unit tests.

Verify that each adapter's build_request produces the right endpoint + payload
keys for its underlying service, and normalize_output handles representative
output directory layouts.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from server.adapters.folding.alphafold import AlphaFoldFoldingAdapter, AlphaFoldOptions
from server.adapters.folding.boltz import BoltzFoldingAdapter, BoltzOptions
from server.adapters.folding.esmfold2 import ESMFold2FoldingAdapter, ESMFold2Options
from server.adapters.folding.promera import PromeraFoldingAdapter, PromeraOptions
from server.folding.aggregator import aggregate_folding
from server.folding.schemas import FoldingInput, SequenceEntry
from server.orchestrator.models import SubTaskRecord, SubTaskStatus


def _fc_mock():
    m = MagicMock()
    m.function = "fake"
    return m


# ---------------------------------------------------------------------------
# AlphaFold
# ---------------------------------------------------------------------------

def test_alphafold_build_request_has_fasta_upload_and_form_fields():
    adapter = AlphaFoldFoldingAdapter(_fc_mock())
    input = FoldingInput(
        sequences=[SequenceEntry(id="A", sequence="MKQH")],
        msa_mode="empty",
    )
    endpoint, payload, files = adapter.build_request(input, AlphaFoldOptions())

    assert endpoint == "/api/tasks/fold"
    assert payload["model_preset"] == "monomer_ptm"
    assert payload["db_preset"] == "reduced_dbs"
    assert payload["models_to_relax"] == "best"
    assert "input_fasta" in files
    fasta_path = files["input_fasta"]
    assert isinstance(fasta_path, Path)
    content = fasta_path.read_text()
    assert ">A" in content and "MKQH" in content


def test_alphafold_normalize_output_uses_relative_urls_and_extracts_plddt(tmp_path):
    """alphafold-server emits ``output/input/ranked_<N>.pdb`` (the orchestrator
    strips the ``output/`` root) plus ``output/input/ranking_debug.json`` with
    per-model plDDT on the **0-100 scale**.  URLs must use the relative path
    under outputs/<method>/ or the download route 404s (verified v0.0.9 prod).
    plDDT is normalized to 0-1 to align with the other folding methods so the
    aggregator ranks meaningfully across methods."""
    nested = tmp_path / "input"
    nested.mkdir()
    for i in range(3):
        (nested / f"ranked_{i}.pdb").write_bytes(b"REMARK fake pdb\nATOM\nEND\n")
    (nested / "ranking_debug.json").write_text(json.dumps({
        "order": ["model_3_ptm_pred_0", "model_1_ptm_pred_0", "model_5_ptm_pred_0"],
        "plddts": {
            "model_3_ptm_pred_0": 84.2,   # 0-100 input
            "model_1_ptm_pred_0": 79.5,
            "model_5_ptm_pred_0": 71.1,
        },
    }))

    adapter = AlphaFoldFoldingAdapter(_fc_mock())
    result = adapter.normalize_output("ens_fold_xyz__alphafold", tmp_path)
    assert result.method == "alphafold"
    assert result.status == "completed"
    assert result.fc_job_id == "ens_fold_xyz__alphafold"
    assert len(result.structures) == 3
    for i, s in enumerate(result.structures):
        assert s.format == "pdb"
        assert s.rank == i
        # Crucially: URL carries the ``input/`` subdir so download_structure
        # can resolve it under outputs/alphafold/.
        assert s.url == f"/v1/jobs/ens_fold_xyz/structures/alphafold/input/ranked_{i}.pdb"
    # plDDT rescaled to 0-1 to match esmfold2 / boltz / promera convention.
    assert result.structures[0].plddt == pytest.approx(0.842)
    assert result.structures[1].plddt == pytest.approx(0.795)
    assert result.structures[2].plddt == pytest.approx(0.711)
    assert result.confidence["plddt"] == pytest.approx(0.842)
    assert 0.0 <= result.structures[0].plddt <= 1.0


# ---------------------------------------------------------------------------
# ESMFold2
# ---------------------------------------------------------------------------

def test_esmfold2_build_request_sequences_json():
    adapter = ESMFold2FoldingAdapter(_fc_mock())
    input = FoldingInput(
        sequences=[SequenceEntry(id="A", sequence="MKQH"), SequenceEntry(id="B", sequence="LLLL")],
        msa_mode="auto",
    )
    endpoint, payload, files = adapter.build_request(input, ESMFold2Options(num_loops=2))

    assert endpoint == "/api/tasks/fold"
    seqs = json.loads(payload["sequences"])
    assert len(seqs) == 2
    assert seqs[0]["id"] == "A" and seqs[0]["sequence"] == "MKQH"
    assert payload["num_loops"] == 2
    assert files == {}


def test_esmfold2_normalize_output_with_metrics(tmp_path):
    """esmfold2's metrics.json is `{"samples": [{plddt_mean, ptm, iptm, output_file}, ...]}`
    — each sample row keys by `output_file` to pair with its CIF."""
    (tmp_path / "prediction_0.cif").write_bytes(b"loop_\nfake cif\n")
    (tmp_path / "prediction_1.cif").write_bytes(b"loop_\nfake cif\n")
    (tmp_path / "metrics.json").write_text(json.dumps({
        "samples": [
            {"sample_index": 0, "output_file": "prediction_0.cif",
             "plddt_mean": 0.87, "ptm": 0.92, "iptm": 0.0},
            {"sample_index": 1, "output_file": "prediction_1.cif",
             "plddt_mean": 0.81, "ptm": 0.88, "iptm": 0.0},
        ],
        "inference_time_s": 12.3,
    }))

    adapter = ESMFold2FoldingAdapter(_fc_mock())
    result = adapter.normalize_output("ens_fold_abc__esmfold2", tmp_path)
    assert result.method == "esmfold2"
    assert len(result.structures) == 2
    # Rank 0 paired with sample 0
    assert result.structures[0].rank == 0
    assert result.structures[0].plddt == pytest.approx(0.87)
    # Rank 1 paired with sample 1
    assert result.structures[1].rank == 1
    assert result.structures[1].plddt == pytest.approx(0.81)
    # Top-level confidence = rank-0 scores (plus mean_plddt alias)
    assert result.confidence["plddt_mean"] == pytest.approx(0.87)
    assert result.confidence["mean_plddt"] == pytest.approx(0.87)
    assert result.confidence["ptm"] == pytest.approx(0.92)


def test_esmfold2_normalize_output_uses_relative_url(tmp_path):
    """URL path is relative to outputs/<method>/ so multi-segment layouts work."""
    nested = tmp_path / "output"
    nested.mkdir()
    (nested / "prediction_0.cif").write_bytes(b"loop_\n")

    adapter = ESMFold2FoldingAdapter(_fc_mock())
    result = adapter.normalize_output("ens_fold_q__esmfold2", tmp_path)
    assert result.structures[0].url == (
        "/v1/jobs/ens_fold_q/structures/esmfold2/output/prediction_0.cif"
    )


# ---------------------------------------------------------------------------
# Boltz
# ---------------------------------------------------------------------------

def test_boltz_build_request_passes_msa_mode_and_empty_msa_uri():
    adapter = BoltzFoldingAdapter(_fc_mock())
    input = FoldingInput(
        sequences=[SequenceEntry(id="A", sequence="MKQH")],
        msa_mode="empty",
    )
    endpoint, payload, files = adapter.build_request(input, BoltzOptions(recycling_steps=2))

    assert endpoint == "/api/tasks/predict_structure"
    seqs = json.loads(payload["sequences"])
    assert seqs[0]["msa_uri"] == "empty"  # boltz needs this when msa_mode=empty
    assert payload["msa_mode"] == "empty"
    assert payload["recycling_steps"] == 2
    # `name` is not exposed because boltz-server hardcodes the YAML stem to
    # `input` (services/boltz-server/tools.py), so any value here would be
    # silently ignored.
    assert "name" not in payload
    assert files == {}


def test_boltz_normalize_output_pairs_cif_with_confidence(tmp_path):
    """Boltz nests outputs under predictions/<stem>/; URL must use the
    relative path so the download route can resolve it.  complex_plddt is
    Boltz's per-structure plDDT key (not `plddt`)."""
    pred_dir = tmp_path / "predictions" / "input"
    pred_dir.mkdir(parents=True)
    (pred_dir / "input_model_0.cif").write_bytes(b"loop_\n")
    (pred_dir / "confidence_input_model_0.json").write_text(json.dumps({
        "complex_plddt": 0.81, "ptm": 0.79, "iptm": 0.0,
        "confidence_score": 0.85,
    }))

    adapter = BoltzFoldingAdapter(_fc_mock())
    result = adapter.normalize_output("ens_fold_q__boltz", tmp_path)
    assert result.method == "boltz"
    assert len(result.structures) == 1
    assert result.structures[0].format == "cif"
    assert result.structures[0].rank == 0
    assert result.structures[0].plddt == pytest.approx(0.81)
    assert result.structures[0].url == (
        "/v1/jobs/ens_fold_q/structures/boltz/predictions/input/input_model_0.cif"
    )
    assert result.confidence["complex_plddt"] == pytest.approx(0.81)
    assert result.confidence["confidence_score"] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Promera
# ---------------------------------------------------------------------------

def test_promera_build_request_uploads_chain_keyed_schema(tmp_path):
    """promera-server expects a JSON file keyed by chain_id (uploaded as
    `input_schema`).  The adapter builds it from FoldingInput.sequences."""
    adapter = PromeraFoldingAdapter(_fc_mock())
    input = FoldingInput(
        sequences=[
            SequenceEntry(id="A", sequence="MKQH"),
            SequenceEntry(id="B", sequence="LLLL"),
        ],
        msa_mode="empty",
    )
    endpoint, payload, files = adapter.build_request(
        input, PromeraOptions(num_seeds=2, diffusion_samples=3),
    )

    assert endpoint == "/api/tasks/cofold"
    assert payload["num_seeds"] == 2
    assert payload["diffusion_samples"] == 3
    assert payload["recycling_steps"] == 4  # default
    assert "input_schema" in files
    schema_path = files["input_schema"]
    assert isinstance(schema_path, Path)
    schema = json.loads(schema_path.read_text())
    assert set(schema.keys()) == {"A", "B"}
    assert schema["A"] == {"type": "protein", "sequence": "MKQH"}
    assert schema["B"] == {"type": "protein", "sequence": "LLLL"}


def test_promera_normalize_output_reads_complex_plddt_and_chain_plddt(tmp_path):
    """promera writes ``<stem>_conf.json`` with ``complex_plddt`` (NOT ``plddt``)
    at the top level, plus a per-chain ``chain_plddt`` dict — verified against
    a v0.0.8 FC run."""
    # The orchestrator unzips into the downloaded_dir flat; in production the
    # zip structure starts at `cofold/`.
    out_dir = tmp_path / "cofold"
    out_dir.mkdir()
    (out_dir / "cofold_seed0_samp0.cif").write_bytes(b"data_pred\n")
    (out_dir / "cofold_seed0_samp0_conf.json").write_text(json.dumps({
        "complex_plddt": 0.91,
        "complex_ptm": 0.88,
        "chain_plddt": {"A": 0.92, "B": 0.85},
        "ptm": {"A": 0.88},      # per-chain dict — must NOT leak into confidence
    }))
    (out_dir / "cofold_seed0_samp1.cif").write_bytes(b"data_pred\n")
    (out_dir / "cofold_seed0_samp1_conf.json").write_text(json.dumps({
        "complex_plddt": 0.82,
        "complex_ptm": 0.75,
        "chain_plddt": {"A": 0.83, "B": 0.76},
    }))

    adapter = PromeraFoldingAdapter(_fc_mock())
    result = adapter.normalize_output("ens_fold_p__promera", tmp_path)

    assert result.method == "promera"
    assert len(result.structures) == 2
    # Sorted by complex_plddt desc — samp0 (0.91) above samp1 (0.82).
    assert result.structures[0].plddt == pytest.approx(0.91)
    assert result.structures[0].rank == 0
    assert result.structures[1].plddt == pytest.approx(0.82)
    assert result.structures[1].rank == 1
    # URLs use relative paths so the multi-segment download route can resolve them.
    assert result.structures[0].url == (
        "/v1/jobs/ens_fold_p/structures/promera/cofold/cofold_seed0_samp0.cif"
    )
    # Top-level confidence == rank-0 scalars only (no chain dict here)
    assert result.confidence["plddt"] == pytest.approx(0.91)
    assert result.confidence["ptm"] == pytest.approx(0.88)
    assert all(isinstance(v, float) for v in result.confidence.values())
    # Per-chain plddt goes in metadata (which is dict[str, Any])
    assert result.metadata["chain_plddt"] == {"A": pytest.approx(0.92), "B": pytest.approx(0.85)}


def test_promera_normalize_output_skips_trajectory_cif(tmp_path):
    """``*_traj.cif`` are multi-frame diagnostic files, not predictions —
    must not be surfaced as a structure result."""
    out_dir = tmp_path / "cofold"
    out_dir.mkdir()
    (out_dir / "cofold_seed0_samp0.cif").write_bytes(b"data_pred\n")
    (out_dir / "cofold_seed0_samp0_conf.json").write_text(
        json.dumps({"complex_plddt": 0.9, "complex_ptm": 0.8})
    )
    (out_dir / "cofold_seed0_samp0_traj.cif").write_bytes(b"data_pred\n")  # ignored

    adapter = PromeraFoldingAdapter(_fc_mock())
    result = adapter.normalize_output("ens_fold_p__promera", tmp_path)
    assert len(result.structures) == 1
    assert "traj" not in result.structures[0].url


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def test_aggregator_ranks_by_plddt_descending():
    """Three successful methods with different plddt → ranking is descending."""
    subs = [
        SubTaskRecord(
            method="alphafold", sub_task_id="x__alphafold",
            status=SubTaskStatus.SUCCEEDED,
            output={"structures": [{"plddt": 0.9, "url": "u1"}], "confidence": {}},
        ),
        SubTaskRecord(
            method="esmfold2", sub_task_id="x__esmfold2",
            status=SubTaskStatus.SUCCEEDED,
            output={"structures": [{"plddt": 0.85, "url": "u2"}], "confidence": {}},
        ),
        SubTaskRecord(
            method="boltz", sub_task_id="x__boltz",
            status=SubTaskStatus.SUCCEEDED,
            output={"structures": [{"plddt": 0.88, "url": "u3"}], "confidence": {}},
        ),
    ]
    agg = aggregate_folding(subs)
    ranking = agg["ensemble_ranking"]
    assert [r["method"] for r in ranking] == ["alphafold", "boltz", "esmfold2"]
    assert [r["overall_rank"] for r in ranking] == [0, 1, 2]
    assert agg["ensemble_score"] == pytest.approx(0.9)


def test_aggregator_skips_failed_methods():
    subs = [
        SubTaskRecord(
            method="alphafold", sub_task_id="x__alphafold",
            status=SubTaskStatus.SUCCEEDED,
            output={"structures": [{"plddt": 0.9, "url": "u1"}], "confidence": {}},
        ),
        SubTaskRecord(
            method="boltz", sub_task_id="x__boltz",
            status=SubTaskStatus.FAILED,
            error_summary="GPU timed out",
        ),
    ]
    agg = aggregate_folding(subs)
    assert len(agg["ensemble_ranking"]) == 1
    assert agg["ensemble_ranking"][0]["method"] == "alphafold"


def test_aggregator_handles_missing_plddt():
    subs = [
        SubTaskRecord(
            method="alphafold", sub_task_id="x__alphafold",
            status=SubTaskStatus.SUCCEEDED,
            output={"structures": [{"url": "u1"}], "confidence": {}},  # no plddt
        ),
    ]
    agg = aggregate_folding(subs)
    assert agg["ensemble_ranking"][0]["score"] == 0.0
    assert agg["ensemble_score"] == 0.0
