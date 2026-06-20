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


def test_alphafold_normalize_output_picks_ranked_pdbs(tmp_path):
    # Simulate alphafold output dir with 3 ranked PDBs
    for i in range(3):
        (tmp_path / f"ranked_{i}.pdb").write_bytes(b"REMARK fake pdb\nATOM\nEND\n")

    adapter = AlphaFoldFoldingAdapter(_fc_mock())
    result = adapter.normalize_output(
        "ens_fold_xyz__alphafold", tmp_path,
    )
    assert result.method == "alphafold"
    assert result.status == "completed"
    assert result.fc_job_id == "ens_fold_xyz__alphafold"
    assert len(result.structures) == 3
    for i, s in enumerate(result.structures):
        assert s.format == "pdb"
        assert s.rank == i
        assert s.url == f"/v1/jobs/ens_fold_xyz/structures/alphafold/ranked_{i}.pdb"


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
    (tmp_path / "prediction_0.cif").write_bytes(b"loop_\nfake cif\n")
    (tmp_path / "metrics.json").write_text(json.dumps({"mean_plddt": 0.87, "ptm": 0.92}))

    adapter = ESMFold2FoldingAdapter(_fc_mock())
    result = adapter.normalize_output("ens_fold_abc__esmfold2", tmp_path)
    assert result.method == "esmfold2"
    assert len(result.structures) == 1
    assert result.structures[0].format == "cif"
    assert result.structures[0].plddt == pytest.approx(0.87)
    assert result.confidence["mean_plddt"] == pytest.approx(0.87)
    assert result.confidence["ptm"] == pytest.approx(0.92)


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
    assert files == {}


def test_boltz_normalize_output_picks_cif_files(tmp_path):
    (tmp_path / "predictions").mkdir()
    (tmp_path / "predictions" / "model_0.cif").write_bytes(b"loop_\n")
    (tmp_path / "predictions" / "confidence_0.json").write_text(json.dumps({"plddt": 0.81, "ptm": 0.79}))

    adapter = BoltzFoldingAdapter(_fc_mock())
    result = adapter.normalize_output("ens_fold_q__boltz", tmp_path)
    assert result.method == "boltz"
    assert len(result.structures) == 1
    assert result.structures[0].format == "cif"
    assert result.confidence.get("plddt") == pytest.approx(0.81)
    assert result.structures[0].plddt == pytest.approx(0.81)


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
