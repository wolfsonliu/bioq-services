"""CLI batch-mode tests for openadmet-server.

Tests endpoint registration, argv builders (predict/compare + shell wrapper),
and end-to-end create_cli.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic_settings import SettingsConfigDict

from server.models import CompareRequest, PredictRequest
from server.settings import ModelInfo, OpenAdmetSettings
from server.tools import (
    augment_csv_with_aliases,
    build_predict_shell,
    compare_argv_mode_a,
    compare_argv_mode_b,
    group_models_by_input_col,
    predict_argv,
    predict_composite_argv,
    sniff_smiles_column,
    split_inline_smiles,
    write_alias_csv,
)


LOSARTAN = "CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl"


class _Off(OpenAdmetSettings):
    """Test-scoped settings that ignore .env / real environment."""
    model_config = SettingsConfigDict(
        env_prefix="OPENADMET_TEST_",
        env_file=None,
        extra="ignore",
    )


def _fake_model(models_root: Path, name: str, input_col: str,
                target_cols: list[str], biotargets: list[str]) -> ModelInfo:
    model_dir = models_root / name
    (model_dir / "recipe_components").mkdir(parents=True, exist_ok=True)
    (model_dir / "recipe_components" / "metadata.yaml").write_text(yaml.safe_dump({
        "biotargets": biotargets, "tag": name, "build_number": 0,
        "description": "test", "name": "t",
    }))
    (model_dir / "recipe_components" / "data.yaml").write_text(yaml.safe_dump({
        "input_col": input_col, "target_cols": target_cols,
    }))
    (model_dir / "recipe_components" / "procedure.yaml").write_text(yaml.safe_dump({
        "model": {"type": "ChemPropModel"}, "feat": {"type": "ChemPropFeaturizer"},
    }))
    (model_dir / "model.pth").write_bytes(b"\x00")
    return ModelInfo(
        name=name, path=model_dir, input_col=input_col,
        target_cols=target_cols, biotargets=biotargets,
        tag=name, description="", model_type="ChemPropModel",
        feat_type="ChemPropFeaturizer", build_number=0,
    )


# ===== SMILES parsing / CSV prep ============================================


def test_split_inline_smiles_comma_separated():
    result = split_inline_smiles(f"{LOSARTAN},CC(=O)O")
    assert result == [LOSARTAN, "CC(=O)O"]


def test_split_inline_smiles_whitespace_and_newlines():
    result = split_inline_smiles("CCO\nCC(=O)O\t c1ccccc1")
    assert result == ["CCO", "CC(=O)O", "c1ccccc1"]


def test_split_inline_smiles_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        split_inline_smiles("   ,,,,   ")


def test_split_inline_smiles_max_n_enforced():
    payload = ",".join(["CCO"] * 250)
    with pytest.raises(ValueError, match="max 200"):
        split_inline_smiles(payload, max_n=200)


def test_write_alias_csv_has_all_alias_columns(tmp_path):
    aliases = ["OPENADMET_SMILES", "OPENADMET_CANONICAL_SMILES", "smiles"]
    dest = write_alias_csv(["CCO", "CC(=O)O"], tmp_path / "in.csv", aliases)
    text = dest.read_text()
    header = text.splitlines()[0].split(",")
    assert header == aliases
    row1 = text.splitlines()[1].split(",")
    assert row1 == ["CCO", "CCO", "CCO"]


def test_sniff_smiles_column_returns_first_match(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("junk,OPENADMET_CANONICAL_SMILES,other\n1,CCO,2\n")
    got = sniff_smiles_column(
        p, ["OPENADMET_SMILES", "OPENADMET_CANONICAL_SMILES", "smiles"]
    )
    assert got == "OPENADMET_CANONICAL_SMILES"


def test_sniff_smiles_column_returns_none_when_absent(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("foo,bar\n1,2\n")
    assert sniff_smiles_column(p, ["OPENADMET_SMILES"]) is None


def test_augment_csv_with_aliases_adds_missing_columns(tmp_path):
    src = tmp_path / "src.csv"
    src.write_text("OPENADMET_CANONICAL_SMILES,extra\nCCO,x\nCC,y\n")
    dst = augment_csv_with_aliases(
        src, tmp_path / "aug.csv",
        aliases=["OPENADMET_CANONICAL_SMILES", "OPENADMET_SMILES", "smiles"],
        detected_col="OPENADMET_CANONICAL_SMILES",
    )
    lines = dst.read_text().splitlines()
    header = lines[0].split(",")
    # extra + canonical (kept) + new aliases added at the end
    assert set(header) == {
        "OPENADMET_CANONICAL_SMILES", "extra", "OPENADMET_SMILES", "smiles",
    }
    row1 = dict(zip(header, lines[1].split(",")))
    assert row1["OPENADMET_SMILES"] == "CCO"
    assert row1["smiles"] == "CCO"


# ===== input_col grouping ====================================================


def test_group_models_same_input_col(tmp_path):
    m1 = _fake_model(tmp_path, "m1", "OPENADMET_CANONICAL_SMILES", ["y1"], ["A"])
    m2 = _fake_model(tmp_path, "m2", "OPENADMET_CANONICAL_SMILES", ["y2"], ["B"])
    groups = group_models_by_input_col([m1, m2], None)
    assert set(groups.keys()) == {"OPENADMET_CANONICAL_SMILES"}
    assert len(groups["OPENADMET_CANONICAL_SMILES"]) == 2


def test_group_models_different_input_col(tmp_path):
    m1 = _fake_model(tmp_path, "m1", "OPENADMET_CANONICAL_SMILES", ["y1"], ["A"])
    m2 = _fake_model(tmp_path, "m2", "OPENADMET_SMILES", ["y2"], ["B"])
    groups = group_models_by_input_col([m1, m2], None)
    assert set(groups.keys()) == {"OPENADMET_CANONICAL_SMILES", "OPENADMET_SMILES"}


def test_group_models_override_col_flattens(tmp_path):
    m1 = _fake_model(tmp_path, "m1", "OPENADMET_CANONICAL_SMILES", ["y1"], ["A"])
    m2 = _fake_model(tmp_path, "m2", "OPENADMET_SMILES", ["y2"], ["B"])
    groups = group_models_by_input_col([m1, m2], "custom_col")
    assert set(groups.keys()) == {"custom_col"}
    assert len(groups["custom_col"]) == 2


# ===== argv builders ========================================================


def test_predict_argv_has_required_flags(tmp_path):
    s = _Off(python="/bin/python", weights_dir=tmp_path / "w")
    req = PredictRequest(
        input_smiles=LOSARTAN, model_names=["m1"],
        accelerator="cpu",
    )
    argv = predict_argv(
        req,
        input_path=tmp_path / "in.csv",
        input_col="OPENADMET_CANONICAL_SMILES",
        output_csv=tmp_path / "out.csv",
        model_dirs=[tmp_path / "mA"],
        settings=s,
    )
    assert argv[0] == "/bin/python"
    assert "openadmet.models.cli.cli" in argv
    assert "predict" in argv
    assert "--input-path" in argv
    assert "--input-col" in argv and "OPENADMET_CANONICAL_SMILES" in argv
    assert "--accelerator" in argv and "cpu" in argv
    assert "--model-dir" in argv and str(tmp_path / "mA") in argv


def test_predict_argv_multiple_models(tmp_path):
    s = _Off(python="/bin/python")
    req = PredictRequest(input_smiles=LOSARTAN, model_names=["m1", "m2"])
    argv = predict_argv(
        req,
        input_path=tmp_path / "in.csv",
        input_col="OPENADMET_CANONICAL_SMILES",
        output_csv=tmp_path / "out.csv",
        model_dirs=[tmp_path / "mA", tmp_path / "mB"],
        settings=s,
    )
    # Two `--model-dir` flags.
    md_positions = [i for i, x in enumerate(argv) if x == "--model-dir"]
    assert len(md_positions) == 2


def test_predict_argv_acquisition_flags(tmp_path):
    s = _Off()
    req = PredictRequest(
        input_smiles=LOSARTAN, model_names=["m1"],
        aq_fxns=["ucb", "ei"], beta=[1.5], best_y=[6.0], xi=[0.01],
    )
    argv = predict_argv(
        req,
        input_path=tmp_path / "in.csv",
        input_col="OPENADMET_SMILES",
        output_csv=tmp_path / "out.csv",
        model_dirs=[tmp_path / "mA"],
        settings=s,
    )
    assert "--aq-fxn" in argv and "ucb" in argv and "ei" in argv
    assert "--beta" in argv and "1.5" in argv
    assert "--best-y" in argv and "6.0" in argv
    assert "--xi" in argv and "0.01" in argv


def test_predict_composite_argv_groups(tmp_path):
    s = _Off(weights_dir=tmp_path / "w")
    m1 = _fake_model(tmp_path, "m1", "OPENADMET_CANONICAL_SMILES", ["y1"], ["A"])
    m2 = _fake_model(tmp_path, "m2", "OPENADMET_SMILES", ["y2"], ["B"])
    req = PredictRequest(input_smiles=LOSARTAN, model_names=["m1", "m2"])
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    argvs = predict_composite_argv(
        req, input_csv=tmp_path / "in.csv", job_dir=job_dir,
        settings=s, models=[m1, m2],
    )
    assert len(argvs) == 2
    input_cols_used = []
    for argv in argvs:
        i = argv.index("--input-col")
        input_cols_used.append(argv[i + 1])
    assert set(input_cols_used) == {"OPENADMET_CANONICAL_SMILES", "OPENADMET_SMILES"}


def test_build_predict_shell_wraps_argvs(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    argvs = [
        ["/bin/echo", "a"],
        ["/bin/echo", "b"],
    ]
    shell = build_predict_shell(argvs, output_dir=output_dir)
    assert shell[0] == "bash"
    assert shell[1] == "-c"
    # Merge step calls python -c
    assert "python -c" in shell[2]
    # Both argv commands appear before the merge.
    assert shell[2].index("echo a") < shell[2].index("python -c")


def test_build_predict_shell_embedded_python_is_syntactically_valid(tmp_path):
    """Regression: v0.0.2 shipped with `shlex.quote` on the path inside the
    embedded python -c code, producing invalid Python for plain /-paths.
    See engineering/decisions/2026-07-05-openadmet-server-design.md and
    the failed job at 2026-07-06 (probe-1783378692 → SyntaxError)."""
    import ast, shlex

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    shell = build_predict_shell([["/bin/echo", "x"]], output_dir=output_dir)
    joined = shell[2]

    # Extract the `python -c '...'` payload back out via shell tokenization.
    tokens = shlex.split(joined)
    py_arg_idx = tokens.index("-c") + 1
    py_code = tokens[py_arg_idx]

    # This must parse. Old broken output emitted `glob.glob(/tmp/... + "/...")`
    # which raises SyntaxError immediately.
    ast.parse(py_code)


def test_compare_argv_mode_a(tmp_path):
    s = _Off()
    req = CompareRequest(
        model_names=["m1", "m2"],
        label_types=["biotarget", "biotarget"],
        mt_id="CYP3A4",
        report=True,
    )
    argv = compare_argv_mode_a(
        req,
        output_dir=tmp_path / "out",
        model_dirs=[tmp_path / "mA", tmp_path / "mB"],
        settings=s,
    )
    assert "compare" in argv
    md_flags = [i for i, x in enumerate(argv) if x == "--model-dirs"]
    lt_flags = [i for i, x in enumerate(argv) if x == "--label-types"]
    assert len(md_flags) == 2
    assert len(lt_flags) == 2
    assert "--mt-id" in argv and "CYP3A4" in argv
    assert "--report" in argv


def test_compare_argv_mode_b(tmp_path):
    s = _Off()
    req = CompareRequest(
        labels=["a", "b"],
        task_names=["t1", "t2"],
    )
    argv = compare_argv_mode_b(
        req,
        output_dir=tmp_path / "out",
        stats_files=[tmp_path / "a.json", tmp_path / "b.json"],
        settings=s,
    )
    stats_flags = [i for i, x in enumerate(argv) if x == "--model-stats-fns"]
    labels_flags = [i for i, x in enumerate(argv) if x == "--labels"]
    task_flags = [i for i, x in enumerate(argv) if x == "--task-names"]
    assert len(stats_flags) == 2
    assert len(labels_flags) == 2
    assert len(task_flags) == 2


# ===== Pydantic model validators ============================================


def test_predict_request_rejects_acquisition_mismatch():
    with pytest.raises(ValueError, match="beta"):
        PredictRequest(
            input_smiles=LOSARTAN, model_names=["m"],
            aq_fxns=["ucb"],  # missing beta
        )


def test_predict_request_rejects_duplicate_aq_fxn():
    with pytest.raises(ValueError, match="only be specified once"):
        PredictRequest(
            input_smiles=LOSARTAN, model_names=["m"],
            aq_fxns=["ucb", "ucb"], beta=[1.0, 2.0],
        )


def test_compare_request_rejects_both_modes():
    with pytest.raises(ValueError, match="mutually exclusive"):
        CompareRequest(
            model_names=["a", "b"], labels=["a", "b"], task_names=["t", "t"],
        )


def test_compare_request_rejects_neither_mode():
    with pytest.raises(ValueError, match="either"):
        CompareRequest()


def test_compare_request_mode_a_needs_two_models():
    with pytest.raises(ValueError, match="≥ 2"):
        CompareRequest(model_names=["only-one"])


# ===== Endpoint registration (mirror of __main__.py) =========================
# We redefine the endpoint dict locally instead of importing from server.__main__
# because that module calls create_cli() at import, which reads pytest's argv.


def _predict_build_dummy(req, inputs, job_dir, settings):
    return ["/bin/true"]


def _compare_build_dummy(req, inputs, job_dir, settings):
    return ["/bin/true"]


from bioagent_service.cli import CLIEndpoint  # noqa: E402


ENDPOINTS = {
    "predict": CLIEndpoint(
        name="predict",
        help="Predict",
        request_model=PredictRequest,
        build_argv=_predict_build_dummy,
        inputs={
            "input_csv": ("Input CSV", False),
            "input_sdf": ("Input SDF", False),
        },
    ),
    "compare": CLIEndpoint(
        name="compare",
        help="Compare",
        request_model=CompareRequest,
        build_argv=_compare_build_dummy,
        inputs={
            "stats_file_0": ("Stats slot 0", False),
            "stats_file_1": ("Stats slot 1", False),
            "stats_file_2": ("Stats slot 2", False),
            "stats_file_3": ("Stats slot 3", False),
        },
    ),
}


def test_endpoint_keys_match_design():
    assert set(ENDPOINTS.keys()) == {"predict", "compare"}


def test_predict_endpoint_request_model():
    assert ENDPOINTS["predict"].request_model is PredictRequest
    assert "input_csv" in ENDPOINTS["predict"].inputs
    assert "input_sdf" in ENDPOINTS["predict"].inputs
    assert ENDPOINTS["predict"].inputs["input_csv"][1] is False


def test_compare_endpoint_declares_stats_file_slots():
    assert "stats_file_0" in ENDPOINTS["compare"].inputs
    assert "stats_file_1" in ENDPOINTS["compare"].inputs
    assert "stats_file_2" in ENDPOINTS["compare"].inputs
    assert "stats_file_3" in ENDPOINTS["compare"].inputs


# ===== End-to-end create_cli =================================================


def test_cli_no_subcommand_exits_nonzero(tmp_path):
    """Running the CLI without a subcommand should exit non-zero."""
    from server.adapter import OpenAdmetAdapter
    from bioagent_service.cli import create_cli

    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = OpenAdmetAdapter(settings=s)

    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit) as exc:
            create_cli(adapter, s, ENDPOINTS)
        assert exc.value.code != 0
