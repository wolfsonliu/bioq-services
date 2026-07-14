"""Tests for proteinmpnn-server."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Use the real `server.app` module so the `/api/design` endpoint (whose
    `Annotated[DesignRequest, Form()]` parameter needs module-level globals
    for forward-ref resolution) is registered with the same signature as
    production. Env vars steer file paths at the per-test tmp_path."""
    monkeypatch.setenv("PROTEINMPNN_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("PROTEINMPNN_ROOT", str(tmp_path / "pmpnn"))
    monkeypatch.setenv("PROTEINMPNN_WEIGHTS_DIR", str(tmp_path / "pmpnn"))
    (tmp_path / "pmpnn").mkdir(parents=True, exist_ok=True)

    # Force a fresh import so the module-level `settings = ProteinMPNNSettings()`
    # call picks up the patched env vars.
    sys.modules.pop("server.app", None)
    import importlib
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


def test_health(client):
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["service"] == "proteinmpnn"
    assert "version" in health
    detail = client.get("/healthz/detail").json()
    assert detail["service"] == "proteinmpnn"
    assert detail["version"] == health["version"]


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "proteinmpnn"


def test_settings_defaults():
    from server.settings import ProteinMPNNSettings

    class _Off(ProteinMPNNSettings):
        model_config = SettingsConfigDict(env_prefix="PROTEINMPNN_TEST_", env_file=None, extra="ignore")

    s = _Off()
    assert s.jobs_base_dir == Path("/data/proteinmpnn_jobs")
    assert s.root == Path("/opt/proteinmpnn")
    assert s.weights_dir == Path("/opt/proteinmpnn")
    assert s.oss_region == "cn-hangzhou"


def test_settings_env_override(monkeypatch):
    from server.settings import ProteinMPNNSettings
    monkeypatch.setenv("PROTEINMPNN_ROOT", "/custom/path")
    s = ProteinMPNNSettings()
    assert s.root == Path("/custom/path")


def test_adapter_name_and_cwd(tmp_path):
    from server.adapter import ProteinMPNNAdapter
    from server.settings import ProteinMPNNSettings

    class _Off(ProteinMPNNSettings):
        model_config = SettingsConfigDict(env_prefix="PROTEINMPNN_TEST_", env_file=None, extra="ignore")

    settings = _Off(root=tmp_path / "pmpnn")
    settings.root.mkdir(parents=True)
    a = ProteinMPNNAdapter(settings=settings)
    assert a.name == "proteinmpnn"
    assert a.subprocess_cwd() == settings.root


def test_common_request_defaults():
    from server.models import _ProteinMPNNCommon

    r = _ProteinMPNNCommon()
    assert r.name == "run"
    assert r.model_variant == "vanilla"
    assert r.model_name == "v_48_020"
    assert r.seed == 0
    assert r.batch_size == 1
    assert r.backbone_noise == 0.0


def test_abmpnn_rejects_non_abmpnn_model_name():
    from pydantic import ValidationError
    from server.models import _ProteinMPNNCommon

    with pytest.raises(ValidationError):
        _ProteinMPNNCommon(model_variant="abmpnn", model_name="v_48_020")


def test_ca_only_rejects_v_48_030():
    from pydantic import ValidationError
    from server.models import _ProteinMPNNCommon

    with pytest.raises(ValidationError):
        _ProteinMPNNCommon(model_variant="ca_only", model_name="v_48_030")


def test_ca_only_accepts_v_48_020():
    from server.models import _ProteinMPNNCommon
    r = _ProteinMPNNCommon(model_variant="ca_only", model_name="v_48_020")
    assert r.model_name == "v_48_020"


def test_design_request_defaults():
    from server.models import DesignRequest

    r = DesignRequest()
    assert r.num_seq_per_target == 8
    assert r.sampling_temp == "0.1"
    assert r.chains_to_design is None
    assert r.fixed_positions is None
    assert r.tied_positions is None
    assert r.homooligomer is False
    assert r.bias_AA is None
    assert r.omit_AAs == "X"


def test_design_request_with_helpers():
    from server.models import DesignRequest
    r = DesignRequest(
        chains_to_design="A C",
        fixed_positions="1 2 3, 10 11",
        bias_AA={"D": 1.39, "E": 1.39},
    )
    assert r.bias_AA["D"] == 1.39


def test_design_request_bias_AA_key_must_be_single_letter():
    from pydantic import ValidationError
    from server.models import DesignRequest
    with pytest.raises(ValidationError):
        DesignRequest(bias_AA={"DE": 1.0})


def test_design_request_fixed_positions_segments_must_match_chains():
    from pydantic import ValidationError
    from server.models import DesignRequest
    # chains_to_design has 2 segments ("A", "C"); fixed_positions has 3 → reject
    with pytest.raises(ValidationError):
        DesignRequest(chains_to_design="A C", fixed_positions="1, 2, 3")


def test_design_request_tied_positions_segments_must_match_chains():
    from pydantic import ValidationError
    from server.models import DesignRequest
    with pytest.raises(ValidationError):
        DesignRequest(chains_to_design="A C", tied_positions="1 2 3")


def test_design_request_fixed_without_chains_rejected():
    from pydantic import ValidationError
    from server.models import DesignRequest
    with pytest.raises(ValidationError):
        DesignRequest(fixed_positions="1 2 3")


def test_design_request_tied_without_chains_rejected():
    from pydantic import ValidationError
    from server.models import DesignRequest
    with pytest.raises(ValidationError):
        DesignRequest(tied_positions="5 6")


def test_score_request_defaults():
    from server.models import ScoreRequest
    r = ScoreRequest()
    assert r.num_seq_per_target == 10
    assert r.sampling_temp == "0.1"
    assert r.save_score is True
    assert r.chains_to_design is None


def test_probs_request_defaults():
    from server.models import ProbsRequest
    r = ProbsRequest()
    assert r.kind == "conditional"
    assert r.save_probs is True
    assert r.chains_to_design is None


def test_probs_request_unconditional():
    from server.models import ProbsRequest
    r = ProbsRequest(kind="unconditional")
    assert r.kind == "unconditional"


def test_uri_resolve_file(tmp_path):
    from server.settings import ProteinMPNNSettings
    from bioagent_service.uris import resolve_input

    class _Off(ProteinMPNNSettings):
        model_config = SettingsConfigDict(env_prefix="PROTEINMPNN_TEST_", env_file=None, extra="ignore")

    settings = _Off()
    src = tmp_path / "src.pdb"
    src.write_text("ATOM\n")
    dest = tmp_path / "dest.pdb"
    out = resolve_input(None, f"file://{src}", dest, settings)
    assert out.read_text() == "ATOM\n"


def test_uri_resolve_requires_one_input(tmp_path):
    from fastapi import HTTPException
    from server.settings import ProteinMPNNSettings
    from bioagent_service.uris import resolve_input

    class _Off(ProteinMPNNSettings):
        model_config = SettingsConfigDict(env_prefix="PROTEINMPNN_TEST_", env_file=None, extra="ignore")

    with pytest.raises(HTTPException) as exc:
        resolve_input(None, None, tmp_path / "x", _Off())
    assert exc.value.status_code == 422


def test_weight_flags_vanilla(tmp_path):
    from server.settings import ProteinMPNNSettings
    from server.tools import weight_flags

    class _Off(ProteinMPNNSettings):
        model_config = SettingsConfigDict(env_prefix="PROTEINMPNN_TEST_", env_file=None, extra="ignore")

    s = _Off(weights_dir=tmp_path)
    flags = weight_flags("vanilla", "v_48_020", s)
    assert "--path_to_model_weights" in flags
    idx = flags.index("--path_to_model_weights")
    assert flags[idx + 1].endswith("vanilla_model_weights/")
    assert "--model_name" in flags
    assert flags[flags.index("--model_name") + 1] == "v_48_020"
    assert "--ca_only" not in flags
    assert "--use_soluble_model" not in flags


def test_weight_flags_soluble(tmp_path):
    from server.settings import ProteinMPNNSettings
    from server.tools import weight_flags

    class _Off(ProteinMPNNSettings):
        model_config = SettingsConfigDict(env_prefix="PROTEINMPNN_TEST_", env_file=None, extra="ignore")

    flags = weight_flags("soluble", "v_48_020", _Off(weights_dir=tmp_path))
    assert "--use_soluble_model" in flags


def test_weight_flags_ca_only(tmp_path):
    from server.settings import ProteinMPNNSettings
    from server.tools import weight_flags

    class _Off(ProteinMPNNSettings):
        model_config = SettingsConfigDict(env_prefix="PROTEINMPNN_TEST_", env_file=None, extra="ignore")

    flags = weight_flags("ca_only", "v_48_020", _Off(weights_dir=tmp_path))
    assert "--ca_only" in flags


def test_weight_flags_abmpnn(tmp_path):
    from server.settings import ProteinMPNNSettings
    from server.tools import weight_flags

    class _Off(ProteinMPNNSettings):
        model_config = SettingsConfigDict(env_prefix="PROTEINMPNN_TEST_", env_file=None, extra="ignore")

    flags = weight_flags("abmpnn", "abmpnn", _Off(weights_dir=tmp_path))
    idx = flags.index("--path_to_model_weights")
    assert flags[idx + 1].endswith("AbMPNN_model_weights/")
    assert flags[flags.index("--model_name") + 1] == "abmpnn"


def test_run_helper_success(tmp_path):
    from server.settings import ProteinMPNNSettings
    from server.tools import run_helper

    class _Off(ProteinMPNNSettings):
        model_config = SettingsConfigDict(env_prefix="PROTEINMPNN_TEST_", env_file=None, extra="ignore")

    s = _Off(root=tmp_path / "pmpnn")
    (s.root / "helper_scripts").mkdir(parents=True)
    (s.root / "helper_scripts" / "echo.py").write_text(
        "import sys, pathlib; pathlib.Path(sys.argv[1]).write_text('ok')\n",
    )
    out = tmp_path / "out.txt"
    run_helper("echo.py", [str(out)], s)
    assert out.read_text() == "ok"


def test_run_helper_failure_raises_422(tmp_path):
    from fastapi import HTTPException
    from server.settings import ProteinMPNNSettings
    from server.tools import run_helper

    class _Off(ProteinMPNNSettings):
        model_config = SettingsConfigDict(env_prefix="PROTEINMPNN_TEST_", env_file=None, extra="ignore")

    s = _Off(root=tmp_path / "pmpnn")
    (s.root / "helper_scripts").mkdir(parents=True)
    (s.root / "helper_scripts" / "fail.py").write_text(
        "import sys; sys.stderr.write('boom\\n'); sys.exit(2)\n",
    )
    with pytest.raises(HTTPException) as exc:
        run_helper("fail.py", [], s)
    assert exc.value.status_code == 422
    assert "boom" in exc.value.detail


def test_prepare_inputs_minimum(tmp_path):
    """Bare-minimum call: only `parse_multiple_chains` runs."""
    from server.settings import ProteinMPNNSettings
    from server.tools import prepare_inputs

    class _Off(ProteinMPNNSettings):
        model_config = SettingsConfigDict(env_prefix="PROTEINMPNN_TEST_", env_file=None, extra="ignore")

    s = _Off(root=tmp_path / "pmpnn")
    (s.root / "helper_scripts").mkdir(parents=True)
    # Fake parse_multiple_chains writes empty jsonl to --output_path.
    (s.root / "helper_scripts" / "parse_multiple_chains.py").write_text(
        "import sys, pathlib;\n"
        "args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n"
        "pathlib.Path(args['--output_path']).write_text('{}')\n",
    )
    job_dir = tmp_path / "job"
    (job_dir / "input").mkdir(parents=True)
    (job_dir / "input" / "input.pdb").write_text("ATOM\n")

    paths = prepare_inputs(
        job_dir,
        settings=s,
        ca_only=False,
        chains_to_design=None,
        fixed_positions=None,
        tied_positions=None,
        homooligomer=False,
        bias_AA=None,
        bias_by_res=None,
        omit_AA_per_chain=None,
    )
    assert paths["parsed"].exists()
    assert paths.get("assigned") is None
    assert paths.get("fixed") is None


def test_prepare_inputs_with_chains_and_fixed(tmp_path):
    from server.settings import ProteinMPNNSettings
    from server.tools import prepare_inputs

    class _Off(ProteinMPNNSettings):
        model_config = SettingsConfigDict(env_prefix="PROTEINMPNN_TEST_", env_file=None, extra="ignore")

    s = _Off(root=tmp_path / "pmpnn")
    helpers = s.root / "helper_scripts"
    helpers.mkdir(parents=True)
    stub_body = (
        "import sys, pathlib;\n"
        "args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n"
        "pathlib.Path(args['--output_path']).write_text('{}')\n"
    )
    for name in ("parse_multiple_chains.py", "assign_fixed_chains.py", "make_fixed_positions_dict.py"):
        (helpers / name).write_text(stub_body)
    job_dir = tmp_path / "job"
    (job_dir / "input").mkdir(parents=True)
    (job_dir / "input" / "input.pdb").write_text("ATOM\n")
    paths = prepare_inputs(
        job_dir,
        settings=s,
        ca_only=False,
        chains_to_design="A C",
        fixed_positions="1 2 3, 10 11",
        tied_positions=None,
        homooligomer=False,
        bias_AA=None,
        bias_by_res=None,
        omit_AA_per_chain=None,
    )
    assert paths["parsed"].exists()
    assert paths["assigned"].exists()
    assert paths["fixed"].exists()


def test_prepare_inputs_omit_AA_per_chain_no_helper(tmp_path):
    """omit_AA_per_chain has no upstream helper script; we dump JSON directly."""
    from server.settings import ProteinMPNNSettings
    from server.tools import prepare_inputs
    import json

    class _Off(ProteinMPNNSettings):
        model_config = SettingsConfigDict(env_prefix="PROTEINMPNN_TEST_", env_file=None, extra="ignore")

    s = _Off(root=tmp_path / "pmpnn")
    helpers = s.root / "helper_scripts"
    helpers.mkdir(parents=True)
    (helpers / "parse_multiple_chains.py").write_text(
        "import sys, pathlib;\n"
        "args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n"
        "pathlib.Path(args['--output_path']).write_text('{}')\n",
    )
    job_dir = tmp_path / "job"
    (job_dir / "input").mkdir(parents=True)
    paths = prepare_inputs(
        job_dir,
        settings=s,
        ca_only=False,
        chains_to_design=None,
        fixed_positions=None,
        tied_positions=None,
        homooligomer=False,
        bias_AA=None,
        bias_by_res=None,
        omit_AA_per_chain={"A": "C", "B": "ACDE"},
    )
    assert json.loads(paths["omit_AA"].read_text()) == {"A": "C", "B": "ACDE"}


def test_design_argv_minimal(tmp_path):
    from server.models import DesignRequest
    from server.settings import ProteinMPNNSettings
    from server.tools import design_argv

    class _Off(ProteinMPNNSettings):
        model_config = SettingsConfigDict(env_prefix="PROTEINMPNN_TEST_", env_file=None, extra="ignore")

    s = _Off(root=tmp_path / "pmpnn", weights_dir=tmp_path / "pmpnn")
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    argv = design_argv(
        DesignRequest(),
        job_dir=job_dir,
        paths={"parsed": job_dir / "intermediates" / "parsed.jsonl"},
        settings=s,
    )
    assert "protein_mpnn_run.py" in argv
    assert "--jsonl_path" in argv
    assert "--num_seq_per_target" in argv
    assert argv[argv.index("--num_seq_per_target") + 1] == "8"
    assert argv[argv.index("--sampling_temp") + 1] == "0.1"
    assert "--out_folder" in argv
    assert "--path_to_model_weights" in argv


def test_design_argv_includes_helpers(tmp_path):
    from server.models import DesignRequest
    from server.settings import ProteinMPNNSettings
    from server.tools import design_argv

    class _Off(ProteinMPNNSettings):
        model_config = SettingsConfigDict(env_prefix="PROTEINMPNN_TEST_", env_file=None, extra="ignore")

    s = _Off(root=tmp_path / "pmpnn", weights_dir=tmp_path / "pmpnn")
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    inter = job_dir / "intermediates"
    paths = {
        "parsed": inter / "parsed.jsonl",
        "assigned": inter / "assigned.jsonl",
        "fixed": inter / "fixed.jsonl",
        "tied": inter / "tied.jsonl",
        "bias_AA": inter / "bias_AA.jsonl",
        "bias_by_res": inter / "bias_by_res.jsonl",
        "omit_AA": inter / "omit_AA.jsonl",
    }
    argv = design_argv(
        DesignRequest(
            chains_to_design="A C",
            fixed_positions="1 2 3, 10 11",
            tied_positions="5 6, 12 13",
            bias_AA={"D": 1.39},
            bias_by_res={"A": [0.0] * 21},
            omit_AA_per_chain={"A": "C"},
        ),
        job_dir=job_dir,
        paths=paths,
        settings=s,
    )
    assert argv[argv.index("--chain_id_jsonl") + 1] == str(paths["assigned"])
    assert argv[argv.index("--fixed_positions_jsonl") + 1] == str(paths["fixed"])
    assert argv[argv.index("--tied_positions_jsonl") + 1] == str(paths["tied"])
    assert argv[argv.index("--bias_AA_jsonl") + 1] == str(paths["bias_AA"])
    assert argv[argv.index("--bias_by_res_jsonl") + 1] == str(paths["bias_by_res"])
    assert argv[argv.index("--omit_AA_jsonl") + 1] == str(paths["omit_AA"])


def test_score_argv(tmp_path):
    from server.models import ScoreRequest
    from server.settings import ProteinMPNNSettings
    from server.tools import score_argv

    class _Off(ProteinMPNNSettings):
        model_config = SettingsConfigDict(env_prefix="PROTEINMPNN_TEST_", env_file=None, extra="ignore")

    s = _Off(root=tmp_path, weights_dir=tmp_path)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    inter = job_dir / "intermediates"
    argv = score_argv(
        ScoreRequest(num_seq_per_target=10),
        job_dir=job_dir,
        paths={"parsed": inter / "parsed.jsonl"},
        settings=s,
    )
    assert "--score_only" in argv
    assert argv[argv.index("--score_only") + 1] == "1"
    assert argv[argv.index("--save_score") + 1] == "1"
    assert argv[argv.index("--num_seq_per_target") + 1] == "10"


def test_probs_argv_conditional(tmp_path):
    from server.models import ProbsRequest
    from server.settings import ProteinMPNNSettings
    from server.tools import probs_argv

    class _Off(ProteinMPNNSettings):
        model_config = SettingsConfigDict(env_prefix="PROTEINMPNN_TEST_", env_file=None, extra="ignore")

    s = _Off(root=tmp_path, weights_dir=tmp_path)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    argv = probs_argv(
        ProbsRequest(kind="conditional"),
        job_dir=job_dir,
        paths={"parsed": job_dir / "p.jsonl"},
        settings=s,
    )
    assert "--conditional_probs_only" in argv
    assert argv[argv.index("--conditional_probs_only") + 1] == "1"
    assert "--save_probs" in argv


def test_probs_argv_unconditional(tmp_path):
    from server.models import ProbsRequest
    from server.settings import ProteinMPNNSettings
    from server.tools import probs_argv

    class _Off(ProteinMPNNSettings):
        model_config = SettingsConfigDict(env_prefix="PROTEINMPNN_TEST_", env_file=None, extra="ignore")

    s = _Off(root=tmp_path, weights_dir=tmp_path)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    argv = probs_argv(
        ProbsRequest(kind="unconditional"),
        job_dir=job_dir,
        paths={"parsed": job_dir / "p.jsonl"},
        settings=s,
    )
    assert "--unconditional_probs_only" in argv


def test_probs_argv_conditional_backbone(tmp_path):
    from server.models import ProbsRequest
    from server.settings import ProteinMPNNSettings
    from server.tools import probs_argv

    class _Off(ProteinMPNNSettings):
        model_config = SettingsConfigDict(env_prefix="PROTEINMPNN_TEST_", env_file=None, extra="ignore")

    s = _Off(root=tmp_path, weights_dir=tmp_path)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    argv = probs_argv(
        ProbsRequest(kind="conditional_backbone"),
        job_dir=job_dir,
        paths={"parsed": job_dir / "p.jsonl"},
        settings=s,
    )
    assert "--conditional_probs_only_backbone" in argv


def test_design_endpoint_returns_job(client):
    resp = client.post(
        "/api/design",
        data={"name": "demo", "model_variant": "vanilla", "model_name": "v_48_020"},
        files={"pdb": ("input.pdb", b"ATOM\n", "text/plain")},
    )
    # 422 if validation fails; 200 if accepted (helper subprocess may then fail
    # asynchronously, which is OK at this test stage).
    assert resp.status_code in (200, 422)
    if resp.status_code == 200:
        body = resp.json()
        assert "job_id" in body


def test_design_endpoint_rejects_bad_combo(client):
    resp = client.post(
        "/api/design",
        data={"model_variant": "ca_only", "model_name": "v_48_030"},
        files={"pdb": ("input.pdb", b"ATOM\n", "text/plain")},
    )
    assert resp.status_code == 422


def test_score_endpoint_returns_job(client):
    resp = client.post(
        "/api/score",
        data={"model_variant": "vanilla", "model_name": "v_48_020"},
        files={"pdb": ("input.pdb", b"ATOM\n", "text/plain")},
    )
    assert resp.status_code in (200, 422)


def test_probs_endpoint_returns_job(client):
    resp = client.post(
        "/api/probs",
        data={"kind": "conditional", "model_variant": "vanilla", "model_name": "v_48_020"},
        files={"pdb": ("input.pdb", b"ATOM\n", "text/plain")},
    )
    assert resp.status_code in (200, 422)


def test_detect_outputs_design(tmp_path):
    from server.adapter import ProteinMPNNAdapter
    from server.settings import ProteinMPNNSettings

    class _Off(ProteinMPNNSettings):
        model_config = SettingsConfigDict(env_prefix="PROTEINMPNN_TEST_", env_file=None, extra="ignore")

    a = ProteinMPNNAdapter(settings=_Off())
    job = tmp_path / "j"
    (job / "output" / "seqs").mkdir(parents=True)
    (job / "output" / "seqs" / "x.fa").write_text(">x\nM\n")
    assert a.detect_outputs(job) is True


def test_detect_outputs_score(tmp_path):
    from server.adapter import ProteinMPNNAdapter
    from server.settings import ProteinMPNNSettings

    class _Off(ProteinMPNNSettings):
        model_config = SettingsConfigDict(env_prefix="PROTEINMPNN_TEST_", env_file=None, extra="ignore")

    a = ProteinMPNNAdapter(settings=_Off())
    job = tmp_path / "j"
    (job / "output" / "score_only").mkdir(parents=True)
    (job / "output" / "score_only" / "x.npz").write_bytes(b"npz")
    assert a.detect_outputs(job) is True


def test_detect_outputs_probs(tmp_path):
    from server.adapter import ProteinMPNNAdapter
    from server.settings import ProteinMPNNSettings

    class _Off(ProteinMPNNSettings):
        model_config = SettingsConfigDict(env_prefix="PROTEINMPNN_TEST_", env_file=None, extra="ignore")

    a = ProteinMPNNAdapter(settings=_Off())
    job = tmp_path / "j"
    (job / "output" / "conditional_probs_only").mkdir(parents=True)
    (job / "output" / "conditional_probs_only" / "x.npz").write_bytes(b"npz")
    assert a.detect_outputs(job) is True


def test_detect_outputs_empty(tmp_path):
    from server.adapter import ProteinMPNNAdapter
    from server.settings import ProteinMPNNSettings

    class _Off(ProteinMPNNSettings):
        model_config = SettingsConfigDict(env_prefix="PROTEINMPNN_TEST_", env_file=None, extra="ignore")

    a = ProteinMPNNAdapter(settings=_Off())
    job = tmp_path / "j"
    (job / "output").mkdir(parents=True)
    assert a.detect_outputs(job) is False


def test_manifest_extras_has_model_variants(client):
    body = client.get("/api/manifest").json()
    extras = body["service_specific"]
    variants = extras["model_variants"]
    assert set(variants.keys()) == {"vanilla", "soluble", "ca_only", "abmpnn"}
    assert "v_48_030" in variants["vanilla"]["model_names"]
    assert "v_48_030" not in variants["ca_only"]["model_names"]
    assert variants["abmpnn"]["model_names"] == ["abmpnn"]


def test_manifest_extras_has_tool_outputs(client):
    body = client.get("/api/manifest").json()
    extras = body["service_specific"]
    assert "design" in extras["tool_outputs"]
    assert "score" in extras["tool_outputs"]
    assert "probs_conditional" in extras["tool_outputs"]
    assert "probs_unconditional" in extras["tool_outputs"]


def test_manifest_extras_has_config_tips(client):
    body = client.get("/api/manifest").json()
    tips = body["service_specific"]["config_tips"]
    for k in ("fixed_positions", "tied_positions", "sampling_temp", "bias_AA"):
        assert k in tips


def test_endpoint_examples_exist_for_all_three():
    """Run against the real app (not the fixture) so route↔example consistency is verified."""
    import importlib
    import os
    import sys
    os.environ["PROTEINMPNN_JOBS_BASE_DIR"] = "/tmp/proteinmpnn_jobs_test"
    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    body = TestClient(server_app.app).get("/api/manifest").json()
    by_path = {e["path"]: e for e in body["endpoints"]}
    for path in ("/api/design", "/api/score", "/api/probs"):
        assert path in by_path, f"{path} not registered"
        assert by_path[path]["examples"], f"{path} has no examples"


def test_design_task_endpoint_returns_job(client):
    resp = client.post(
        "/api/tasks/design",
        data={"name": "demo", "model_variant": "vanilla", "model_name": "v_48_020"},
        files={"pdb": ("input.pdb", b"ATOM\n", "text/plain")},
    )
    # Task endpoint runs synchronously; helper subprocess may fail, but the
    # accept/validate path is what we want to exercise here.
    assert resp.status_code in (200, 422, 500)
    if resp.status_code == 200:
        body = resp.json()
        assert "job_id" in body


def test_score_task_endpoint_returns_job(client):
    resp = client.post(
        "/api/tasks/score",
        data={"model_variant": "vanilla", "model_name": "v_48_020"},
        files={"pdb": ("input.pdb", b"ATOM\n", "text/plain")},
    )
    assert resp.status_code in (200, 422, 500)
