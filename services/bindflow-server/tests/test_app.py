"""Offline tests for bindflow-server (subprocess stubbed via BINDFLOW_PYTHON=/bin/true).

Real BindFlow / GROMACS is never invoked in these tests.  `shutil.which` is
monkeypatched to control what `/healthz/detail` reports.
"""

from __future__ import annotations

import importlib
import io
import sys
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _tiny_pdb_bytes() -> bytes:
    """Minimum-valid PDB text bytes (empty ATOM records suffice for form parsing)."""
    return (
        b"HEADER    TEST PROTEIN\n"
        b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        b"END\n"
    )


def _tiny_sdf_bytes(name: str = "LIG") -> bytes:
    return (
        f"{name}\n"
        "  test\n"
        "\n"
        "  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END\n"
        "$$$$\n"
    ).encode()


# ---------------------------------------------------------------------------
# Client fixture — recreates the app in an isolated tmp env.
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("BINDFLOW_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("BINDFLOW_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("BINDFLOW_PYTHON", "/bin/true")
    monkeypatch.setenv("BINDFLOW_INFERENCE_SCRIPT", str(tmp_path / "inference.py"))
    monkeypatch.setenv("BINDFLOW_WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("BINDFLOW_MAX_CONCURRENT_JOBS", "1")
    monkeypatch.setenv("BINDFLOW_TASK_ENDPOINTS_ENABLED", "false")
    (tmp_path / "root").mkdir(parents=True, exist_ok=True)

    sys.modules.pop("server.app", None)
    sys.modules.pop("server.settings", None)
    sys.modules.pop("server.adapter", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


# ---------------------------------------------------------------------------
# Health / manifest
# ---------------------------------------------------------------------------


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "bindflow"


def test_healthz_detail_reports_runtime_deps(client, monkeypatch):
    # No gmx / snakemake on PATH in test env → weights_loaded=False
    import server.app as app_mod
    monkeypatch.setattr(app_mod.shutil, "which", lambda _: None)
    body = client.get("/healthz/detail").json()
    assert body["status"] == "ok"
    assert body["gmx_available"] is False
    assert body["snakemake_available"] is False
    assert body["gmx_mmpbsa_available"] is False
    assert body["weights_loaded"] is False
    assert body["task_endpoints_enabled"] is False


def test_healthz_detail_when_all_present(client, monkeypatch):
    import server.app as app_mod

    def _fake_which(cmd):
        return {"gmx": "/opt/conda/envs/bindflow/bin/gmx",
                "snakemake": "/opt/conda/envs/bindflow/bin/snakemake",
                "gmx_MMPBSA": "/opt/conda/envs/bindflow/bin/gmx_MMPBSA"}.get(cmd)

    class _Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def _fake_run(_argv, **_kwargs):
        return _Result("GROMACS version: 2024.5\n")

    monkeypatch.setattr(app_mod.shutil, "which", _fake_which)
    monkeypatch.setattr(app_mod.subprocess, "run", _fake_run)

    body = client.get("/healthz/detail").json()
    assert body["gmx_available"] is True
    assert body["gmx_version"] == "2024.5"
    assert body["gmx_version_supported"] is True
    assert body["snakemake_available"] is True
    assert body["gmx_mmpbsa_available"] is True
    assert body["weights_loaded"] is True


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "bindflow"


def test_manifest_lists_both_endpoints(client):
    body = client.get("/api/manifest").json()
    paths = {e["path"] for e in body["endpoints"]}
    assert "/api/calculate/fep" in paths
    assert "/api/calculate/mmpbsa" in paths


def test_zip_uri_fields_match_upload_fields(client):
    """Regression: zip URI fields must be `<upload>_uri`, not the short form.

    bioq CLI's `--file custom_ff_zip=<path>` emits ``custom_ff_zip_uri`` (see
    bioq/upload.py); the endpoint must expose that exact URI field name, not the
    old ``custom_ff_uri`` / ``topology_uri``.
    """
    body = client.get("/api/manifest").json()
    eps = {e["path"]: e for e in body["endpoints"]}
    for path in ("/api/calculate/fep", "/api/calculate/mmpbsa"):
        fields = {f["name"]: f for f in eps[path]["request_fields"]}
        assert "custom_ff_zip" in fields, f"{path}: missing custom_ff_zip upload field"
        assert fields["custom_ff_zip"]["is_file"] is True, f"{path}: custom_ff_zip not marked file"
        assert "custom_ff_zip_uri" in fields, f"{path}: missing custom_ff_zip_uri URI field"
        assert "custom_ff_uri" not in fields, f"{path}: stale custom_ff_uri field still exposed"
        assert "topology_zip" in fields, f"{path}: missing topology_zip upload field"
        assert "topology_zip_uri" in fields, f"{path}: missing topology_zip_uri URI field"
        assert "topology_uri" not in fields, f"{path}: stale topology_uri field still exposed"


def test_manifest_task_endpoints_absent(client):
    """Task endpoints are disabled — /api/tasks/* must NOT appear."""
    body = client.get("/openapi.json").json()
    paths = set(body["paths"].keys())
    assert not any(p.startswith("/api/tasks/") for p in paths)


def test_manifest_extras_have_long_running_flag(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert extras["model"]["long_running"] is True
    assert "fep" in extras["tool_outputs"]
    assert "mmpbsa" in extras["tool_outputs"]


def test_manifest_examples_have_curl(client):
    body = client.get("/api/manifest").json()
    by_path = {e["path"]: e for e in body["endpoints"]}
    assert len(by_path["/api/calculate/fep"]["examples"]) >= 2
    assert any(
        "receptor.pdb" in (e.get("curl") or "")
        for e in by_path["/api/calculate/fep"]["examples"]
    )


def test_openapi_served(client):
    r = client.get("/openapi.json")
    r.raise_for_status()
    paths = r.json()["paths"]
    assert "/api/calculate/fep" in paths
    assert "/api/calculate/mmpbsa" in paths


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_missing_protein_returns_422(client):
    r = client.post(
        "/api/calculate/fep",
        files={"ligands": ("lig.sdf", _tiny_sdf_bytes(), "chemical/x-mdl-sdfile")},
        data={"water_model": "amber/tip3p"},
    )
    assert r.status_code == 422


def test_missing_ligands_returns_422(client):
    r = client.post(
        "/api/calculate/fep",
        files={"protein": ("prot.pdb", _tiny_pdb_bytes(), "chemical/x-pdb")},
    )
    assert r.status_code == 422


def test_invalid_water_model_returns_422(client):
    r = client.post(
        "/api/calculate/fep",
        files={
            "protein": ("prot.pdb", _tiny_pdb_bytes(), "chemical/x-pdb"),
            "ligands": ("lig.sdf", _tiny_sdf_bytes(), "chemical/x-mdl-sdfile"),
        },
        data={"water_model": "made-up-water"},
    )
    assert r.status_code == 422


def test_espaloma_ff_rejected(client):
    r = client.post(
        "/api/calculate/fep",
        files={
            "protein": ("prot.pdb", _tiny_pdb_bytes(), "chemical/x-pdb"),
            "ligands": ("lig.sdf", _tiny_sdf_bytes(), "chemical/x-mdl-sdfile"),
        },
        data={"ligand_ff_type": "espaloma"},
    )
    assert r.status_code == 422


def test_hmr_factor_out_of_range_returns_422(client):
    r = client.post(
        "/api/calculate/fep",
        files={
            "protein": ("prot.pdb", _tiny_pdb_bytes(), "chemical/x-pdb"),
            "ligands": ("lig.sdf", _tiny_sdf_bytes(), "chemical/x-mdl-sdfile"),
        },
        data={"hmr_factor": "10.0"},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Smoke (subprocess stubbed via /bin/true)
# ---------------------------------------------------------------------------


def test_fep_submit_returns_job(client):
    r = client.post(
        "/api/calculate/fep",
        files={
            "protein": ("prot.pdb", _tiny_pdb_bytes(), "chemical/x-pdb"),
            "ligands": ("lig_a.sdf", _tiny_sdf_bytes("A"), "chemical/x-mdl-sdfile"),
        },
        data={
            "water_model": "amber/tip3p",
            "replicas": "1",
            "threads": "2",
            "num_jobs": "1",
            "nwindows_ligand_vdw": "5",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "job_id" in body
    assert body["input_params"]["water_model"] == "amber/tip3p"
    assert body["input_params"]["replicas"] == 1
    assert body["input_params"]["nwindows_ligand_vdw"] == 5


def test_mmpbsa_submit_returns_job(client):
    r = client.post(
        "/api/calculate/mmpbsa",
        files={
            "protein": ("prot.pdb", _tiny_pdb_bytes(), "chemical/x-pdb"),
            "ligands": ("lig_a.sdf", _tiny_sdf_bytes("A"), "chemical/x-mdl-sdfile"),
        },
        data={"samples": "5", "replicas": "1"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "job_id" in body
    assert body["input_params"]["samples"] == 5


def test_ligands_zip_upload(client):
    """Zip of multiple ligands should unpack + submit."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("lig_a.sdf", _tiny_sdf_bytes("A"))
        zf.writestr("lig_b.sdf", _tiny_sdf_bytes("B"))
    buf.seek(0)

    # ligands_zip via `ligands_zip_uri=file://...` requires an on-disk path;
    # write to tmp for this test.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tf:
        tf.write(buf.getvalue())
        zip_path = tf.name
    try:
        r = client.post(
            "/api/calculate/fep",
            files={"protein": ("prot.pdb", _tiny_pdb_bytes(), "chemical/x-pdb")},
            data={
                "ligands_zip_uri": f"file://{zip_path}",
                "water_model": "amber/tip3p",
                "replicas": "1",
            },
        )
    finally:
        Path(zip_path).unlink(missing_ok=True)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "job_id" in body


def test_ligands_unsafe_filename_rejected(client):
    """Filename with shell meta must be rejected — snakemake wildcards would break."""
    r = client.post(
        "/api/calculate/fep",
        files={
            "protein": ("prot.pdb", _tiny_pdb_bytes(), "chemical/x-pdb"),
            "ligands": ("lig with spaces.sdf", _tiny_sdf_bytes("X"),
                        "chemical/x-mdl-sdfile"),
        },
        data={"replicas": "1"},
    )
    # Filename sanitization raises during submit → 500 from framework's
    # error-wrapping; the exact code is less important than "not 200".
    assert r.status_code >= 400


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_settings_defaults():
    from server.settings import BindFlowSettings

    class _Off(BindFlowSettings):
        model_config = SettingsConfigDict(
            env_prefix="BINDFLOW_TEST_",
            env_file=None,
            extra="ignore",
        )

    s = _Off()
    assert s.jobs_base_dir == Path("/data/bindflow_jobs")
    assert s.root == Path("/opt/bindflow")
    assert s.python == "/opt/conda/envs/bindflow/bin/python"
    assert s.inference_script == Path("/opt/bindflow/server/inference.py")
    assert s.weights_dir == Path("/data/models/bindflow")
    assert s.max_concurrent_jobs == 1
    # HPC-primary — task endpoints default OFF.
    assert s.task_endpoints_enabled is False
    assert s.subprocess_timeout_s == 7 * 24 * 3600


def test_settings_env_override(monkeypatch):
    from server.settings import BindFlowSettings
    monkeypatch.setenv("BINDFLOW_PYTHON", "/custom/python")
    monkeypatch.setenv("BINDFLOW_MAX_CONCURRENT_JOBS", "3")
    monkeypatch.setenv("BINDFLOW_TASK_ENDPOINTS_ENABLED", "true")
    s = BindFlowSettings()
    assert s.python == "/custom/python"
    assert s.max_concurrent_jobs == 3
    assert s.task_endpoints_enabled is True


# ---------------------------------------------------------------------------
# tools.calculate_argv
# ---------------------------------------------------------------------------


def test_calculate_argv_fep_flags(tmp_path):
    from server.models import FepCalculateRequest
    from server.settings import BindFlowSettings
    from server.tools import calculate_argv

    class _Off(BindFlowSettings):
        model_config = SettingsConfigDict(
            env_prefix="BINDFLOW_TEST_",
            env_file=None,
            extra="ignore",
        )

    s = _Off(python="/bin/true", inference_script=tmp_path / "inference.py")
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    protein = tmp_path / "p.pdb"
    protein.write_bytes(_tiny_pdb_bytes())
    ligands_dir = tmp_path / "ligs"
    ligands_dir.mkdir()
    (ligands_dir / "a.sdf").write_bytes(_tiny_sdf_bytes())

    argv = calculate_argv(
        FepCalculateRequest(replicas=2, threads=4, nwindows_ligand_vdw=7),
        calculation_type="fep",
        job_dir=job_dir,
        protein_path=protein,
        ligands_dir=ligands_dir,
        settings=s,
    )
    assert argv[0] == "/bin/true"
    assert str(s.inference_script) in argv
    assert "--calculation-type" in argv and "fep" in argv
    assert "--protein" in argv and str(protein) in argv
    assert "--ligands-dir" in argv and str(ligands_dir) in argv
    assert "--replicas" in argv and "2" in argv
    assert "--threads" in argv and "4" in argv
    assert "--nwindows-ligand-vdw" in argv and "7" in argv
    # Boolean flag defaults
    assert "--fix-protein" in argv
    assert "--submit" in argv
    # Optional not-set fields must not have their flags
    assert "--cofactor" not in argv
    assert "--membrane" not in argv


def test_calculate_argv_mmpbsa_specific(tmp_path):
    from server.models import MmpbsaCalculateRequest
    from server.settings import BindFlowSettings
    from server.tools import calculate_argv

    class _Off(BindFlowSettings):
        model_config = SettingsConfigDict(
            env_prefix="BINDFLOW_TEST_",
            env_file=None,
            extra="ignore",
        )

    s = _Off()
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    protein = tmp_path / "p.pdb"
    protein.write_bytes(_tiny_pdb_bytes())
    ligands_dir = tmp_path / "ligs"
    ligands_dir.mkdir()
    (ligands_dir / "a.sdf").write_bytes(_tiny_sdf_bytes())

    argv = calculate_argv(
        MmpbsaCalculateRequest(samples=42),
        calculation_type="mmpbsa",
        job_dir=job_dir,
        protein_path=protein,
        ligands_dir=ligands_dir,
        settings=s,
    )
    assert "--samples" in argv and "42" in argv
    # FEP-only flags absent
    assert "--nwindows-ligand-vdw" not in argv


def test_calculate_argv_optional_inputs(tmp_path):
    from server.models import FepCalculateRequest
    from server.settings import BindFlowSettings
    from server.tools import calculate_argv

    class _Off(BindFlowSettings):
        model_config = SettingsConfigDict(
            env_prefix="BINDFLOW_TEST_",
            env_file=None,
            extra="ignore",
        )

    s = _Off()
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    protein = tmp_path / "p.pdb"
    protein.write_bytes(_tiny_pdb_bytes())
    ligands_dir = tmp_path / "ligs"
    ligands_dir.mkdir()
    (ligands_dir / "a.sdf").write_bytes(_tiny_sdf_bytes())
    cofactor = tmp_path / "cof.sdf"
    cofactor.write_bytes(_tiny_sdf_bytes())
    membrane = tmp_path / "mem.pdb"
    membrane.write_bytes(_tiny_pdb_bytes())

    argv = calculate_argv(
        FepCalculateRequest(fix_protein=False, hmr_factor=None),
        calculation_type="fep",
        job_dir=job_dir,
        protein_path=protein,
        ligands_dir=ligands_dir,
        cofactor_path=cofactor,
        membrane_path=membrane,
        settings=s,
    )
    assert "--cofactor" in argv and str(cofactor) in argv
    assert "--membrane" in argv and str(membrane) in argv
    # Boolean flip
    assert "--no-fix-protein" in argv
    # hmr_factor=None → no flag
    assert "--hmr-factor" not in argv


def test_calculate_argv_writes_yaml(tmp_path):
    from server.models import FepCalculateRequest
    from server.settings import BindFlowSettings
    from server.tools import calculate_argv

    class _Off(BindFlowSettings):
        model_config = SettingsConfigDict(
            env_prefix="BINDFLOW_TEST_",
            env_file=None,
            extra="ignore",
        )

    s = _Off()
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    protein = tmp_path / "p.pdb"
    protein.write_bytes(_tiny_pdb_bytes())
    ligands_dir = tmp_path / "ligs"
    ligands_dir.mkdir()
    (ligands_dir / "a.sdf").write_bytes(_tiny_sdf_bytes())

    yaml_text = "cluster:\n  options:\n    calculation:\n      partition: cpu\n"
    argv = calculate_argv(
        FepCalculateRequest(global_config_yaml=yaml_text),
        calculation_type="fep",
        job_dir=job_dir,
        protein_path=protein,
        ligands_dir=ligands_dir,
        settings=s,
    )
    yaml_path = job_dir / "input" / "global_config.yaml"
    assert "--global-config-yaml" in argv
    assert str(yaml_path) in argv
    assert yaml_path.exists()
    assert "partition: cpu" in yaml_path.read_text()


def test_list_ligands_rejects_empty_dir(tmp_path):
    from server.tools import list_ligands
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(ValueError, match="no ligand"):
        list_ligands(d)


def test_sanitize_ligand_filename():
    from server.tools import sanitize_ligand_filename
    assert sanitize_ligand_filename("lig_a.sdf") == "lig_a.sdf"
    assert sanitize_ligand_filename("Lig-1.mol2") == "Lig-1.mol2"
    with pytest.raises(ValueError):
        sanitize_ligand_filename("lig with spaces.sdf")
    with pytest.raises(ValueError):
        sanitize_ligand_filename("lig;rm -rf /.sdf")


# ---------------------------------------------------------------------------
# inference.py: validation + config assembly
# ---------------------------------------------------------------------------


def test_inference_parse_and_validate(tmp_path):
    """inference.py's cheap validation runs without importing bindflow."""
    from server import inference

    protein = tmp_path / "p.pdb"
    protein.write_bytes(_tiny_pdb_bytes())
    ligands_dir = tmp_path / "ligs"
    ligands_dir.mkdir()
    (ligands_dir / "a.sdf").write_bytes(_tiny_sdf_bytes())

    args = inference.parse_args([
        "--calculation-type", "fep",
        "--protein", str(protein),
        "--ligands-dir", str(ligands_dir),
        "--output-dir", str(tmp_path / "out"),
        "--water-model", "amber/tip3p",
        "--ligand-ff-type", "openff",
    ])
    inference.validate(args)
    assert args._ligand_files == [ligands_dir / "a.sdf"]


def test_inference_hmr_dt_constraint(tmp_path):
    from server import inference

    protein = tmp_path / "p.pdb"
    protein.write_bytes(_tiny_pdb_bytes())
    ligands_dir = tmp_path / "ligs"
    ligands_dir.mkdir()
    (ligands_dir / "a.sdf").write_bytes(_tiny_sdf_bytes())

    # hmr_factor=1.5 (<2) with dt_max=0.004 (>0.002) → SystemExit
    args = inference.parse_args([
        "--calculation-type", "fep",
        "--protein", str(protein),
        "--ligands-dir", str(ligands_dir),
        "--output-dir", str(tmp_path / "out"),
        "--water-model", "amber/tip3p",
        "--ligand-ff-type", "openff",
        "--hmr-factor", "1.5",
        "--dt-max", "0.004",
    ])
    with pytest.raises(SystemExit):
        inference.validate(args)


def test_inference_build_global_config_fep(tmp_path):
    from server import inference

    protein = tmp_path / "p.pdb"
    protein.write_bytes(_tiny_pdb_bytes())
    ligands_dir = tmp_path / "ligs"
    ligands_dir.mkdir()
    (ligands_dir / "a.sdf").write_bytes(_tiny_sdf_bytes())

    args = inference.parse_args([
        "--calculation-type", "fep",
        "--protein", str(protein),
        "--ligands-dir", str(ligands_dir),
        "--output-dir", str(tmp_path / "out"),
        "--water-model", "amber/tip3p",
        "--ligand-ff-type", "openff",
        "--nwindows-ligand-vdw", "7",
        "--nwindows-complex-vdw", "15",
    ])
    inference.validate(args)
    cfg = inference.build_global_config(args)
    assert cfg["nwindows"]["ligand"]["vdw"] == 7
    assert cfg["nwindows"]["complex"]["vdw"] == 15
    assert cfg["cluster"]["options"]["calculation"] == {}


def test_inference_build_ligands_list(tmp_path):
    from server import inference

    protein = tmp_path / "p.pdb"
    protein.write_bytes(_tiny_pdb_bytes())
    ligands_dir = tmp_path / "ligs"
    ligands_dir.mkdir()
    (ligands_dir / "a.sdf").write_bytes(_tiny_sdf_bytes())
    (ligands_dir / "b.mol").write_bytes(_tiny_sdf_bytes("B"))

    args = inference.parse_args([
        "--calculation-type", "fep",
        "--protein", str(protein),
        "--ligands-dir", str(ligands_dir),
        "--output-dir", str(tmp_path / "out"),
        "--water-model", "amber/tip3p",
        "--ligand-ff-type", "gaff",
        "--ligand-ff-code", "gaff-2.11",
    ])
    inference.validate(args)
    ligs = inference.build_ligands_list(args)
    assert len(ligs) == 2
    assert all(li["ff"]["type"] == "gaff" for li in ligs)
    assert all(li["ff"]["code"] == "gaff-2.11" for li in ligs)


def test_inference_deep_merge():
    from server.inference import _deep_merge
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"b": 10}, "e": 4}
    merged = _deep_merge(base, override)
    assert merged == {"a": {"b": 10, "c": 2}, "d": 3, "e": 4}
    # base unchanged
    assert base == {"a": {"b": 1, "c": 2}, "d": 3}
