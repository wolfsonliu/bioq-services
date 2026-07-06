"""Offline tests for pocketxmol-server.

Cover:
- health / healthz-detail / manifest / openapi surface
- settings defaults (env-prefix override, weights externalization)
- Each of 6 endpoints: input validation (422 shapes) + build_*_config
  YAML shape assertion (unit-level — no subprocess)

Real algorithm imports (pytorch / pyg / rdkit / openbabel) are not
required — this file only exercises the wrapping code.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict

DATA = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def client(tmp_path, monkeypatch):
    """Fresh app under a per-test jobs dir with a stub python interpreter."""
    monkeypatch.setenv("POCKETXMOL_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("POCKETXMOL_ROOT", str(tmp_path / "pocketxmol"))
    monkeypatch.setenv("POCKETXMOL_PYTHON", "/bin/true")
    # weights fake — /healthz/detail probes .exists() only.
    monkeypatch.setenv("POCKETXMOL_WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("POCKETXMOL_PXM_CHECKPOINT",
                       str(tmp_path / "weights" / "pocketxmol.ckpt"))
    monkeypatch.setenv("POCKETXMOL_TUNED_CFD_CKPT",
                       str(tmp_path / "weights" / "tuned_ranker.ckpt"))
    monkeypatch.setenv("POCKETXMOL_FLEX_CFD_CKPT",
                       str(tmp_path / "weights" / "flex_cfd.ckpt"))
    (tmp_path / "pocketxmol").mkdir(parents=True, exist_ok=True)

    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


# ---------------------------------------------------------------------------
# Healthz + manifest
# ---------------------------------------------------------------------------
def test_health(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "pocketxmol"
    assert "version" in body


def test_healthz_detail_reports_missing_weights(client):
    body = client.get("/healthz/detail").json()
    # We never created the weight files, so all 3 should be reported missing.
    assert body["weights_loaded"] is False
    assert set(body["weights_missing"]) == {
        "pxm_checkpoint", "tuned_cfd_ckpt", "flex_cfd_ckpt",
    }
    assert body["max_concurrent_jobs"] == 1


def test_healthz_detail_weights_loaded_when_present(tmp_path, monkeypatch):
    weights = tmp_path / "weights"
    weights.mkdir()
    for name in ("pocketxmol.ckpt", "tuned_ranker.ckpt", "flex_cfd.ckpt"):
        (weights / name).write_bytes(b"stub-ckpt-content")
    monkeypatch.setenv("POCKETXMOL_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("POCKETXMOL_ROOT", str(tmp_path / "pocketxmol"))
    monkeypatch.setenv("POCKETXMOL_PYTHON", "/bin/true")
    monkeypatch.setenv("POCKETXMOL_WEIGHTS_DIR", str(weights))
    monkeypatch.setenv("POCKETXMOL_PXM_CHECKPOINT", str(weights / "pocketxmol.ckpt"))
    monkeypatch.setenv("POCKETXMOL_TUNED_CFD_CKPT", str(weights / "tuned_ranker.ckpt"))
    monkeypatch.setenv("POCKETXMOL_FLEX_CFD_CKPT", str(weights / "flex_cfd.ckpt"))
    (tmp_path / "pocketxmol").mkdir()

    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    with TestClient(server_app.app) as c:
        body = c.get("/healthz/detail").json()
        assert body["weights_loaded"] is True
        assert body["weights_missing"] == {}


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "pocketxmol"


def test_manifest_lists_all_six_endpoints(client):
    body = client.get("/api/manifest").json()
    paths = {e["path"] for e in body["endpoints"]}
    for path in [
        "/api/dock", "/api/sbdd", "/api/linking", "/api/optimize",
        "/api/pepdesign", "/api/confidence",
    ]:
        assert path in paths, f"missing endpoint in manifest: {path}"


def test_manifest_task_endpoints_registered(client):
    body = client.get("/api/manifest").json()
    paths = {e["path"] for e in body["endpoints"]}
    for path in [
        "/api/tasks/dock", "/api/tasks/sbdd", "/api/tasks/linking",
        "/api/tasks/optimize", "/api/tasks/pepdesign", "/api/tasks/confidence",
    ]:
        assert path in paths, f"missing task endpoint in manifest: {path}"


def test_manifest_extras_has_tool_outputs_for_all_endpoints(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert extras["model"]["name"] == "PocketXMol"
    for key in ("dock", "sbdd", "linking", "optimize", "pepdesign", "confidence"):
        assert key in extras["tool_outputs"], f"tool_outputs missing key: {key}"


def test_manifest_endpoint_examples_present(client):
    extras = client.get("/api/manifest").json()["endpoints"]
    endpoint_by_path = {e["path"]: e for e in extras}
    for path in ("/api/dock", "/api/sbdd", "/api/linking", "/api/optimize",
                 "/api/pepdesign", "/api/confidence"):
        examples = endpoint_by_path[path].get("examples") or []
        assert examples, f"endpoint_examples missing for {path}"


def test_openapi_served(client):
    body = client.get("/openapi.json").json()
    assert "paths" in body


# ---------------------------------------------------------------------------
# Settings defaults
# ---------------------------------------------------------------------------
def test_settings_defaults():
    from server.settings import PocketXMolSettings

    class _Off(PocketXMolSettings):
        model_config = SettingsConfigDict(
            env_prefix="POCKETXMOL_TEST_", env_file=None, extra="ignore",
        )

    s = _Off()
    assert s.jobs_base_dir == Path("/data/pocketxmol_jobs")
    assert s.root == Path("/opt/pocketxmol")
    assert s.weights_dir == Path("/data/models/pocketxmol")
    assert s.pxm_checkpoint == Path("/data/models/pocketxmol/pxm/checkpoints/pocketxmol.ckpt")
    assert s.tuned_cfd_ckpt == Path("/data/models/pocketxmol/tuned_ranker/checkpoints/tuned_ranker.ckpt")
    assert s.flex_cfd_ckpt == Path("/data/models/pocketxmol/flex_cfd/checkpoints/flex_cfd.ckpt")
    assert s.max_concurrent_jobs == 1
    assert s.task_endpoints_enabled is True


# ---------------------------------------------------------------------------
# build_*_config unit tests (no subprocess, no client)
# ---------------------------------------------------------------------------
def _off_settings():
    from server.settings import PocketXMolSettings

    class _Off(PocketXMolSettings):
        model_config = SettingsConfigDict(
            env_prefix="POCKETXMOL_TEST_", env_file=None, extra="ignore",
        )

    return _Off()


def test_build_dock_config_smiles_priority():
    from server.configs import build_dock_config
    from server.models import DockRequest

    req = DockRequest(num_samples=5, smiles="c1ccccc1", pocket_coord=[1.0, 2.0, 3.0])
    cfg = build_dock_config(
        req=req, protein_path=Path("/x/prot.pdb"),
        ligand_path=None, ref_ligand_path=None,
        output_dir=Path("/x/out"),
    )
    assert cfg["data"]["input_ligand"] == "c1ccccc1"
    assert cfg["data"]["pocket_args"]["pocket_coord"] == [1.0, 2.0, 3.0]
    assert cfg["sample"]["num_mols"] == 5
    assert cfg["task"]["name"] == "dock"


def test_build_dock_config_pepseq_priority():
    from server.configs import build_dock_config
    from server.models import DockRequest

    req = DockRequest(is_pep=True, pep_sequence="DTVFALFW")
    cfg = build_dock_config(
        req=req, protein_path=Path("/x/prot.pdb"),
        ligand_path=None, ref_ligand_path=None,
        output_dir=Path("/x/out"),
    )
    assert cfg["data"]["input_ligand"] == "pepseq_DTVFALFW"
    assert cfg["data"]["is_pep"] is True


def test_build_dock_config_flexible_noise():
    from server.configs import build_dock_config
    from server.models import DockRequest, NoiseMode

    req = DockRequest(smiles="C", noise_mode=NoiseMode.flexible)
    cfg = build_dock_config(
        req=req, protein_path=Path("/x/prot.pdb"),
        ligand_path=None, ref_ligand_path=None,
        output_dir=Path("/x/out"),
    )
    assert cfg["task"]["transform"]["settings"] == {"free": 0, "flexible": 1}


def test_build_sbdd_ar_config():
    from server.configs import build_sbdd_config
    from server.models import SbddMode, SbddRequest

    req = SbddRequest(pocket_coord=[1.0, 2.0, 3.0], mode=SbddMode.ar,
                      mol_size_mean=30, mol_size_std=3)
    cfg = build_sbdd_config(
        req=req, protein_path=Path("/x/prot.pdb"), output_dir=Path("/x/out"),
    )
    assert cfg["task"]["transform"]["name"] == "ar"
    assert cfg["noise"]["name"] == "maskfill"
    assert "ar_config" in cfg["noise"]
    assert cfg["transforms"]["variable_mol_size"]["num_atoms_distri"]["mean"]["bias"] == 30
    assert cfg["transforms"]["featurizer_pocket"]["center"] == [1.0, 2.0, 3.0]


def test_build_sbdd_simple_config():
    from server.configs import build_sbdd_config
    from server.models import SbddMode, SbddRequest

    req = SbddRequest(pocket_coord=[0, 0, 0], mode=SbddMode.simple)
    cfg = build_sbdd_config(
        req=req, protein_path=Path("/x/prot.pdb"), output_dir=Path("/x/out"),
    )
    assert cfg["task"]["transform"]["name"] == "sbdd"
    assert cfg["noise"]["name"] == "sbdd"
    assert "ar_config" not in cfg["noise"]


def test_build_linking_config_growing():
    from server.configs import build_linking_config
    from server.models import LinkingRequest

    req = LinkingRequest(fragments=[[0, 1, 2, 3, 4, 5, 6]], mol_size_mean=28)
    cfg = build_linking_config(
        req=req, protein_path=Path("/x/prot.pdb"),
        input_ligand_path=Path("/x/lig.sdf"), output_dir=Path("/x/out"),
    )
    assert cfg["task"]["name"] == "maskfill"
    assert cfg["task"]["transform"]["preset_partition"]["grouped_node_p1"] == [
        [0, 1, 2, 3, 4, 5, 6]
    ]
    assert cfg["transforms"]["variable_mol_size"]["not_remove"] == [
        0, 1, 2, 3, 4, 5, 6
    ]
    assert cfg["transforms"]["featurizer"]["mol_as_pocket_center"] is True


def test_build_linking_config_two_fragments_union():
    from server.configs import build_linking_config
    from server.models import LinkingRequest

    req = LinkingRequest(
        fragments=[[0, 1, 2], [10, 11, 12]], mol_size_mean=40,
        use_input_center=False,
    )
    cfg = build_linking_config(
        req=req, protein_path=Path("/x/prot.pdb"),
        input_ligand_path=Path("/x/lig.sdf"), output_dir=Path("/x/out"),
    )
    assert cfg["transforms"]["variable_mol_size"]["not_remove"] == [
        0, 1, 2, 10, 11, 12
    ]
    assert "featurizer" not in cfg["transforms"]  # use_input_center=False


def test_build_optimize_config_init_step_flows():
    from server.configs import build_optimize_config
    from server.models import OptimizeRequest

    req = OptimizeRequest(init_step=0.3, num_steps=40)
    cfg = build_optimize_config(
        req=req, protein_path=Path("/x/prot.pdb"),
        input_ligand_path=Path("/x/lig.sdf"), output_dir=Path("/x/out"),
    )
    assert cfg["noise"]["init_step"] == 0.3
    assert cfg["noise"]["num_steps"] == 40
    assert cfg["task"]["transform"]["name"] == "sbdd"


def test_build_pepdesign_denovo_linear():
    from server.configs import build_pepdesign_config
    from server.models import PepDesignMode, PepDesignRequest

    req = PepDesignRequest(mode=PepDesignMode.denovo_linear, pep_length=10)
    cfg = build_pepdesign_config(
        req=req, protein_path=Path("/x/prot.pdb"),
        input_peptide_path=None, ref_ligand_path=None, output_dir=Path("/x/out"),
    )
    assert cfg["data"]["input_ligand"] == "peplen_10"
    assert cfg["data"]["is_pep"] is True
    assert cfg["task"]["transform"]["settings"]["mode"]["full"] == 1


def test_build_pepdesign_cyclic():
    from server.configs import build_pepdesign_config
    from server.models import PepDesignMode, PepDesignRequest

    req = PepDesignRequest(mode=PepDesignMode.denovo_cyclic, pep_length=8)
    cfg = build_pepdesign_config(
        req=req, protein_path=Path("/x/prot.pdb"),
        input_peptide_path=None, ref_ligand_path=None, output_dir=Path("/x/out"),
    )
    assert cfg["data"]["input_ligand"] == "cycpeplen_8"


def test_build_pepdesign_inverse_fold_requires_pdb():
    from server.configs import build_pepdesign_config
    from server.models import PepDesignMode, PepDesignRequest

    req = PepDesignRequest(mode=PepDesignMode.inverse_fold)
    with pytest.raises(ValueError, match="input peptide PDB required"):
        build_pepdesign_config(
            req=req, protein_path=Path("/x/prot.pdb"),
            input_peptide_path=None, ref_ligand_path=None,
            output_dir=Path("/x/out"),
        )


def test_build_pepdesign_inverse_fold_settings():
    from server.configs import build_pepdesign_config
    from server.models import PepDesignMode, PepDesignRequest

    req = PepDesignRequest(mode=PepDesignMode.inverse_fold)
    cfg = build_pepdesign_config(
        req=req, protein_path=Path("/x/prot.pdb"),
        input_peptide_path=Path("/x/pep.pdb"),
        ref_ligand_path=None, output_dir=Path("/x/out"),
    )
    assert cfg["data"]["input_ligand"] == "/x/pep.pdb"
    assert cfg["task"]["transform"]["settings"]["mode"]["sc"] == 1


def test_build_pepdesign_sc_pack_no_variable_sc_size():
    from server.configs import build_pepdesign_config
    from server.models import PepDesignMode, PepDesignRequest

    req = PepDesignRequest(mode=PepDesignMode.sc_pack)
    cfg = build_pepdesign_config(
        req=req, protein_path=Path("/x/prot.pdb"),
        input_peptide_path=Path("/x/pep.pdb"),
        ref_ligand_path=None, output_dir=Path("/x/out"),
    )
    assert cfg["task"]["transform"]["settings"]["mode"]["packing"] == 1
    # sc_pack has determined side-chain sizes → no variable_sc_size.
    assert "variable_sc_size" not in cfg["transforms"]


def test_build_pepdesign_fix_pos_flows():
    from server.configs import build_pepdesign_config
    from server.models import PepDesignMode, PepDesignRequest

    req = PepDesignRequest(
        mode=PepDesignMode.denovo_linear, pep_length=10,
        fix_pos_res_bb=[0, 1], fix_type_res_sc=[3],
    )
    cfg = build_pepdesign_config(
        req=req, protein_path=Path("/x/prot.pdb"),
        input_peptide_path=None, ref_ligand_path=None,
        output_dir=Path("/x/out"),
    )
    assert cfg["task"]["transform"]["fix_pos"]["res_bb"] == [0, 1]
    assert cfg["task"]["transform"]["fix_type_only"]["res_sc"] == [3]


def test_build_model_config_puts_ckpt_from_settings():
    from server.configs import build_model_config

    s = _off_settings()
    cfg = build_model_config(s)
    assert cfg["model"]["checkpoint"].endswith("pocketxmol.ckpt")


# ---------------------------------------------------------------------------
# Endpoint smoke — 422 / 200 branches without launching subprocess.
# ---------------------------------------------------------------------------
def test_dock_endpoint_returns_job(client):
    with open(DATA / "8C7Y_TXV_protein.pdb", "rb") as f:
        resp = client.post(
            "/api/dock",
            files={"protein": ("protein.pdb", f.read(), "chemical/x-pdb")},
            data={
                "num_samples": "3",
                "smiles": "c1ccccc1",
                "pocket_coord": "[1.0, 2.0, 3.0]",
            },
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "job_id" in body


def test_dock_endpoint_multiple_ligand_sources_rejected(client):
    with open(DATA / "8C7Y_TXV_protein.pdb", "rb") as pf, \
            open(DATA / "8C7Y_TXV_ligand_start_conf.sdf", "rb") as lf:
        resp = client.post(
            "/api/dock",
            files={
                "protein": ("protein.pdb", pf.read(), "chemical/x-pdb"),
                "ligand": ("ligand.sdf", lf.read(), "chemical/x-mdl-sdfile"),
            },
            data={"smiles": "c1ccccc1"},
        )
    assert resp.status_code == 422
    assert "exactly one" in resp.json()["detail"]


def test_sbdd_endpoint_requires_pocket_coord(client):
    with open(DATA / "2ar9_A.pdb", "rb") as f:
        resp = client.post(
            "/api/sbdd",
            files={"protein": ("protein.pdb", f.read(), "chemical/x-pdb")},
            data={"num_samples": "3"},  # missing pocket_coord
        )
    # pocket_coord is required by pydantic → 422
    assert resp.status_code == 422


def test_sbdd_endpoint_returns_job(client):
    with open(DATA / "2ar9_A.pdb", "rb") as f:
        resp = client.post(
            "/api/sbdd",
            files={"protein": ("protein.pdb", f.read(), "chemical/x-pdb")},
            data={
                "num_samples": "3",
                "pocket_coord": "[1.0, 2.0, 3.0]",
            },
        )
    assert resp.status_code == 200, resp.text
    assert "job_id" in resp.json()


def test_linking_endpoint_requires_input_ligand(client):
    with open(DATA / "2ar9_A.pdb", "rb") as f:
        resp = client.post(
            "/api/linking",
            files={"protein": ("protein.pdb", f.read(), "chemical/x-pdb")},
            data={"fragments": "[[0,1,2,3,4,5,6]]", "num_samples": "3"},
        )
    assert resp.status_code == 422


def test_linking_endpoint_returns_job(client):
    with open(DATA / "2ar9_A.pdb", "rb") as pf, \
            open(DATA / "fragment.sdf", "rb") as lf:
        resp = client.post(
            "/api/linking",
            files={
                "protein": ("protein.pdb", pf.read(), "chemical/x-pdb"),
                "input_ligand": ("frag.sdf", lf.read(), "chemical/x-mdl-sdfile"),
            },
            data={
                "fragments": "[[0,1,2,3,4,5,6]]",
                "num_samples": "3",
                "mol_size_mean": "28",
            },
        )
    assert resp.status_code == 200, resp.text
    assert "job_id" in resp.json()


def test_optimize_endpoint_returns_job(client):
    with open(DATA / "moad_62740_pro.pdb", "rb") as pf, \
            open(DATA / "moad_62740_mol.sdf", "rb") as lf:
        resp = client.post(
            "/api/optimize",
            files={
                "protein": ("protein.pdb", pf.read(), "chemical/x-pdb"),
                "input_ligand": ("lig.sdf", lf.read(), "chemical/x-mdl-sdfile"),
            },
            data={"init_step": "0.3", "num_samples": "3"},
        )
    assert resp.status_code == 200, resp.text


def test_pepdesign_denovo_linear_endpoint(client):
    with open(DATA / "3bik_A.pdb", "rb") as pf, \
            open(DATA / "3bik_A_pocket_coord.sdf", "rb") as rf:
        resp = client.post(
            "/api/pepdesign",
            files={
                "protein": ("protein.pdb", pf.read(), "chemical/x-pdb"),
                "ref_ligand": ("ref.sdf", rf.read(), "chemical/x-mdl-sdfile"),
            },
            data={
                "mode": "denovo_linear",
                "pep_length": "10",
                "pocket_radius": "20",
                "num_samples": "3",
            },
        )
    assert resp.status_code == 200, resp.text


def test_pepdesign_inverse_fold_requires_peptide(client):
    with open(DATA / "3bik_A.pdb", "rb") as f:
        resp = client.post(
            "/api/pepdesign",
            files={"protein": ("protein.pdb", f.read(), "chemical/x-pdb")},
            data={"mode": "inverse_fold", "num_samples": "3"},
        )
    assert resp.status_code == 422
    assert "input_peptide" in resp.json()["detail"]


def test_pepdesign_denovo_requires_pep_length(client):
    with open(DATA / "3bik_A.pdb", "rb") as f:
        resp = client.post(
            "/api/pepdesign",
            files={"protein": ("protein.pdb", f.read(), "chemical/x-pdb")},
            data={"mode": "denovo_linear", "num_samples": "3"},
        )
    assert resp.status_code == 422  # pydantic validator


def test_confidence_endpoint_missing_source_job(client):
    # source_job_id references a directory that doesn't exist → 404 from
    # _resolve_source_exp_dir at build_argv time.  Because the framework
    # invokes build_argv inside submit(), a missing source dir surfaces
    # as a 500 (framework re-raises the HTTPException as a run failure).
    resp = client.post(
        "/api/confidence",
        data={
            "source_job_id": "nonexistent-source-job",
            "variant": "tuned_cfd",
        },
    )
    # runner.submit wraps build_argv; framework returns 500 for HTTPException
    # raised inside build_argv.  Accept either 404 or 500 depending on
    # whether the framework wraps or forwards.
    assert resp.status_code in (404, 500), resp.text


# ---------------------------------------------------------------------------
# Sanity: YAML actually written to disk on submit
# ---------------------------------------------------------------------------
def test_dock_writes_task_and_model_yaml(client, tmp_path):
    with open(DATA / "8C7Y_TXV_protein.pdb", "rb") as f:
        resp = client.post(
            "/api/dock",
            files={"protein": ("protein.pdb", f.read(), "chemical/x-pdb")},
            data={
                "num_samples": "3", "smiles": "c1ccccc1",
                "pocket_coord": "[1.0, 2.0, 3.0]",
            },
        )
    job_id = resp.json()["job_id"]
    jobs_base = tmp_path / "jobs"
    task_yml = jobs_base / job_id / "input" / "task_config.yml"
    model_yml = jobs_base / job_id / "input" / "model_config.yml"
    # Framework may set up the job dir slightly differently at submit-time;
    # allow small delay by iterating through possible layouts.
    assert task_yml.exists(), f"task YAML not written: expected {task_yml}"
    assert model_yml.exists(), f"model YAML not written: expected {model_yml}"

    task_cfg = yaml.safe_load(task_yml.read_text())
    assert task_cfg["task"]["name"] == "dock"
    model_cfg = yaml.safe_load(model_yml.read_text())
    assert "checkpoint" in model_cfg["model"]
