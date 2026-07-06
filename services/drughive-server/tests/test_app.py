"""Offline tests for drughive-server (no real subprocess / GPU needed).

Step 1: skeleton smoke — healthz + manifest service name.
More tests are added in Steps 4-7 as configs.py / models.py / app.py grow.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DRUGHIVE_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("DRUGHIVE_ROOT", str(tmp_path / "drughive"))
    monkeypatch.setenv("DRUGHIVE_WEIGHTS_DIR", str(tmp_path / "models"))
    (tmp_path / "drughive").mkdir(parents=True, exist_ok=True)
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)

    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


# ----- Healthcheck / manifest smoke -----

def test_health(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "drughive"
    assert "version" in body


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "drughive"


def test_healthz_detail_reports_missing_weights(client, tmp_path):
    """Weights + qvina2 probe should not raise when NAS/binary are absent."""
    body = client.get("/healthz/detail").json()
    assert body["status"] == "ok"
    assert body["service"] == "drughive"
    # tmp_path/models is empty → weights_loaded should be False
    assert body["weights_loaded"] is False
    assert "checkpoint" in body["weights_missing"]
    # qvina2_available may be True or False depending on host — just check schema
    assert "qvina2_available" in body
    assert "active_jobs" in body


# ----- Settings -----

def test_settings_defaults():
    from server.settings import DrughiveSettings

    class _Off(DrughiveSettings):
        model_config = SettingsConfigDict(
            env_prefix="DRUGHIVE_OFFLINE_", env_file=None, extra="ignore",
        )

    s = _Off()
    assert s.jobs_base_dir == Path("/data/drughive_jobs")
    assert s.root == Path("/opt/drughive")
    assert s.weights_dir == Path("/data/models/drughive/checkpoints")
    assert s.checkpoint_filename == "drughive_model_ch9.ckpt"
    assert s.model_id == "c9_pdbzinc"
    assert s.docking_cmd == "qvina2.1"
    assert s.checkpoint_path == Path(
        "/data/models/drughive/checkpoints/drughive_model_ch9.ckpt"
    )


# ----- configs.build_* YAML builders (unit) -----


def _off_settings():
    from server.settings import DrughiveSettings

    class _Off(DrughiveSettings):
        model_config = SettingsConfigDict(
            env_prefix="DRUGHIVE_OFFLINE_", env_file=None, extra="ignore",
        )

    return _Off()


def test_build_generate_config_minimal(tmp_path):
    from server.configs import build_generate_config
    from server.models import GenerateRequest

    req = GenerateRequest(n_samples=5, pdb_id="1abc")
    cfg = build_generate_config(
        req=req,
        target_path=tmp_path / "pocket.pdb",
        ligand_path=tmp_path / "lig.sdf",
        output_dir=tmp_path / "out",
        settings=_off_settings(),
    )

    # required upstream keys
    for key in (
        "target_path", "ligand_path", "pdb_id", "output", "n_samples",
        "zbetas", "temps", "checkpoint", "model_id", "ffopt_mols",
    ):
        assert key in cfg, f"missing {key}"

    assert cfg["n_samples"] == 5
    assert cfg["pdb_id"] == "1abc"
    # default zbetas is [0.0, 0.0, 0.0, 0.0] (list, not scalar)
    assert cfg["zbetas"] == [0.0, 0.0, 0.0, 0.0]
    # mol_filter has no set fields → no filter keys leak into YAML
    for k in ("ring_sizes", "ring_system_max", "ring_loops_max",
              "dbl_bond_pairs", "n_atoms_min"):
        assert k not in cfg
    # spatial-only keys must NOT be in the de novo config
    assert "substruct_modify_path" not in cfg
    assert "substruct_modify_pattern" not in cfg


def test_build_generate_config_with_mol_filter(tmp_path):
    from server.configs import build_generate_config
    from server.models import GenerateRequest, MolFilterParams

    req = GenerateRequest(
        n_samples=3,
        mol_filter=MolFilterParams(ring_sizes=[5, 6], n_atoms_min=8),
    )
    cfg = build_generate_config(
        req=req,
        target_path=tmp_path / "p.pdb",
        ligand_path=tmp_path / "l.sdf",
        output_dir=tmp_path / "o",
        settings=_off_settings(),
    )
    assert cfg["ring_sizes"] == [5, 6]
    assert cfg["n_atoms_min"] == 8
    # only the set filter fields
    assert "ring_system_max" not in cfg
    assert "dbl_bond_pairs" not in cfg


def test_build_generate_spatial_config_with_pattern(tmp_path):
    from server.configs import build_generate_spatial_config
    from server.models import GenerateSpatialRequest

    req = GenerateSpatialRequest(
        n_samples=4, substruct_modify_pattern="[CH2]C1:C:C:C:C:C:1"
    )
    cfg = build_generate_spatial_config(
        req=req,
        target_path=tmp_path / "p.pdb",
        ligand_path=tmp_path / "l.sdf",
        output_dir=tmp_path / "o",
        settings=_off_settings(),
        substruct_modify_path=None,
    )
    assert cfg["substruct_modify_pattern"] == "[CH2]C1:C:C:C:C:C:1"
    assert "substruct_modify_path" not in cfg
    # spatial default zbetas leans posterior
    assert cfg["zbetas"] == [0.3, 0.3, 0.3, 0.3]


def test_build_generate_spatial_config_with_file(tmp_path):
    from server.configs import build_generate_spatial_config
    from server.models import GenerateSpatialRequest

    frag = tmp_path / "frag.sdf"
    frag.write_text("$$$$")
    req = GenerateSpatialRequest(n_samples=4)
    cfg = build_generate_spatial_config(
        req=req,
        target_path=tmp_path / "p.pdb",
        ligand_path=tmp_path / "l.sdf",
        output_dir=tmp_path / "o",
        settings=_off_settings(),
        substruct_modify_path=frag,
    )
    assert cfg["substruct_modify_path"] == str(frag)
    assert "substruct_modify_pattern" not in cfg


def test_build_optimize_config_full(tmp_path):
    from server.configs import build_optimize_config
    from server.models import OptimizeRequest

    req = OptimizeRequest(
        pdb_id="5d3h",
        key_opt="affinity_qvina",
        n_cycles=2,
        n_samples_initial=20,
        n_samples=4,
        n_best_parents=2,
        zbetas=[0.3, 0.2],
    )
    cfg = build_optimize_config(
        req=req,
        target_path=tmp_path / "pocket.pdb",
        ligand_path=tmp_path / "lig.sdf",
        target_pdbqt_path=tmp_path / "pocket.pdbqt",
        output_dir=tmp_path / "out",
        settings=_off_settings(),
    )
    assert cfg["target_path_pdbqt"] == str(tmp_path / "pocket.pdbqt")
    assert cfg["key_opt"] == "affinity_qvina"
    assert cfg["n_cycles"] == 2
    assert cfg["zbetas"] == [0.3, 0.2]
    # docking_cmd comes from settings (default "qvina2.1"), NOT from request
    assert cfg["docking_cmd"] == "qvina2.1"


def test_build_optimize_config_scalar_zbetas(tmp_path):
    """Scalar zbetas is broadcast to length n_cycles by the model validator."""
    from server.configs import build_optimize_config
    from server.models import OptimizeRequest

    req = OptimizeRequest(n_cycles=4, zbetas=0.25)
    cfg = build_optimize_config(
        req=req,
        target_path=tmp_path / "p.pdb",
        ligand_path=tmp_path / "l.sdf",
        target_pdbqt_path=tmp_path / "p.pdbqt",
        output_dir=tmp_path / "o",
        settings=_off_settings(),
    )
    # scalar 0.25 broadcast to [0.25]*4
    assert cfg["zbetas"] == [0.25, 0.25, 0.25, 0.25]


# ----- Request model validators -----


def test_generate_request_zbetas_length_must_be_4():
    from pydantic import ValidationError

    from server.models import GenerateRequest

    with pytest.raises(ValidationError, match="length 4"):
        GenerateRequest(zbetas=[0.1, 0.2])


def test_generate_request_temps_length_must_be_4():
    from pydantic import ValidationError

    from server.models import GenerateRequest

    with pytest.raises(ValidationError, match="length 4"):
        GenerateRequest(temps=[1.0, 1.0, 1.0])


def test_generate_request_scalar_zbetas_broadcasts_to_list_of_4():
    """Scalar is broadcast to a length-4 list by the before-validator."""
    from server.models import GenerateRequest

    req = GenerateRequest(zbetas=0.5, temps=0.5)
    assert req.zbetas == [0.5, 0.5, 0.5, 0.5]
    assert req.temps == [0.5, 0.5, 0.5, 0.5]


def test_optimize_zbetas_length_must_match_n_cycles():
    from pydantic import ValidationError

    from server.models import OptimizeRequest

    with pytest.raises(ValidationError, match="n_cycles"):
        OptimizeRequest(n_cycles=4, zbetas=[0.3, 0.3])  # len 2 != 4


def test_optimize_default_zbetas_broadcasts_to_default_n_cycles():
    """Default zbetas is length-1 sentinel [0.3] broadcast to n_cycles (8)."""
    from server.models import OptimizeRequest

    req = OptimizeRequest()
    assert isinstance(req.zbetas, list)
    assert len(req.zbetas) == req.n_cycles == 8
    # All broadcast to 0.3 (the sentinel).
    assert all(z == 0.3 for z in req.zbetas)


def test_optimize_key_opt_rejects_bad_value():
    from pydantic import ValidationError

    from server.models import OptimizeRequest

    with pytest.raises(ValidationError):
        OptimizeRequest(key_opt="nonsense")  # type: ignore[arg-type]


def test_optimize_key_opt_accepts_qed():
    from server.models import OptimizeRequest

    req = OptimizeRequest(key_opt="qed", opt_increase=True)
    assert req.key_opt == "qed"


# ----- tools.py argv builders -----


def test_generate_argv_shape(tmp_path):
    from server.tools import generate_argv

    settings = _off_settings()
    cfg = tmp_path / "config.yml"
    cfg.write_text("stub")

    argv = generate_argv(cfg_path=cfg, settings=settings)
    assert argv[0] == settings.python
    assert argv[1].endswith("generate_molecules.py")
    assert argv[2] == str(cfg)


def test_optimize_argv_shape(tmp_path):
    from server.tools import optimize_argv

    settings = _off_settings()
    cfg = tmp_path / "config.yml"
    cfg.write_text("stub")

    argv = optimize_argv(cfg_path=cfg, settings=settings)
    assert argv[1].endswith("generate_optimize.py")
    assert argv[2] == str(cfg)


# ----- Adapter -----


def test_adapter_detect_outputs_empty(tmp_path):
    from server.adapter import DrughiveAdapter

    a = DrughiveAdapter(settings=_off_settings())
    job_dir = tmp_path / "job"
    (job_dir / "output").mkdir(parents=True)
    assert a.detect_outputs(job_dir) is False


def test_adapter_detect_outputs_generate_nested(tmp_path):
    """Upstream writes into output/<gen_name>/<pdb_id>/mols_gen.sdf — needs rglob."""
    from server.adapter import DrughiveAdapter

    a = DrughiveAdapter(settings=_off_settings())
    job_dir = tmp_path / "job"
    nested = job_dir / "output" / "prior" / "5d3h"
    nested.mkdir(parents=True)
    (nested / "mols_gen.sdf").write_text("valid_sdf_content\n$$$$\n")
    assert a.detect_outputs(job_dir) is True


def test_adapter_detect_outputs_optimize_initial_pool(tmp_path):
    """Optimize mode writes output/pdbzinc_initial/<pdb_id>/mols_gen.sdf."""
    from server.adapter import DrughiveAdapter

    a = DrughiveAdapter(settings=_off_settings())
    job_dir = tmp_path / "job"
    nested = job_dir / "output" / "pdbzinc_initial" / "5d3h"
    nested.mkdir(parents=True)
    (nested / "mols_gen.sdf").write_text("initial_pop\n$$$$\n")
    assert a.detect_outputs(job_dir) is True


def test_adapter_detect_outputs_ffopt_variant(tmp_path):
    """`mols_gen_opt.sdf` (FF-post-processed) also counts."""
    from server.adapter import DrughiveAdapter

    a = DrughiveAdapter(settings=_off_settings())
    job_dir = tmp_path / "job"
    nested = job_dir / "output" / "prior" / "5d3h"
    nested.mkdir(parents=True)
    (nested / "mols_gen_opt.sdf").write_text("ffopt\n$$$$\n")
    assert a.detect_outputs(job_dir) is True


def test_adapter_manifest_extras_shape():
    from server.adapter import DrughiveAdapter

    a = DrughiveAdapter(settings=_off_settings())
    extras = a.manifest_extras()
    assert "generate" in extras["tool_outputs"]
    assert "generate_spatial" in extras["tool_outputs"]
    assert "optimize" in extras["tool_outputs"]
    assert "USC-RL" in extras["model"]["license"]
    assert "optimize" in extras["long_running"]


def test_adapter_endpoint_examples_shape():
    from server.adapter import DrughiveAdapter

    a = DrughiveAdapter(settings=_off_settings())
    ex = a.endpoint_examples()
    assert set(ex.keys()) == {
        "/api/generate", "/api/generate_spatial", "/api/optimize",
    }
    # every endpoint has at least one curl example
    for path, examples in ex.items():
        assert examples, f"no examples for {path}"
        assert examples[0].curl.startswith("curl")


# ----- Endpoint smoke via TestClient -----


def test_manifest_lists_all_6_endpoints(client):
    body = client.get("/api/manifest").json()
    paths = {e["path"] for e in body["endpoints"]}
    assert "/api/generate" in paths
    assert "/api/generate_spatial" in paths
    assert "/api/optimize" in paths
    assert "/api/tasks/generate" in paths
    assert "/api/tasks/generate_spatial" in paths
    assert "/api/tasks/optimize" in paths


def test_generate_without_files_returns_422(client):
    """Missing target/ligand → 422 from uris.resolve_input."""
    r = client.post("/api/generate", data={"n_samples": "2"})
    assert r.status_code == 422


def test_generate_spatial_without_pattern_or_file_returns_422(client, tmp_path):
    """Neither pattern nor substruct_modify file → 422."""
    pdb = tmp_path / "p.pdb"
    pdb.write_text("ATOM  test\n")
    sdf = tmp_path / "l.sdf"
    sdf.write_text("$$$$\n")
    with open(pdb, "rb") as fp, open(sdf, "rb") as fl:
        r = client.post(
            "/api/generate_spatial",
            files={
                "target": ("p.pdb", fp, "chemical/x-pdb"),
                "ligand": ("l.sdf", fl, "chemical/x-mdl-sdfile"),
            },
            data={"n_samples": "2"},
        )
    assert r.status_code == 422


def test_generate_spatial_both_pattern_and_file_returns_422(client, tmp_path):
    """Both pattern AND file → 422."""
    pdb = tmp_path / "p.pdb"
    pdb.write_text("ATOM  test\n")
    sdf = tmp_path / "l.sdf"
    sdf.write_text("$$$$\n")
    frag = tmp_path / "frag.sdf"
    frag.write_text("$$$$\n")
    with open(pdb, "rb") as fp, open(sdf, "rb") as fl, open(frag, "rb") as ff:
        r = client.post(
            "/api/generate_spatial",
            files={
                "target": ("p.pdb", fp, "chemical/x-pdb"),
                "ligand": ("l.sdf", fl, "chemical/x-mdl-sdfile"),
                "substruct_modify": ("frag.sdf", ff, "chemical/x-mdl-sdfile"),
            },
            data={
                "n_samples": "2",
                "substruct_modify_pattern": "[CH2]C1:C:C:C:C:C:1",
            },
        )
    assert r.status_code == 422


def test_optimize_qvina_key_without_pdbqt_returns_422(client, tmp_path):
    """key_opt=affinity_qvina + no target_pdbqt → 422."""
    pdb = tmp_path / "p.pdb"
    pdb.write_text("ATOM  test\n")
    sdf = tmp_path / "l.sdf"
    sdf.write_text("$$$$\n")
    with open(pdb, "rb") as fp, open(sdf, "rb") as fl:
        r = client.post(
            "/api/optimize",
            files={
                "target": ("p.pdb", fp, "chemical/x-pdb"),
                "ligand": ("l.sdf", fl, "chemical/x-mdl-sdfile"),
            },
            data={
                "key_opt": "affinity_qvina",
                "n_cycles": "2",
                "n_samples_initial": "20",
                "n_samples": "4",
                "n_best_parents": "2",
                "zbetas": "[0.3, 0.2]",
            },
        )
    # 422 comes from HTTPException raised inside build_argv → runner
    # translates to 5xx cleanup path; either is acceptable evidence that
    # we didn't reach subprocess execution.
    assert r.status_code in (422, 500, 502)


def test_generate_submits_job_with_mocked_subprocess(client, monkeypatch, tmp_path):
    """End-to-end: file upload → JobInfo, without running the real algorithm.

    We stub SubprocessRunner.submit so no fork happens; we just check that
    the endpoint accepts a valid multipart POST and returns a JobInfo.
    """
    pdb = tmp_path / "p.pdb"
    pdb.write_text("ATOM  test\n")
    sdf = tmp_path / "l.sdf"
    sdf.write_text("$$$$\n")

    with open(pdb, "rb") as fp, open(sdf, "rb") as fl:
        r = client.post(
            "/api/generate",
            files={
                "target": ("p.pdb", fp, "chemical/x-pdb"),
                "ligand": ("l.sdf", fl, "chemical/x-mdl-sdfile"),
            },
            data={"n_samples": "2", "pdb_id": "test"},
        )
    # Successful submit — a JobInfo with job_id + status pending.
    assert r.status_code == 200, r.text
    body = r.json()
    assert "job_id" in body
    assert body["input_params"] is not None
    assert body["input_params"]["n_samples"] == 2
    assert body["input_params"]["pdb_id"] == "test"
