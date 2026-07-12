import json
import sys
import types
from pathlib import Path


def test_resolve_device_cpu_passthrough():
    from server import reinvent_cli
    assert reinvent_cli._resolve_device("cpu") == "cpu"


def test_resolve_device_cuda_falls_back_when_unavailable(monkeypatch):
    from server import reinvent_cli
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert reinvent_cli._resolve_device("cuda:0") == "cpu"


def test_resolve_device_cuda_kept_when_available(monkeypatch):
    from server import reinvent_cli
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert reinvent_cli._resolve_device("cuda:0") == "cuda:0"


def _write_params(work: Path, params: dict) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    pj = work / "params.json"
    pj.write_text(json.dumps(params))
    return pj


def test_stage_and_build_writes_config(tmp_path, monkeypatch):
    from server import reinvent_cli

    work = tmp_path / "work"
    out = tmp_path / "output"
    out.mkdir(parents=True)
    seed = tmp_path / "seed.smi"
    seed.write_text("CCO\n")
    params = {"generator": "mol2mol", "model_file": ".mol2mol", "num_smiles": 5,
              "unique_molecules": True, "randomize_smiles": True, "temperature": 1.0,
              "sample_strategy": "beamsearch"}
    pj = _write_params(work, params)

    calls = {}

    def fake_run(argv, **kwargs):
        calls["argv"] = argv
        calls["cwd"] = kwargs.get("cwd")
        calls["env"] = kwargs.get("env")

        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(reinvent_cli.subprocess, "run", fake_run)

    rc = reinvent_cli.run(
        run_type="sampling", params_json=pj, work_dir=work, output_dir=out,
        device="cpu", prior_base=Path("/nas/priors"),
        reinvent_bin=Path("/opt/x/reinvent"), files={"smiles_file": seed},
    )
    assert rc == 0
    cfg = (work / "config.toml").read_text()
    assert 'run_type = "sampling"' in cfg
    assert "/nas/priors/pubchem_ecfp4_with_count_with_rank_reinvent4_dict_voc.prior" in cfg
    # seed staged into work/ and referenced
    assert (work / "seed.smi").exists()
    assert str(work / "seed.smi") in cfg
    # reinvent invoked with config path, cwd=work, prior base in env
    assert calls["argv"][0] == "/opt/x/reinvent"
    assert str(work / "config.toml") in calls["argv"]
    assert calls["cwd"] == work
    assert calls["env"]["REINVENT_PRIOR_BASE"] == "/nas/priors"
    # audit copy of config on success
    assert (out / "config.toml").exists()


def test_nonzero_rc_collects_diagnostics_not_json_echo(tmp_path, monkeypatch):
    from server import reinvent_cli
    work = tmp_path / "work"
    out = tmp_path / "output"
    out.mkdir(parents=True)
    pj = _write_params(work, {"scoring": {"type": "geometric_mean", "component": []},
                              "smiles_column": "SMILES", "standardize_smiles": True,
                              "parallel": 1})
    smi = tmp_path / "c.smi"
    smi.write_text("CCO\n")

    def fake_run(argv, **kwargs):
        # Simulate reinvent writing its -l log then failing.
        (work / "reinvent.log").write_text("Traceback: boom\n")
        (work / "_scoring.json").write_text("{}")

        class R:
            returncode = 7
        return R()
    monkeypatch.setattr(reinvent_cli.subprocess, "run", fake_run)

    rc = reinvent_cli.run(
        run_type="scoring", params_json=pj, work_dir=work, output_dir=out,
        device="cpu", prior_base=tmp_path, reinvent_bin=Path("/opt/x/reinvent"),
        files={"smiles_file": smi},
    )
    assert rc == 7
    # On failure we STILL copy config.toml + reinvent.log for diagnosis,
    # but not the (meaningless) JSON config echo.
    assert (out / "config.toml").exists()
    assert (out / "reinvent.log").read_text() == "Traceback: boom\n"
    assert not (out / "_scoring.json").exists()
