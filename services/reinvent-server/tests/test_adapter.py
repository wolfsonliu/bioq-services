import json
from pathlib import Path


def _adapter(tmp_path):
    from server.adapter import ReinventAdapter
    from server.settings import ReinventSettings

    class _S(ReinventSettings):
        pass
    s = _S(jobs_base_dir=tmp_path / "jobs", prior_base=tmp_path / "priors")
    return ReinventAdapter(settings=s)


def _job(tmp_path, label: str) -> Path:
    jd = tmp_path / "jobs" / "j1"
    (jd / "output").mkdir(parents=True)
    (jd / "manifest.json").write_text(json.dumps({"label": label}))
    return jd


def test_detect_sampling(tmp_path):
    a = _adapter(tmp_path)
    jd = _job(tmp_path, "sampling")
    assert a.detect_outputs(jd) is False
    (jd / "output" / "sampling.csv").write_text("SMILES,NLL\nCCO,-1.2\n")
    assert a.detect_outputs(jd) is True


def test_detect_staged_learning(tmp_path):
    a = _adapter(tmp_path)
    jd = _job(tmp_path, "staged_learning")
    assert a.detect_outputs(jd) is False
    (jd / "output" / "staged_learning_1.csv").write_text("step,score\n")
    assert a.detect_outputs(jd) is False  # need a chkpt too
    (jd / "output" / "s1.chkpt").write_text("x")
    assert a.detect_outputs(jd) is True


def test_detect_transfer_learning(tmp_path):
    a = _adapter(tmp_path)
    jd = _job(tmp_path, "transfer_learning")
    assert a.detect_outputs(jd) is False
    (jd / "output" / "TL_model.model").write_text("weights")
    assert a.detect_outputs(jd) is True


def test_manifest_extras(tmp_path):
    a = _adapter(tmp_path)
    ex = a.manifest_extras()
    assert "sampling" in ex["run_modes"]
    assert ex["prior_registry"][".reinvent"] == "reinvent.prior"
    assert "chemprop2" in ex["scoring_backends"]
