from pathlib import Path


def _adapter(tmp_path):
    from server.adapter import ReinventAdapter
    from server.settings import ReinventSettings

    class _S(ReinventSettings):
        pass
    s = _S(jobs_base_dir=tmp_path / "jobs", prior_base=tmp_path / "priors")
    return ReinventAdapter(settings=s)


def _job(tmp_path) -> Path:
    jd = tmp_path / "jobs" / "j1"
    (jd / "output").mkdir(parents=True)
    return jd


def test_no_outputs_when_empty(tmp_path):
    a = _adapter(tmp_path)
    jd = _job(tmp_path)
    assert a.detect_outputs(jd) is False


def test_audit_files_alone_are_not_a_result(tmp_path):
    # reinvent_cli copies these even on FAILURE — must NOT count as success.
    a = _adapter(tmp_path)
    jd = _job(tmp_path)
    (jd / "output" / "config.toml").write_text("run_type = 'sampling'\n")
    (jd / "output" / "reinvent.log").write_text("Traceback: boom\n")
    (jd / "output" / "_sampling.json").write_text("{}")
    assert a.detect_outputs(jd) is False


def test_detect_sampling(tmp_path):
    a = _adapter(tmp_path)
    jd = _job(tmp_path)
    (jd / "output" / "config.toml").write_text("x\n")  # audit noise
    assert a.detect_outputs(jd) is False
    (jd / "output" / "sampling.csv").write_text("SMILES,NLL\nCCO,-1.2\n")
    assert a.detect_outputs(jd) is True


def test_empty_result_file_is_not_success(tmp_path):
    a = _adapter(tmp_path)
    jd = _job(tmp_path)
    (jd / "output" / "sampling.csv").write_text("")  # 0 bytes
    assert a.detect_outputs(jd) is False


def test_detect_staged_learning(tmp_path):
    a = _adapter(tmp_path)
    jd = _job(tmp_path)
    (jd / "output" / "staged_learning_1.csv").write_text("step,score\n")
    assert a.detect_outputs(jd) is True


def test_detect_transfer_learning(tmp_path):
    a = _adapter(tmp_path)
    jd = _job(tmp_path)
    (jd / "output" / "TL_model.model").write_text("weights")
    assert a.detect_outputs(jd) is True


def test_manifest_extras(tmp_path):
    a = _adapter(tmp_path)
    ex = a.manifest_extras()
    assert "sampling" in ex["run_modes"]
    assert ex["prior_registry"][".reinvent"] == "reinvent_pubchem.prior"
    assert "chemprop2" in ex["scoring_backends"]
