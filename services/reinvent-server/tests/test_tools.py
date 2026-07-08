import json
from pathlib import Path


class _Settings:
    python = Path("/venv/bin/python")
    reinvent_bin = Path("/venv/bin/reinvent")
    prior_base = Path("/nas/priors")
    device = "cuda:0"


def _argv_map(argv):
    """Flatten [--flag, val, --flag2, val2, ...] tail into a dict (skip prog head)."""
    out = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            out[argv[i]] = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else True
            i += 2
        else:
            i += 1
    return out


def test_sampling_argv(tmp_path):
    from server.models import SamplingRequest
    from server.tools import sampling_argv
    req = SamplingRequest(num_smiles=7, device=None)
    argv = sampling_argv(req, {"smiles_file": tmp_path / "s.smi"}, tmp_path, _Settings())
    assert argv[:3] == ["/venv/bin/python", "-m", "server.reinvent_cli"]
    m = _argv_map(argv)
    assert m["--run-type"] == "sampling"
    assert m["--device"] == "cuda:0"           # None → settings.device
    assert m["--prior-base"] == "/nas/priors"
    assert m["--reinvent-bin"] == "/venv/bin/reinvent"
    assert m["--smiles-file"] == str(tmp_path / "s.smi")
    # params.json written under work/ with the request payload
    pj = Path(m["--params-json"])
    assert pj.exists()
    assert json.loads(pj.read_text())["num_smiles"] == 7


def test_scoring_argv_no_files(tmp_path):
    from server.models import ScoringRequest
    from server.tools import scoring_argv
    req = ScoringRequest(scoring={"type": "geometric_mean", "component": []}, device="cpu")
    argv = scoring_argv(req, {"smiles_file": tmp_path / "c.smi"}, tmp_path, _Settings())
    m = _argv_map(argv)
    assert m["--run-type"] == "scoring"
    assert m["--device"] == "cpu"


def test_staged_learning_argv(tmp_path):
    from server.models import StagedLearningRequest, StageSpec
    from server.tools import staged_learning_argv
    req = StagedLearningRequest(
        stages=[StageSpec(chkpt_name="s1.chkpt", scoring={"type": "geometric_mean", "component": []})],
    )
    argv = staged_learning_argv(req, {"agent_file": tmp_path / "a.chkpt"}, tmp_path, _Settings())
    m = _argv_map(argv)
    assert m["--run-type"] == "staged_learning"
    assert m["--agent-file"] == str(tmp_path / "a.chkpt")
    pj = Path(m["--params-json"])
    assert json.loads(pj.read_text())["stages"][0]["chkpt_name"] == "s1.chkpt"
