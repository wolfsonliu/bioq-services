def test_cli_endpoints_registered():
    from server import __main__ as m
    assert set(m.endpoints) == {
        "sampling", "scoring", "enumeration",
        "transfer-learning", "staged-learning",
    }
    assert m.endpoints["scoring"].request_model.__name__ == "ScoringRequest"


def test_cli_sampling_build_argv(tmp_path):
    from server import __main__ as m
    from server.models import SamplingRequest
    req = SamplingRequest(num_smiles=3)
    ep = m.endpoints["sampling"]
    argv = ep.build_argv(req, {}, tmp_path, m.settings)
    assert "server.reinvent_cli" in argv
    assert "--run-type" in argv and "sampling" in argv
