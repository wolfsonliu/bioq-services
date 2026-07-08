from pathlib import Path


def test_resolve_prior_dot_key():
    from server.config_builder import _resolve_prior
    got = _resolve_prior(".reinvent", "reinvent", Path("/nas/priors"))
    assert got == "/nas/priors/reinvent.prior"


def test_resolve_prior_default_for_generator():
    from server.config_builder import _resolve_prior
    got = _resolve_prior(None, "mol2mol", Path("/nas/priors"))
    assert got == "/nas/priors/mol2mol_medium_similarity.prior"


def test_resolve_prior_explicit_path_passthrough():
    from server.config_builder import _resolve_prior
    got = _resolve_prior("/abs/custom.prior", "reinvent", Path("/nas/priors"))
    assert got == "/abs/custom.prior"


def test_build_sampling_config(tmp_path):
    from server.config_builder import build_sampling_config
    p = {"generator": "reinvent", "model_file": None, "num_smiles": 25,
         "unique_molecules": True, "randomize_smiles": True,
         "temperature": 1.0, "sample_strategy": "multinomial", "device": "cpu"}
    out = tmp_path / "output"
    cfg = build_sampling_config(p, out, Path("/nas/priors"))
    assert cfg["run_type"] == "sampling"
    assert cfg["device"] == "cpu"
    assert cfg["parameters"]["model_file"] == "/nas/priors/reinvent.prior"
    assert cfg["parameters"]["num_smiles"] == 25
    assert cfg["parameters"]["output_file"] == str(out / "sampling.csv")
    assert "smiles_file" not in cfg["parameters"]


def test_build_sampling_config_with_seed(tmp_path):
    from server.config_builder import build_sampling_config
    p = {"generator": "mol2mol", "model_file": ".m2m_high", "num_smiles": 10,
         "unique_molecules": True, "randomize_smiles": False, "temperature": 1.2,
         "sample_strategy": "beamsearch", "device": "cuda:0",
         "smiles_file": "/work/seed.smi"}
    cfg = build_sampling_config(p, tmp_path, Path("/nas/priors"))
    assert cfg["parameters"]["smiles_file"] == "/work/seed.smi"
    assert cfg["parameters"]["sample_strategy"] == "beamsearch"
    assert cfg["parameters"]["model_file"] == "/nas/priors/mol2mol_high_similarity.prior"


def test_build_scoring_config_passthrough(tmp_path):
    from server.config_builder import build_scoring_config
    scoring = {
        "type": "geometric_mean",
        "component": [
            {"QED": {"endpoint": [{"name": "QED", "weight": 0.5}]}},
        ],
    }
    p = {"smiles_column": "SMILES", "standardize_smiles": True,
         "parallel": 4, "scoring": scoring, "device": "cpu",
         "smiles_file": "/work/compounds.smi"}
    cfg = build_scoring_config(p, tmp_path, tmp_path)
    assert cfg["run_type"] == "scoring"
    assert cfg["parameters"]["smiles_file"] == "/work/compounds.smi"
    assert cfg["parameters"]["output_csv"] == str(tmp_path / "score_results.csv")
    assert cfg["parameters"]["smiles_column"] == "SMILES"
    # scoring passed through verbatim, with parallel injected
    assert cfg["scoring"]["type"] == "geometric_mean"
    assert cfg["scoring"]["parallel"] == 4
    assert cfg["scoring"]["component"][0]["QED"]["endpoint"][0]["weight"] == 0.5
