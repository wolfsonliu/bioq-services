from pathlib import Path


def test_resolve_prior_dot_key():
    from server.config_builder import _resolve_prior
    got = _resolve_prior(".reinvent", "reinvent", Path("/nas/priors"))
    assert got == "/nas/priors/reinvent_pubchem.prior"


def test_resolve_prior_default_for_generator():
    from server.config_builder import _resolve_prior
    got = _resolve_prior(None, "mol2mol", Path("/nas/priors"))
    assert got == "/nas/priors/pubchem_ecfp4_with_count_with_rank_reinvent4_dict_voc.prior"


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
    assert cfg["parameters"]["model_file"] == "/nas/priors/reinvent_pubchem.prior"
    assert cfg["parameters"]["num_smiles"] == 25
    assert cfg["parameters"]["output_file"] == str(out / "sampling.csv")
    assert "smiles_file" not in cfg["parameters"]


def test_build_sampling_config_with_seed(tmp_path):
    from server.config_builder import build_sampling_config
    p = {"generator": "mol2mol", "model_file": ".mol2mol", "num_smiles": 10,
         "unique_molecules": True, "randomize_smiles": False, "temperature": 1.2,
         "sample_strategy": "beamsearch", "device": "cuda:0",
         "smiles_file": "/work/seed.smi"}
    cfg = build_sampling_config(p, tmp_path, Path("/nas/priors"))
    assert cfg["parameters"]["smiles_file"] == "/work/seed.smi"
    assert cfg["parameters"]["sample_strategy"] == "beamsearch"
    assert cfg["parameters"]["model_file"] == \
        "/nas/priors/pubchem_ecfp4_with_count_with_rank_reinvent4_dict_voc.prior"


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


def test_build_enumeration_config(tmp_path):
    from server.config_builder import build_enumeration_config
    p = {"batch_size": 20, "amino_acid_name_column": "Name",
         "smiles_column": "Smiles", "scoring": {"type": "geometric_mean", "component": []},
         "device": "cpu", "smiles_file": "/work/peptide.smi",
         "amino_acid_library_file": "/work/library.csv"}
    cfg = build_enumeration_config(p, tmp_path, tmp_path)
    assert cfg["run_type"] == "enumeration"
    par = cfg["parameters"]
    assert par["batch_size"] == 20
    assert par["smiles_file"] == "/work/peptide.smi"
    # upstream SectionParameters (extra="forbid") reads amino_acid_library_file
    # (NOT amino_acid_library) and aa_names_column (NOT amino_acid_name_column).
    assert par["amino_acid_library_file"] == "/work/library.csv"
    assert par["aa_names_column"] == "Name"
    assert "amino_acid_name_column" not in par
    assert par["output_csv"] == str(tmp_path / "peptide_enumeration.csv")
    assert cfg["scoring"]["type"] == "geometric_mean"


def test_build_tl_config_reinvent(tmp_path):
    from server.config_builder import build_tl_config
    p = {"generator": "reinvent", "input_model_file": None,
         "output_model_name": "TL_out.model", "num_epochs": 5,
         "save_every_n_epochs": 1, "batch_size": 50, "num_refs": 0,
         "sample_batch_size": 100, "pairs": None, "device": "cuda:0",
         "smiles_file": "/work/target.smi", "validation_smiles_file": None}
    cfg = build_tl_config(p, tmp_path, Path("/nas/priors"))
    assert cfg["run_type"] == "transfer_learning"
    par = cfg["parameters"]
    assert par["input_model_file"] == "/nas/priors/reinvent_pubchem.prior"
    assert par["output_model_file"] == str(tmp_path / "TL_out.model")
    assert par["smiles_file"] == "/work/target.smi"
    assert par["num_epochs"] == 5
    assert "validation_smiles_file" not in par
    assert "pairs" not in par
    assert cfg["tb_logdir"] == str(tmp_path / "tb_TL")


def test_build_tl_config_mol2mol_pairs(tmp_path):
    from server.config_builder import build_tl_config
    p = {"generator": "mol2mol", "input_model_file": ".mol2mol",
         "output_model_name": "TL.model", "num_epochs": 3,
         "save_every_n_epochs": 3, "batch_size": 50, "num_refs": 100,
         "sample_batch_size": 100,
         "pairs": {"type": "tanimoto", "upper_threshold": 1.0, "lower_threshold": 0.7},
         "device": "cuda:0", "smiles_file": "/work/c.smi",
         "validation_smiles_file": "/work/v.smi"}
    cfg = build_tl_config(p, tmp_path, Path("/nas/priors"))
    par = cfg["parameters"]
    assert par["validation_smiles_file"] == "/work/v.smi"
    assert par["pairs"]["type"] == "tanimoto"
    assert par["pairs"]["lower_threshold"] == 0.7


def test_build_rl_config_full(tmp_path):
    from server.config_builder import build_rl_config
    p = {
        "generator": "reinvent", "prior_file": None, "agent_file": None,
        "batch_size": 64, "summary_csv_prefix": "sl", "use_checkpoint": False,
        "purge_memories": False, "randomize_smiles": True,
        "learning_strategy": {"type": "dap", "sigma": 128, "rate": 0.0001},
        "diversity_filter": {"type": "IdenticalMurckoScaffold", "bucket_size": 25,
                             "minscore": 0.4},
        "inception": {"memory_size": 100, "sample_size": 10},
        "device": "cuda:0", "smiles_file": None,
        "stages": [
            {"chkpt_name": "s1.chkpt", "termination": "simple", "max_score": 0.6,
             "min_steps": 25, "max_steps": 100,
             "scoring": {"type": "geometric_mean", "component": []}},
            {"chkpt_name": "s2.chkpt", "termination": "simple", "max_score": 0.7,
             "min_steps": 10, "max_steps": 100,
             "scoring": {"type": "geometric_mean", "component": []}},
        ],
    }
    cfg = build_rl_config(p, tmp_path, Path("/nas/priors"))
    assert cfg["run_type"] == "staged_learning"
    par = cfg["parameters"]
    assert par["prior_file"] == "/nas/priors/reinvent_pubchem.prior"
    assert par["agent_file"] == "/nas/priors/reinvent_pubchem.prior"  # None → prior
    assert par["summary_csv_prefix"] == str(tmp_path / "sl")
    assert cfg["learning_strategy"]["sigma"] == 128
    assert cfg["diversity_filter"]["type"] == "IdenticalMurckoScaffold"
    assert cfg["inception"]["memory_size"] == 100
    assert isinstance(cfg["stage"], list) and len(cfg["stage"]) == 2
    assert cfg["stage"][0]["chkpt_file"] == str(tmp_path / "s1.chkpt")
    assert cfg["stage"][0]["scoring"]["type"] == "geometric_mean"
    assert cfg["stage"][1]["max_score"] == 0.7
    assert "smiles_file" not in par


def test_build_rl_config_omits_optional_sections(tmp_path):
    from server.config_builder import build_rl_config
    p = {
        "generator": "libinvent", "prior_file": ".libinvent",
        "agent_file": "/work/agent.chkpt", "batch_size": 32,
        "summary_csv_prefix": "sl", "use_checkpoint": True, "purge_memories": False,
        "randomize_smiles": True,
        "learning_strategy": {"type": "dap", "sigma": 120, "rate": 0.0001},
        "diversity_filter": None, "inception": None, "device": "cpu",
        "smiles_file": "/work/scaffolds.smi",
        "stages": [{"chkpt_name": "s1.chkpt", "termination": "simple",
                    "max_score": 0.6, "min_steps": 25, "max_steps": 50,
                    "scoring": {"type": "arithmetic_mean", "component": []}}],
    }
    cfg = build_rl_config(p, tmp_path, Path("/nas/priors"))
    assert "diversity_filter" not in cfg
    assert "inception" not in cfg
    assert cfg["parameters"]["agent_file"] == "/work/agent.chkpt"
    assert cfg["parameters"]["smiles_file"] == "/work/scaffolds.smi"
    assert cfg["parameters"]["use_checkpoint"] is True
