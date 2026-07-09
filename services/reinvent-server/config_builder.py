"""Pure functions: pydantic-request dicts → REINVENT TOML config dicts.

No I/O, no subprocess — trivially unit-testable. `reinvent_cli` writes the
returned dict to config.toml via tomli_w. Output paths are absolute (under the
job's output/ dir) so reinvent writes results directly where detect_outputs
looks; input file paths in `p` are already staged/absolute by reinvent_cli.
"""
from __future__ import annotations

from pathlib import Path

# dot-key → filename, matching the files published on Zenodo record 20701824
# ("REINVENT4 priors", the concept-DOI's current version). This release ships a
# single Mol2Mol prior (the PubChem ECFP4 similarity model) — the older
# mol2mol_{high,medium,mmp,scaffold,...}_similarity.prior variants are NOT
# distributed here — and the de-novo prior is named reinvent_pubchem.prior
# (not reinvent.prior). The two *_transformer_pubchem priors pair with REINVENT's
# LibinventTransformer / LinkinventTransformer generators (not in v0.0.1's
# `generator` enum, but referenceable via an explicit model_file dot-key).
PRIOR_FILES = {
    ".reinvent": "reinvent_pubchem.prior",
    ".libinvent": "libinvent.prior",
    ".libinvent_transformer": "libinvent_transformer_pubchem.prior",
    ".linkinvent": "linkinvent.prior",
    ".linkinvent_transformer": "linkinvent_transformer_pubchem.prior",
    ".mol2mol": "pubchem_ecfp4_with_count_with_rank_reinvent4_dict_voc.prior",
    ".pepinvent": "pepinvent.prior",
}
GENERATOR_DEFAULT_PRIOR = {
    "reinvent": ".reinvent",
    "libinvent": ".libinvent",
    "linkinvent": ".linkinvent",
    "mol2mol": ".mol2mol",
    "pepinvent": ".pepinvent",
}


def _resolve_prior(model_file: str | None, generator: str, prior_base: Path) -> str:
    """Resolve a prior reference to an absolute path REINVENT can open.

    - None  → generator's default dot-key → prior_base/<file>
    - ".x"  → registry dot-key            → prior_base/<file>
    - other → explicit path (staged upload basename or absolute) → passthrough
    """
    key = model_file or GENERATOR_DEFAULT_PRIOR[generator]
    if key.startswith("."):
        return str(prior_base / PRIOR_FILES[key])
    return key


def build_sampling_config(p: dict, output_dir: Path, prior_base: Path) -> dict:
    params = {
        "model_file": _resolve_prior(p.get("model_file"), p["generator"], prior_base),
        "num_smiles": p["num_smiles"],
        "unique_molecules": p["unique_molecules"],
        "randomize_smiles": p["randomize_smiles"],
        "temperature": p["temperature"],
        "sample_strategy": p["sample_strategy"],
        "output_file": str(output_dir / "sampling.csv"),
    }
    if p.get("smiles_file"):
        params["smiles_file"] = p["smiles_file"]
    return {
        "run_type": "sampling",
        "device": p["device"],
        "json_out_config": str(output_dir / "_sampling.json"),
        "parameters": params,
    }


def build_scoring_config(p: dict, output_dir: Path, prior_base: Path) -> dict:
    scoring = dict(p["scoring"])  # shallow copy; do not mutate caller's dict
    scoring["parallel"] = p["parallel"]
    return {
        "run_type": "scoring",
        "device": p["device"],
        "json_out_config": str(output_dir / "_scoring.json"),
        "parameters": {
            "smiles_file": p["smiles_file"],
            "smiles_column": p["smiles_column"],
            "standardize_smiles": p["standardize_smiles"],
            "output_csv": str(output_dir / "score_results.csv"),
        },
        "scoring": scoring,
    }


def build_enumeration_config(p: dict, output_dir: Path, prior_base: Path) -> dict:
    return {
        "run_type": "enumeration",
        "device": p["device"],
        "json_out_config": str(output_dir / "_enumeration.json"),
        "parameters": {
            "batch_size": p["batch_size"],
            "smiles_file": p["smiles_file"],
            # Upstream aa_enumerator.py + validation.py read amino_acid_library_file;
            # the shipped example config's `amino_acid_library` key is stale.
            "amino_acid_library_file": p["amino_acid_library_file"],
            "amino_acid_name_column": p["amino_acid_name_column"],
            "smiles_column": p["smiles_column"],
            "output_csv": str(output_dir / "peptide_enumeration.csv"),
        },
        "scoring": dict(p["scoring"]),
    }


def build_tl_config(p: dict, output_dir: Path, prior_base: Path) -> dict:
    params = {
        "input_model_file": _resolve_prior(
            p.get("input_model_file"), p["generator"], prior_base),
        "output_model_file": str(output_dir / p["output_model_name"]),
        "smiles_file": p["smiles_file"],
        "num_epochs": p["num_epochs"],
        "save_every_n_epochs": p["save_every_n_epochs"],
        "batch_size": p["batch_size"],
        "num_refs": p["num_refs"],
        "sample_batch_size": p["sample_batch_size"],
    }
    if p.get("validation_smiles_file"):
        params["validation_smiles_file"] = p["validation_smiles_file"]
    if p.get("pairs"):
        params["pairs"] = dict(p["pairs"])
    return {
        "run_type": "transfer_learning",
        "device": p["device"],
        "tb_logdir": str(output_dir / "tb_TL"),
        "json_out_config": str(output_dir / "_transfer_learning.json"),
        "parameters": params,
    }


def build_rl_config(p: dict, output_dir: Path, prior_base: Path) -> dict:
    prior = _resolve_prior(p.get("prior_file"), p["generator"], prior_base)
    agent = (_resolve_prior(p["agent_file"], p["generator"], prior_base)
             if p.get("agent_file") else prior)
    params = {
        "prior_file": prior,
        "agent_file": agent,
        "batch_size": p["batch_size"],
        "summary_csv_prefix": str(output_dir / p["summary_csv_prefix"]),
        "use_checkpoint": p["use_checkpoint"],
        "purge_memories": p["purge_memories"],
        "randomize_smiles": p["randomize_smiles"],
    }
    if p.get("smiles_file"):
        params["smiles_file"] = p["smiles_file"]

    cfg: dict = {
        "run_type": "staged_learning",
        "device": p["device"],
        "tb_logdir": str(output_dir / "tb_logs"),
        "json_out_config": str(output_dir / "_staged_learning.json"),
        "parameters": params,
        "learning_strategy": dict(p["learning_strategy"]),
        "stage": [
            {
                "chkpt_file": str(output_dir / s["chkpt_name"]),
                "termination": s["termination"],
                "max_score": s["max_score"],
                "min_steps": s["min_steps"],
                "max_steps": s["max_steps"],
                "scoring": dict(s["scoring"]),
            }
            for s in p["stages"]
        ],
    }
    if p.get("diversity_filter"):
        cfg["diversity_filter"] = dict(p["diversity_filter"])
    if p.get("inception"):
        cfg["inception"] = dict(p["inception"])
    return cfg


BUILDERS = {
    "sampling": build_sampling_config,
    "scoring": build_scoring_config,
    "enumeration": build_enumeration_config,
    "transfer_learning": build_tl_config,
    "staged_learning": build_rl_config,
}
