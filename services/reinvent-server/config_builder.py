"""Pure functions: pydantic-request dicts → REINVENT TOML config dicts.

No I/O, no subprocess — trivially unit-testable. `reinvent_cli` writes the
returned dict to config.toml via tomli_w. Output paths are absolute (under the
job's output/ dir) so reinvent writes results directly where detect_outputs
looks; input file paths in `p` are already staged/absolute by reinvent_cli.
"""
from __future__ import annotations

from pathlib import Path

PRIOR_FILES = {
    ".reinvent": "reinvent.prior",
    ".libinvent": "libinvent.prior",
    ".linkinvent": "linkinvent.prior",
    ".m2m_high": "mol2mol_high_similarity.prior",
    ".m2m_medium": "mol2mol_medium_similarity.prior",
    ".m2m_mmp": "mol2mol_mmp.prior",
    ".m2m_scaffold": "mol2mol_scaffold.prior",
    ".m2m_scaffold_generic": "mol2mol_scaffold_generic.prior",
    ".pepinvent": "pepinvent.prior",
}
GENERATOR_DEFAULT_PRIOR = {
    "reinvent": ".reinvent",
    "libinvent": ".libinvent",
    "linkinvent": ".linkinvent",
    "mol2mol": ".m2m_medium",
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


BUILDERS = {
    "sampling": build_sampling_config,
    "scoring": build_scoring_config,
    "enumeration": build_enumeration_config,
}
