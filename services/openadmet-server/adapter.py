"""Service-wide policy for openadmet-server."""

from __future__ import annotations

from pathlib import Path

from bioagent_service import EndpointExample, JobAdapter

from .settings import OpenAdmetSettings


class OpenAdmetAdapter(JobAdapter):
    name = "openadmet"

    settings: OpenAdmetSettings  # narrow for IDEs

    def __init__(self, settings: OpenAdmetSettings) -> None:
        super().__init__(settings)

    # ---- Output detection ----

    def detect_outputs(self, job_dir: Path) -> bool:
        """Any endpoint is considered complete iff its primary artifact is non-empty.

        The framework dispatches by label; we treat both known labels here
        (predict / compare) plus fall back to "output_dir non-empty" for
        forward compatibility.
        """
        out = self.output_dir(job_dir)
        # predict → predictions.csv
        pred = out / "predictions.csv"
        if pred.exists() and pred.stat().st_size > 0:
            return True
        # compare → comparison_stats.json
        stats = out / "comparison_stats.json"
        if stats.exists() and stats.stat().st_size > 0:
            return True
        # Fallback: any non-empty file under output/
        return any(p.is_file() and p.stat().st_size > 0 for p in out.rglob("*"))

    # ---- Subprocess env ----

    def subprocess_cwd(self) -> Path | None:
        # Upstream openadmet CLI is CWD-agnostic (uses absolute paths), but
        # pin cwd to the upstream root for consistency with other services.
        return self.settings.root

    # ---- Manifest enrichments (agent protocol) ----

    def manifest_extras(self) -> dict:
        # Snapshot the currently registered models so agents can inspect
        # the registry without hitting /api/models.
        models = [
            {
                "name": m.name,
                "biotargets": m.biotargets,
                "target_cols": m.target_cols,
                "input_col": m.input_col,
                "model_type": m.model_type,
                "feat_type": m.feat_type,
            }
            for m in self.settings.list_models()
        ]
        return {
            "model": {
                "name": "OpenADMET Models",
                "method": (
                    "General-purpose ADMET (Absorption / Distribution / "
                    "Metabolism / Excretion / Toxicity) property prediction "
                    "toolbox. Multi-model registry — each `model_name` is a "
                    "separately-trained anvil model_dir."
                ),
                "task": "ADMET property prediction (multi-model, multi-task)",
                "license_note": "MIT (upstream OpenADMET / OMSF).",
                "output_format": (
                    "predict: CSV with OADMET_PRED_<tag>_<taskname>, "
                    "OADMET_STD_<tag>_<taskname> per (model, task); "
                    "compare: comparison_stats.json + PNG plots + optional PDF."
                ),
            },
            "tool_outputs": {
                "predict": "output/predictions.csv",
                "compare": "output/comparison_stats.json (+ output/plots/*.png, output/report.pdf if requested)",
            },
            "input_uri_schemes": {
                "input_csv_uri": "oss:// | file:// | http(s):// | job://",
                "input_sdf_uri": "oss:// | file:// | http(s):// | job://",
                "upload": "multipart/form-data (input_csv or input_sdf)",
            },
            "model_registry": {
                "endpoint": "GET /api/models",
                "current_count": len(models),
                "models": models,
            },
            "config_tips": {
                "accelerator": (
                    "'gpu' (default) is required for chemprop/TabPFN/pairwise NN models. "
                    "'cpu' works for sklearn/xgboost/lgbm/catboost/SVM/RF."
                ),
                "input_col": (
                    "Omit to auto-derive from the first model's data.yaml. "
                    "Only set explicitly if you want to override for a "
                    "custom input CSV."
                ),
                "aq_fxns": (
                    "Active-learning acquisition. UCB needs `beta`; "
                    "EI/PI each need matching `best_y` and `xi`."
                ),
            },
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/predict": [
                EndpointExample(
                    title="Inline SMILES against a single model",
                    curl=(
                        "curl -X POST $URL/api/predict \\\n"
                        "    -F 'input_smiles=CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl,"
                        "CC(=O)OC1=CC=CC=C1C(=O)O' \\\n"
                        "    -F 'model_names=herg-chemeleon-baseline'"
                    ),
                    notes="Predict hERG pIC50 for 2 SMILES using the pre-staged model.",
                ),
                EndpointExample(
                    title="CSV upload against multiple CYP + hERG models",
                    curl=(
                        "curl -X POST $URL/api/predict \\\n"
                        "    -F 'input_csv=@my_compounds.csv' \\\n"
                        "    -F 'model_names=herg-chemeleon-baseline' \\\n"
                        "    -F 'model_names=cyp1a2-cyp2d6-cyp3a4-cyp2c9-chemeleon-v1'"
                    ),
                    notes=(
                        "Multiple `-F 'model_names=X'` accumulate into a list. "
                        "Server auto-derives input_col from the first model's data.yaml."
                    ),
                ),
                EndpointExample(
                    title="Large batch via OSS URI + acquisition scoring",
                    curl=(
                        "curl -X POST $URL/api/predict \\\n"
                        "    -F 'input_csv_uri=oss://my-bucket/screening/lib.csv' \\\n"
                        "    -F 'model_names=cyp1a2-cyp2d6-cyp3a4-cyp2c9-chemeleon-v1' \\\n"
                        "    -F 'aq_fxns=ucb' -F 'beta=1.5' \\\n"
                        "    -F 'aq_fxns=ei'  -F 'xi=0.01' -F 'best_y=6.0'"
                    ),
                    notes=(
                        "URI fallback for GB-scale CSVs. UCB β=1.5 + EI ξ=0.01 (best_y=6.0)."
                    ),
                ),
            ],
            "/api/compare": [
                EndpointExample(
                    title="Compare 2 pre-registered CYP models by biotarget label",
                    curl=(
                        "curl -X POST $URL/api/compare \\\n"
                        "    -F 'model_names=cyp1a2-cyp2d6-cyp3a4-cyp2c9-chemeleon-v1' \\\n"
                        "    -F 'model_names=cyp1a2-cyp2d6-cyp3a4-cyp3c9-chemeleon-baseline' \\\n"
                        "    -F 'label_types=biotarget' -F 'label_types=biotarget' \\\n"
                        "    -F 'mt_id=CYP3A4'"
                    ),
                    notes="Mode A — model_dir-based comparison with biotarget labels.",
                ),
                EndpointExample(
                    title="Compare via pre-computed stats JSON",
                    curl=(
                        "curl -X POST $URL/api/compare \\\n"
                        "    -F 'model_stats_files=@stats_a.json' \\\n"
                        "    -F 'model_stats_files=@stats_b.json' \\\n"
                        "    -F 'labels=modelA' -F 'labels=modelB' \\\n"
                        "    -F 'task_names=pchembl_value_mean' \\\n"
                        "    -F 'task_names=pchembl_value_mean' \\\n"
                        "    -F 'report=true'"
                    ),
                    notes="Mode B — JSON stats + PDF report.",
                ),
            ],
        }
