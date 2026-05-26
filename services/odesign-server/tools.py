"""Argv builders for odesign-server.

Constructs the Hydra CLI command for ODesign inference.
"""

from __future__ import annotations

from pathlib import Path

from .models import DesignRequest
from .settings import ODesignSettings

MODEL_TO_MODALITY: dict[str, str] = {
    "odesign_base_prot_flex": "protein",
    "odesign_base_prot_rigid": "protein",
    "odesign_base_ligand_rigid": "ligand",
}


def infer_modality(req: DesignRequest) -> str:
    if req.design_modality:
        return req.design_modality
    modality = MODEL_TO_MODALITY.get(req.model)
    if modality is None:
        raise ValueError(
            f"design_modality is required for model '{req.model}' "
            "(na_rigid needs explicit 'dna' or 'rna')"
        )
    return modality


def design_argv(
    req: DesignRequest,
    *,
    job_dir: Path,
    json_path: Path,
    settings: ODesignSettings,
) -> list[str]:
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    modality = infer_modality(req)

    argv = [
        settings.python,
        settings.inference_script,
        f"exp=train_{req.model}",
        f"data_root_dir={settings.data_root_dir}",
        f"ckpt_root_dir={settings.ckpt_root_dir}",
        f"exp.infer_model_name={req.model}",
        f"exp.design_modality={modality}",
        f"exp.input_json_path={json_path}",
        f"exp.exp_name=job",
        f"exp.seeds={req.seeds}",
        f"exp.model.sample_diffusion.N_sample={req.n_sample}",
        f"exp.use_msa=false",
        f"exp.num_workers={req.num_workers}",
        f"exp.invfold_topk={req.invfold_topk}",
        f"exp.invfold_temp={req.invfold_temp}",
        f"exp.model.inference_noise_schedulers.coordinate.partial_diffusion.enable={str(req.enable_partial_diff).lower()}",
        f"exp.model.inference_noise_schedulers.coordinate.partial_diffusion.snr={req.partial_diff_snr}",
        f"hydra.run.dir={output_dir}",
    ]
    return argv
