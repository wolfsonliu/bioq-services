"""argv builders for `scripts/run_inference.py`.

RFdiffusion is fully Hydra-driven: every config key in `config/inference/base.yaml`
(or `symmetry.yaml`) can be overridden by `key.path=value` on the command line.
Each builder here turns a typed pydantic request into the corresponding override
list and returns the argv that `JobRunner.submit` will execute.

Output convention
-----------------
All endpoints write under `<job_dir>/output/`:

  * `design_0.pdb`, `design_1.pdb`, ...  — final per-trajectory backbones
  * `design_0.trb`, ...                  — metadata pickles (contig, config, mapping)
  * `traj/design_0_*.pdb`                — full trajectories (only if write_trajectory=True)

`inference.output_prefix=<output_dir>/design` realises this layout. The adapter's
`detect_outputs` walks `output/` for any `*.pdb` to decide COMPLETED vs
NO_OUTPUTS.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from .models import (
    MODEL_CHOICES,
    BinderRequest,
    CustomRequest,
    MotifRequest,
    SymmetryRequest,
    UnconditionalRequest,
)
from .settings import RFdiffusionSettings

logger = logging.getLogger(__name__)

# All endpoints land outputs at `<job>/output/design_*.pdb`. Keep the prefix
# stem here so adapter.detect_outputs and endpoint examples agree.
OUTPUT_STEM = "design"


def _output_prefix(job_dir: Path) -> Path:
    """Absolute prefix for `inference.output_prefix=` — RFdiffusion appends `_N.pdb` per design."""
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / OUTPUT_STEM


def _resolve_ckpt(model: Optional[str], settings: RFdiffusionSettings) -> Optional[Path]:
    """Map a friendly model name (`base`, `complex_base`, ...) to a checkpoint path."""
    if model is None:
        return None
    fname = MODEL_CHOICES.get(model)
    if fname is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model {model!r}. Choices: {sorted(MODEL_CHOICES)}",
        )
    path = settings.models_dir / fname
    if not path.exists():
        # Don't 500 — surface a clear 422 so the caller knows the image is missing weights.
        raise HTTPException(
            status_code=422,
            detail=f"Checkpoint not present in image: {path}",
        )
    return path


def _common_overrides(
    req,  # _GenerationCommon subclass
    job_dir: Path,
    settings: RFdiffusionSettings,
) -> list[str]:
    """Hydra overrides shared by every endpoint."""
    overrides = [
        f"inference.output_prefix={_output_prefix(job_dir)}",
        f"inference.model_directory_path={settings.models_dir}",
        f"inference.num_designs={req.num_designs}",
        f"inference.final_step={req.final_step}",
        f"inference.write_trajectory={str(req.write_trajectory).lower()}",
        f"diffuser.T={req.diffuser_t}",
        # Reduce noise on both ca and frame in lock-step (the upstream README
        # recommends keeping these tied; clients that want them split can use
        # `/api/generate` with extra_overrides).
        f"denoiser.noise_scale_ca={req.noise_scale}",
        f"denoiser.noise_scale_frame={req.noise_scale}",
    ]
    if req.deterministic:
        overrides.append("inference.deterministic=True")
    ckpt = _resolve_ckpt(req.model, settings)
    if ckpt is not None:
        overrides.append(f"inference.ckpt_override_path={ckpt}")
    return overrides


def _quote_contigs(contigs: str) -> str:
    """Wrap a contig string in the `[...]` Hydra list syntax."""
    return f"contigmap.contigs=[{contigs}]"


# ---------------------------------------------------------------------------
# Mode-specific argv builders
# ---------------------------------------------------------------------------


def unconditional_argv(
    req: UnconditionalRequest,
    job_dir: Path,
    settings: RFdiffusionSettings,
) -> list[str]:
    """Unconditional monomer (`contigmap.contigs=[min-max]`)."""
    if req.min_length > req.max_length:
        raise HTTPException(
            status_code=422,
            detail=f"min_length ({req.min_length}) > max_length ({req.max_length})",
        )

    cmd: list[str] = [
        str(settings.python),
        str(settings.inference_script),
        _quote_contigs(f"{req.min_length}-{req.max_length}"),
        *_common_overrides(req, job_dir, settings),
    ]
    if req.cyclic:
        # RFpeptides macrocycle mode — see opensource/RFdiffusion/README.md
        cmd.extend(["inference.cyclic=True", "inference.cyc_chains=a"])
    return cmd


def motif_argv(
    req: MotifRequest,
    input_pdb: Path,
    job_dir: Path,
    settings: RFdiffusionSettings,
) -> list[str]:
    """Motif scaffolding — needs `inference.input_pdb` + a contig that references the PDB."""
    cmd: list[str] = [
        str(settings.python),
        str(settings.inference_script),
        f"inference.input_pdb={input_pdb.resolve()}",
        _quote_contigs(req.contigs),
        *_common_overrides(req, job_dir, settings),
    ]
    if req.length:
        cmd.append(f"contigmap.length={req.length}")
    if req.inpaint_seq:
        cmd.append(f"contigmap.inpaint_seq=[{req.inpaint_seq}]")
    return cmd


def binder_argv(
    req: BinderRequest,
    target_pdb: Path,
    job_dir: Path,
    settings: RFdiffusionSettings,
) -> list[str]:
    """PPI binder — target PDB + contig (`A1-150/0 70-100`) + optional hotspots."""
    cmd: list[str] = [
        str(settings.python),
        str(settings.inference_script),
        f"inference.input_pdb={target_pdb.resolve()}",
        _quote_contigs(req.contigs),
        *_common_overrides(req, job_dir, settings),
    ]
    if req.hotspots:
        hs = [h.strip() for h in req.hotspots.split(",") if h.strip()]
        cmd.append(f"ppi.hotspot_res=[{','.join(hs)}]")
    return cmd


def symmetry_argv(
    req: SymmetryRequest,
    job_dir: Path,
    settings: RFdiffusionSettings,
) -> list[str]:
    """Symmetric oligomer — separate Hydra config and a few extra knobs."""
    cmd: list[str] = [
        str(settings.python),
        str(settings.inference_script),
        "--config-name", "symmetry",
        f"inference.symmetry={req.symmetry}",
        _quote_contigs(f"{req.total_length}-{req.total_length}"),
        *_common_overrides(req, job_dir, settings),
    ]
    if req.guiding_potentials:
        cmd.append(f"potentials.guiding_potentials={req.guiding_potentials}")
    if req.guide_scale is not None:
        cmd.append(f"potentials.guide_scale={req.guide_scale}")
    if req.guide_decay:
        cmd.append(f"potentials.guide_decay={req.guide_decay}")
    if req.olig_inter_all:
        cmd.append("potentials.olig_inter_all=True")
    if req.olig_intra_all:
        cmd.append("potentials.olig_intra_all=True")
    return cmd


def custom_argv(
    req: CustomRequest,
    input_pdb: Optional[Path],
    job_dir: Path,
    settings: RFdiffusionSettings,
) -> list[str]:
    """Freeform endpoint — raw contig + JSON of extra `key.path=value` overrides."""
    cmd: list[str] = [
        str(settings.python),
        str(settings.inference_script),
        "--config-name", req.config_name,
        _quote_contigs(req.contigs),
        *_common_overrides(req, job_dir, settings),
    ]
    if input_pdb is not None:
        cmd.append(f"inference.input_pdb={input_pdb.resolve()}")

    if req.extra_overrides:
        try:
            extras = json.loads(req.extra_overrides)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=422,
                detail=f"extra_overrides must be valid JSON: {e}",
            ) from e
        if not isinstance(extras, dict):
            raise HTTPException(
                status_code=422,
                detail="extra_overrides must be a JSON object (mapping)",
            )
        for k, v in extras.items():
            if not isinstance(k, str) or "=" in k:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid override key: {k!r}",
                )
            # Hydra parses lowercase true/false; let booleans through unquoted.
            if isinstance(v, bool):
                cmd.append(f"{k}={str(v).lower()}")
            else:
                cmd.append(f"{k}={v}")
    return cmd
