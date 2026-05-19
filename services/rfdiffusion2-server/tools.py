"""argv builders for `rf_diffusion/run_inference.py`.

RFdiffusion2 is Hydra-driven; every key in `config/inference/base.yaml` (and
the `aa.yaml` overlay that inference scripts default to) can be overridden by
`key.path=value` on the command line. Each builder here turns a typed pydantic
request into the corresponding override list and returns the argv that
`JobRunner.submit` will execute.

Output convention
-----------------
All endpoints write under `<job_dir>/output/`:

  * `design_0.pdb`, `design_1.pdb`, ... — final per-trajectory backbones
  * `design_<N>.trb`                    — metadata pickles (contig, config, mapping)
  * `traj/...`                          — full trajectories (only if write_trajectory=True)

`inference.output_prefix=<output_dir>/design` makes that happen. The adapter's
`detect_outputs` walks `output/` for any `design_*.pdb` to decide COMPLETED vs
NO_OUTPUTS.

Hydra value-syntax notes
------------------------
OmegaConf parses the RHS of each `key=value` token. For Python -> argv:

  * Strings with commas must be quoted: `inference.ligand='NAD,OXM'` (else OmegaConf
    interprets `NAD,OXM` as a list).
  * Dicts: `{A106: 'NE,CD,CZ', A166: 'OD1,CG'}` — keys can be bare, string values quoted.
  * Lists: `[O7N,C7N]` — bare atom names fine because OmegaConf strips bracket-list items.
  * `++key=...` prefix is required when adding a key not already in the base config
    (struct mode); `partially_fixed_ligand` is the one such key today.

Because subprocess argv tokens are NOT shell-interpolated, we don't need to
shell-escape quotes — Hydra sees the raw token as-is.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from .models import (
    MODEL_CHOICES,
    ActiveSiteRequest,
    CustomRequest,
    SmallMoleculeBinderRequest,
)
from .settings import RFdiffusion2Settings

logger = logging.getLogger(__name__)

# All endpoints land outputs at `<job>/output/design_*.pdb`. Keep the stem here
# so adapter.detect_outputs and endpoint examples agree.
OUTPUT_STEM = "design"

# Conservative allow-list for tokens we splice into Hydra values. Hydra args
# go through argv (no shell), but the strings still need to round-trip through
# OmegaConf's YAML-ish parser. Restricting to alphanumerics + a few separators
# keeps the parser happy and rules out injection of stray `=`/`#`/quotes.
_RES_KEY_RE = re.compile(r"^[A-Za-z]\d+$")          # e.g. "A106"
_ATOM_NAME_RE = re.compile(r"^[A-Za-z0-9']{1,8}$")   # PDB atom names
_LIG_NAME_RE = re.compile(r"^[A-Za-z0-9]{1,5}$")     # ligand resnames


def _output_prefix(job_dir: Path) -> Path:
    """Absolute prefix for `inference.output_prefix=` — RFdiffusion2 appends `_N.pdb`."""
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / OUTPUT_STEM


def _resolve_ckpt(model: Optional[str], settings: RFdiffusion2Settings) -> Optional[Path]:
    """Map a friendly model name (`rfd_140`, `rfd_173`) to a checkpoint path."""
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
        raise HTTPException(
            status_code=422,
            detail=f"Checkpoint not present in image: {path}",
        )
    return path


def _common_overrides(
    req,  # _GenerationCommon subclass
    job_dir: Path,
    settings: RFdiffusion2Settings,
) -> list[str]:
    """Hydra overrides shared by every endpoint."""
    overrides = [
        f"inference.output_prefix={_output_prefix(job_dir)}",
        f"inference.num_designs={req.num_designs}",
        f"inference.final_step={req.final_step}",
        f"inference.write_trajectory={str(req.write_trajectory).lower()}",
        f"diffuser.T={req.diffuser_t}",
    ]
    if req.deterministic:
        overrides.append("inference.deterministic=True")
    ckpt = _resolve_ckpt(req.model, settings)
    if ckpt is not None:
        overrides.append(f"inference.ckpt_path={ckpt}")
    return overrides


# ---------------------------------------------------------------------------
# Value rendering helpers
# ---------------------------------------------------------------------------


def _validate_res_key(k: str) -> str:
    if not _RES_KEY_RE.match(k):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid residue key {k!r}; expected `<chain><resnum>` (e.g. `A106`).",
        )
    return k


def _validate_ligand_name(k: str) -> str:
    if not _LIG_NAME_RE.match(k):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid ligand resname {k!r}; expected 1-5 alphanumerics.",
        )
    return k


def _validate_atom_names_csv(s: str) -> str:
    """Validate a comma-separated atom-name string and return it canonicalised."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise HTTPException(
            status_code=422, detail="Empty atom-name list in contig_atoms value."
        )
    for p in parts:
        if not _ATOM_NAME_RE.match(p):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid atom name {p!r}; expected alphanumerics only.",
            )
    return ",".join(parts)


def _render_contig_atoms(d: dict[str, str]) -> str:
    """`{A106: 'NE,CD,CZ', A166: 'OD1,CG'}` — values are quoted strings."""
    if not d:
        raise HTTPException(status_code=422, detail="contig_atoms cannot be empty.")
    parts = [
        f"{_validate_res_key(k)}: '{_validate_atom_names_csv(v)}'"
        for k, v in d.items()
    ]
    return "{" + ", ".join(parts) + "}"


def _render_partially_fixed_ligand(d: dict[str, list[str]]) -> str:
    """`{NAD: [O7N,C7N], OXM: [O3,C2]}` — values are bare-atom lists."""
    if not d:
        raise HTTPException(
            status_code=422, detail="partially_fixed_ligand cannot be empty."
        )
    parts: list[str] = []
    for k, atoms in d.items():
        if not isinstance(atoms, list) or not atoms:
            raise HTTPException(
                status_code=422,
                detail=f"partially_fixed_ligand[{k!r}] must be a non-empty list.",
            )
        for a in atoms:
            if not isinstance(a, str) or not _ATOM_NAME_RE.match(a):
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid atom name {a!r} in partially_fixed_ligand[{k!r}].",
                )
        parts.append(f"{_validate_ligand_name(k)}: [{','.join(atoms)}]")
    return "{" + ", ".join(parts) + "}"


def _render_ligand(ligand: str) -> str:
    """`inference.ligand='NAD,OXM'` — quoted so OmegaConf doesn't list-split."""
    parts = [p.strip() for p in ligand.split(",") if p.strip()]
    if not parts:
        raise HTTPException(status_code=422, detail="ligand cannot be empty.")
    for p in parts:
        _validate_ligand_name(p)
    return f"inference.ligand='{','.join(parts)}'"


def _quote_contigs(contigs: str) -> str:
    """`contigmap.contigs=['46,A106-106,...']` — single-element list, value quoted."""
    if "'" in contigs or '"' in contigs:
        raise HTTPException(
            status_code=422,
            detail="contigs must not contain quote characters.",
        )
    return f"contigmap.contigs=['{contigs}']"


# ---------------------------------------------------------------------------
# Mode-specific argv builders
# ---------------------------------------------------------------------------


def active_site_argv(
    req: ActiveSiteRequest,
    input_pdb: Path,
    job_dir: Path,
    settings: RFdiffusion2Settings,
) -> list[str]:
    """Active-site scaffolding around an atomic motif + ligand."""
    cmd: list[str] = [
        str(settings.python),
        str(settings.inference_script),
        "--config-name=aa",
        f"inference.input_pdb={input_pdb.resolve()}",
        _render_ligand(req.ligand),
        _quote_contigs(req.contigs),
        f"inference.contig_as_guidepost={str(req.contig_as_guidepost).lower()}",
        f"contigmap.contig_atoms={_render_contig_atoms(req.contig_atoms)}",
        *_common_overrides(req, job_dir, settings),
    ]
    if req.only_guidepost_positions:
        if "'" in req.only_guidepost_positions:
            raise HTTPException(
                status_code=422, detail="only_guidepost_positions must not contain quotes."
            )
        cmd.append(f"inference.only_guidepost_positions='{req.only_guidepost_positions}'")
    if req.partially_fixed_ligand:
        # `++` is required because base.yaml has `partially_fixed_ligand: {}`.
        # Hydra accepts the override either way, but the `++` form is what the
        # upstream demos use; keep it consistent for grep-ability.
        cmd.append(
            f"++inference.partially_fixed_ligand={_render_partially_fixed_ligand(req.partially_fixed_ligand)}"
        )
    if req.inpaint_seq:
        cmd.append(f"contigmap.inpaint_seq=[{req.inpaint_seq}]")
    return cmd


def small_molecule_binder_argv(
    req: SmallMoleculeBinderRequest,
    input_pdb: Path,
    job_dir: Path,
    settings: RFdiffusion2Settings,
) -> list[str]:
    """Small-molecule binder design, optionally RASA-conditioned."""
    cmd: list[str] = [
        str(settings.python),
        str(settings.inference_script),
        "--config-name=aa",
        f"inference.input_pdb={input_pdb.resolve()}",
        f"inference.ligand={req.ligand}",  # short single-resname; no comma needed
        _quote_contigs(req.contigs),
        *_common_overrides(req, job_dir, settings),
    ]
    # Single-resname ligand needn't be quoted; comma-separated ligands should
    # use _render_ligand. Validate so we surface bad input early.
    _validate_ligand_name(req.ligand)
    if req.length:
        cmd.append(f"contigmap.length={req.length}")
    if req.rasa_active:
        cmd.append("inference.conditions.relative_sasa_v2.active=True")
        cmd.append(f"inference.conditions.relative_sasa_v2.rasa={req.rasa_target}")
    return cmd


def custom_argv(
    req: CustomRequest,
    input_pdb: Optional[Path],
    job_dir: Path,
    settings: RFdiffusion2Settings,
) -> list[str]:
    """Freeform endpoint — raw contig + JSON of extra `key.path=value` overrides."""
    if req.input_pdb_required and input_pdb is None:
        raise HTTPException(
            status_code=422,
            detail="input_pdb_required=True but no input PDB was provided.",
        )

    cmd: list[str] = [
        str(settings.python),
        str(settings.inference_script),
        f"--config-name={req.config_name}",
        _quote_contigs(req.contigs),
        *_common_overrides(req, job_dir, settings),
    ]
    if input_pdb is not None:
        cmd.append(f"inference.input_pdb={input_pdb.resolve()}")
    if req.ligand:
        cmd.append(_render_ligand(req.ligand))

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
                    status_code=422, detail=f"Invalid override key: {k!r}"
                )
            # Hydra parses lowercase true/false; let booleans through unquoted.
            if isinstance(v, bool):
                cmd.append(f"{k}={str(v).lower()}")
            else:
                cmd.append(f"{k}={v}")
    return cmd
