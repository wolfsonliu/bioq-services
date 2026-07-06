"""Wrapper around upstream ``inference.main``.

This wrapper does three things beyond what a raw ``python -m inference``
call would do:

1. **Input validation** — enforce mutex between ``--protein_path`` and
   ``--protein_sequence`` and normalize the ligand argument.
2. **Environment / CWD setup** — points ``TORCH_HOME`` at the NAS ESM
   cache and chdir's into the upstream repo root so ``.so3_*.npy`` +
   ``.torus_*.npy`` LUT files (loaded by ``utils/so3.py`` and
   ``utils/torus.py`` via CWD-relative paths) resolve correctly.
3. **Post-processing** — scans ``output/<complex_name>/`` for the
   ``rank<r>_confidence<c>.sdf`` files that upstream writes and produces
   a single ``confidence_scores.json`` for downstream consumption.

Named ``run_inference.py`` (not ``inference.py``) to avoid clashing with
the upstream module of the same name — the wrapper does
``from inference import main`` and would recurse into itself if it
shared the name.

The wrapper is invoked as a CLI script by ``tools.dock_argv``; both
HTTP and batch (sbatch) modes go through it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_RANK_RE = re.compile(r"rank(\d+)_confidence(-?[\d.]+)\.sdf")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="diffdock-server run_inference")

    # Protein input (mutex — enforced below)
    p.add_argument("--protein_path", type=Path, default=None)
    p.add_argument("--protein_sequence", type=str, default=None)

    # Ligand input (single arg; upstream's --ligand_description takes
    # either a file path or a SMILES string)
    p.add_argument("--ligand", type=str, required=True)

    # Job-level params
    p.add_argument("--complex_name", type=str, default="complex_0")
    p.add_argument("--out_dir", type=Path, required=True)

    # Diffusion / sampling params
    p.add_argument("--samples_per_complex", type=int, default=10)
    p.add_argument("--inference_steps", type=int, default=20)
    p.add_argument("--actual_steps", type=int, default=19)
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument(
        "--no_final_step_noise",
        type=lambda s: s.lower() in ("true", "1", "yes"),
        default=True,
    )
    p.add_argument(
        "--save_visualisation",
        type=lambda s: s.lower() in ("true", "1", "yes"),
        default=False,
    )
    p.add_argument("--seed", type=int, default=0)

    # Weights / config paths
    p.add_argument("--model_dir", type=Path, required=True)
    p.add_argument("--confidence_model_dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--torchhub_dir", type=Path, required=True)

    args = p.parse_args()

    if (args.protein_path is None) == (args.protein_sequence is None):
        p.error("Exactly one of --protein_path / --protein_sequence must be given")
    return args


def build_upstream_argv(args: argparse.Namespace) -> list[str]:
    """Assemble the argv that upstream ``inference.get_parser`` will parse.

    The leading program name is stripped from ``sys.argv`` in ``main()``
    before ``get_parser().parse_args()`` runs.
    """
    argv = [
        "--config", str(args.config),
        "--complex_name", args.complex_name,
        "--ligand_description", args.ligand,
        "--out_dir", str(args.out_dir),
        "--samples_per_complex", str(args.samples_per_complex),
        "--inference_steps", str(args.inference_steps),
        "--actual_steps", str(args.actual_steps),
        "--batch_size", str(args.batch_size),
        "--model_dir", str(args.model_dir),
        "--confidence_model_dir", str(args.confidence_model_dir),
    ]
    if args.protein_path is not None:
        argv += ["--protein_path", str(args.protein_path)]
    else:
        argv += ["--protein_sequence", args.protein_sequence]
    # Upstream --save_visualisation is store_true; only append when True.
    if args.save_visualisation:
        argv += ["--save_visualisation"]
    # Upstream --no_final_step_noise is store_true, default True.  When
    # user requests False we cannot turn it off from CLI (see design doc
    # §Risks §4).  When user requests True the flag is redundant (default).
    # Either way we skip appending; the config yaml governs the effective value.
    return argv


def postprocess(complex_out_dir: Path) -> Path | None:
    """Scan ``rank<r>_confidence<c>.sdf`` files, write confidence_scores.json.

    Returns the path to the JSON, or None if no ranked files were found
    (a "failed run" — detect_outputs will pick this up).
    """
    if not complex_out_dir.is_dir():
        return None
    entries = []
    for f in sorted(complex_out_dir.iterdir()):
        m = _RANK_RE.fullmatch(f.name)
        if m:
            entries.append({
                "rank": int(m.group(1)),
                "confidence": float(m.group(2)),
                "sdf": f.name,
            })
    if not entries:
        return None
    entries.sort(key=lambda e: e["rank"])
    dst = complex_out_dir / "confidence_scores.json"
    dst.write_text(json.dumps(entries, indent=2))
    return dst


def main() -> None:
    args = parse_args()

    # Env before importing upstream: TORCH_HOME points fair-esm's
    # torch.hub loader at the NAS cache; REPOSITORY_URL disables the
    # in-line release-zip fallback download that would hang on FC.
    os.environ.setdefault("TORCH_HOME", str(args.torchhub_dir))
    os.environ.setdefault("REPOSITORY_URL", "file:///dev/null")

    # Anchor CWD at upstream root so utils/so3.py:46
    # (np.load('.so3_omegas_array4.npy')) and utils/torus.py find the
    # pre-computed LUT `.npy` files that ship in the image.  Upstream
    # root is expected to be settings.root (/opt/diffdock in the image);
    # the wrapper takes it from DIFFDOCK_ROOT if set, else falls back to
    # CWD (already fine for CLI batch use).
    upstream_root = Path(os.environ.get("DIFFDOCK_ROOT", os.getcwd()))
    os.chdir(upstream_root)
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))

    # Rewrite argv for upstream's parse_args().
    sys.argv = ["inference"] + build_upstream_argv(args)

    # Deferred imports: upstream inference module imports torch + fair-esm
    # + prody at top level, all of which take a few seconds and eagerly
    # touch CUDA.  Import only after env / cwd are set.
    from inference import get_parser
    from inference import main as upstream_main  # type: ignore

    parsed = get_parser().parse_args()
    upstream_main(parsed)

    # Post-process the specific complex output directory.
    postprocess(args.out_dir / args.complex_name)


if __name__ == "__main__":
    main()
