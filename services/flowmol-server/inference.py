"""Thin wrapper around `flowmol` for the flowmol-server service.

Why not just call upstream `test.py`?

  1. `flowmol.load_pretrained(name)` (imported by users of the library, and
     transitively via `flowmol.__init__`) runs `subprocess.run("wget ...")`
     when the checkpoint is missing.  FC / airgapped SIF have no egress —
     we load checkpoints explicitly from NAS instead.
  2. `flowmol.models.flowmol` does `from flowmol.analysis.metrics import
     SampleAnalyzer` at module scope, and `FlowMol.__init__` constructs a
     `SampleAnalyzer` unconditionally.  That drags in `import wandb` +
     `import posebusters` (module top of metrics.py) and, at construction,
     `pb.PoseBusters(...)` + a `data/<dataset>/energy_dist.npz` +
     `train_data_valencies_*.json` — none of which the pure-generation path
     touches (`.analyze()` is only called in `training_step`).  We stub
     `wandb` + `posebusters` so the module imports, then replace
     `SampleAnalyzer` with a no-op so `__init__` skips posebusters / the
     energy-dist file (which upstream doesn't even ship for
     `geom_full_kekulized`).  See adding-a-new-service.md §"包装 conda-based
     upstream 的常见陷阱".
  3. Upstream defaults `output_file` to `<model_dir>/samples/sampled_mols.sdf`.
     `model_dir` lives on read-only NAS in production — we route output to
     `<job_dir>/output/molecules.sdf`.

The wrapper writes:

  <output_dir>/molecules.sdf         — up to n_mols valid mols (kekulize=False)
  <output_dir>/sampling_stats.json   — {n_requested, n_written, invalid_count,
                                         sampling_time_seconds, ...}

Exit codes:
  0 = at least one valid mol written
  1 = zero valid mols (adapter treats as FAILED)
  2 = pre-flight validation failed (missing weight / config)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from pathlib import Path


# ---------------------------------------------------------------------------
# Dead-import stubs — MUST be injected before ANY `flowmol` import.
# ---------------------------------------------------------------------------
# `flowmol.analysis.metrics` unconditionally `import wandb` AND
# `import posebusters as pb` at top level.  The metrics path is only reached
# via `SampleAnalyzer.analyze()` (training_step / `--metrics`), never during
# pure generation.  Stubbing both lets metrics.py import without pulling the
# ~50 MB wandb wheel or the ~500 MB posebusters+xtb+dftd4 stack.  The
# matching no-op replacement of `SampleAnalyzer` itself (which `FlowMol.
# __init__` constructs unconditionally) happens in main(), after the module
# is imported.  See engineering/guides/adding-a-new-service.md
# §"包装 conda-based upstream 的常见陷阱" item 4.
sys.modules.setdefault("wandb", types.ModuleType("wandb"))
sys.modules.setdefault("posebusters", types.ModuleType("posebusters"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FlowMol3 unconditional 3D "
                                            "small-molecule generation")
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Trained model directory containing "
                        "checkpoints/last.ckpt + config.yaml.")
    p.add_argument("--output-file", type=Path, required=True,
                   help="Output SDF file path (a single SDF containing "
                        "all valid molecules).")
    p.add_argument("--stats-file", type=Path, required=True,
                   help="Sampling stats JSON output.")
    p.add_argument("--n-mols", type=int, default=100)
    p.add_argument("--n-timesteps", type=int, default=250)
    p.add_argument("--n-atoms-per-mol", type=int, default=None)
    p.add_argument("--stochasticity", type=float, default=None)
    p.add_argument("--hc-thresh", type=float, default=None)
    p.add_argument("--max-batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def validate(args: argparse.Namespace) -> None:
    """Fail early on missing weights — before triggering the ~20 s import.

    Exits with code 2 on validation failure so the framework's rc
    classification does not mistake it for a normal FAILED sample run.
    """
    ckpt = args.model_dir / "checkpoints" / "last.ckpt"
    cfg = args.model_dir / "config.yaml"
    for path in (ckpt, cfg):
        if not path.exists():
            print(f"ERROR: required file not found: {path}", file=sys.stderr)
            sys.exit(2)

    if args.hc_thresh is not None and not (0.0 <= args.hc_thresh <= 1.0):
        print(f"ERROR: --hc-thresh must be in [0, 1]; got {args.hc_thresh}",
              file=sys.stderr)
        sys.exit(2)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.stats_file.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    validate(args)

    # Heavy imports deferred: any validation failure surfaces immediately
    # without paying the ~20 s pytorch + dgl + PyG import cost.
    import torch
    from pytorch_lightning import seed_everything
    from rdkit import Chem, RDLogger
    import flowmol.models.flowmol as flowmol_mod
    from flowmol.models.flowmol import FlowMol

    # Neutralise the metrics-only SampleAnalyzer that FlowMol.__init__
    # constructs unconditionally.  Its real __init__ calls pb.PoseBusters()
    # and reads energy_dist.npz + valency JSON — none needed for generation,
    # and energy_dist.npz isn't even shipped for geom_full_kekulized.  The
    # no-op keeps `.sample_analyzer` a harmless attribute; `.analyze()` is
    # never called on our path.
    class _NoopSampleAnalyzer:
        def __init__(self, *args, **kwargs) -> None:
            pass

    flowmol_mod.SampleAnalyzer = _NoopSampleAnalyzer

    # Silence rdkit's chatty valence warnings — sampled mols legitimately
    # include invalid graphs which are filtered by molecule_builder.
    RDLogger.DisableLog("rdApp.*")

    if args.seed is not None:
        seed_everything(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.model_dir / "checkpoints" / "last.ckpt"
    model = FlowMol.load_from_checkpoint(str(ckpt_path)).to(device).eval()

    sample_kwargs = dict(
        device=device,
        n_timesteps=args.n_timesteps,
        stochasticity=args.stochasticity,
        high_confidence_threshold=args.hc_thresh,
    )

    n_batches = math.ceil(args.n_mols / args.max_batch_size)
    molecules: list = []
    t0 = time.time()
    for _ in range(n_batches):
        need = args.n_mols - len(molecules)
        if need <= 0:
            break
        batch_size = min(need, args.max_batch_size)
        if args.n_atoms_per_mol is None:
            batch = model.sample_random_sizes(batch_size, **sample_kwargs)
        else:
            n_atoms = torch.full(
                (batch_size,), args.n_atoms_per_mol,
                dtype=torch.long, device=device,
            )
            batch = model.sample(n_atoms, **sample_kwargs)
        molecules.extend(batch)
    sampling_time = time.time() - t0

    # Write single SDF; upstream `test.py` uses kekulize=False.
    n_written = 0
    invalid = 0
    writer = Chem.SDWriter(str(args.output_file))
    writer.SetKekulize(False)
    try:
        for m in molecules:
            rd = m.rdkit_mol
            if rd is None:
                invalid += 1
                continue
            writer.write(rd)
            n_written += 1
    finally:
        writer.close()

    stats = {
        "n_requested": args.n_mols,
        "n_written": n_written,
        "invalid_count": invalid,
        "sampling_time_seconds": round(sampling_time, 3),
        "n_timesteps": args.n_timesteps,
        "n_atoms_per_mol": args.n_atoms_per_mol,
        "stochasticity": args.stochasticity,
        "hc_thresh": args.hc_thresh,
        "max_batch_size": args.max_batch_size,
        "seed": args.seed,
        "model_dir": str(args.model_dir),
        "device": str(device),
    }
    args.stats_file.write_text(json.dumps(stats, indent=2))

    if n_written == 0:
        print(
            f"ERROR: 0 valid molecules produced from {args.n_mols} samples "
            f"(invalid: {invalid}). Check checkpoint + config compatibility.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Wrote {n_written} / {args.n_mols} valid molecules "
        f"({invalid} invalid) to {args.output_file} "
        f"in {sampling_time:.1f} s.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
