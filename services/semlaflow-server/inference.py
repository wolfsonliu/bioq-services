"""Thin wrapper around `semlaflow` for the semlaflow-server service.

Why not just call upstream `python -m semlaflow.predict`?

  1. **Capture metrics to a file.** `predict.py:main()` computes the full
     generative metrics table (validity / uniqueness / novelty / energy /
     strain / ...) but only `print`s it — it never writes it to disk.  A
     service consumer needs structured output, so this wrapper reuses the
     upstream module-level functions and dumps the results dict to
     `metrics.json` (avoids brittle stdout-table parsing).
  2. **Route output to the job dir.** We pass an explicit --save-dir under
     `<job_dir>/output/` and also emit `generation_stats.json`.

Everything heavy is reused verbatim from upstream (load_model /
dm_from_ckpt / save_rdkit_sdf / save_raw_smol / init_metrics /
generate_molecules / calc_metrics_) via an argparse.Namespace built to match
the upstream function signatures.

NOTE: `semlaflow.scriptutil.generate_molecules` hardcodes `model.to("cuda")`
+ `batch.cuda()` — there is no CPU fallback; this wrapper requires a GPU.

Outputs (under --save-dir):
  predictions.smol.sdf      — up to n_molecules valid 3D mols (RDKit SDF)
  predictions.smol          — upstream internal GeometricMolBatch bytes
  metrics.json              — generative metrics (tensor values -> float)
  generation_stats.json     — {model_name, n_requested, n_valid, ...}

Exit codes:
  0 = at least one valid mol written
  1 = zero valid mols (adapter treats as FAILED)
  2 = pre-flight validation failed (missing ckpt / dataset split)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from argparse import Namespace
from pathlib import Path

# `semlaflow.train` imports wandb, but the predict path does not.  Stub it
# defensively (zero cost) in case a transitive import pulls it in.
sys.modules.setdefault("wandb", types.ModuleType("wandb"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SemlaFlow unconditional 3D small-molecule generation"
    )
    p.add_argument("--ckpt-path", type=Path, required=True,
                   help="Path to the model checkpoint (.ckpt).")
    p.add_argument("--data-path", type=Path, required=True,
                   help="Directory containing {train,val,test}.smol reference "
                        "splits.")
    p.add_argument("--dataset", required=True, choices=["qm9", "geom-drugs"],
                   help="Dataset kind (fixes coord_std + bucket_limits).")
    p.add_argument("--save-dir", type=Path, required=True,
                   help="Output directory for SDF + metrics.")
    p.add_argument("--n-molecules", type=int, default=100)
    p.add_argument("--integration-steps", type=int, default=100)
    p.add_argument("--dataset-split", default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--ode-sampling-strategy", default="log",
                   choices=["log", "linear"])
    p.add_argument("--cat-sampling-noise-level", type=int, default=1)
    p.add_argument("--batch-cost", type=int, default=8192)
    p.add_argument("--bucket-cost-scale", default="linear")
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()


def validate(args: argparse.Namespace) -> None:
    """Fail early (rc=2) on missing weights / dataset before heavy imports."""
    required = [
        args.ckpt_path,
        args.data_path / f"{args.dataset_split}.smol",
        args.data_path / "train.smol",  # novelty reference (always loaded)
    ]
    for path in required:
        if not path.exists():
            print(f"ERROR: required file not found: {path}", file=sys.stderr)
            sys.exit(2)
    args.save_dir.mkdir(parents=True, exist_ok=True)


def _to_float(value) -> float:
    """Coerce a metric value (torch scalar tensor / numpy / float) to float."""
    try:
        item = value.item()  # torch / numpy scalar
    except AttributeError:
        item = value
    return float(item)


def main() -> int:
    a = parse_args()
    validate(a)

    # Heavy imports deferred until after validation.
    import lightning as L
    import semlaflow.scriptutil as util
    from semlaflow.predict import (
        dm_from_ckpt,
        load_model,
        save_raw_smol,
        save_rdkit_sdf,
    )

    L.seed_everything(a.seed)
    util.disable_lib_stdout()
    util.configure_fs()

    # Upstream functions take an argparse.Namespace with these exact attrs.
    args = Namespace(
        ckpt_path=str(a.ckpt_path),
        data_path=str(a.data_path),
        dataset=a.dataset,
        save_dir=str(a.save_dir),
        save_file="predictions.smol",
        n_molecules=a.n_molecules,
        integration_steps=a.integration_steps,
        dataset_split=a.dataset_split,
        ode_sampling_strategy=a.ode_sampling_strategy,
        cat_sampling_noise_level=a.cat_sampling_noise_level,
        batch_cost=a.batch_cost,
        bucket_cost_scale=a.bucket_cost_scale,
        n_layers=None,  # only used by the egnn architecture branch
    )

    vocab = util.build_vocab()
    dm = dm_from_ckpt(args, vocab)
    model = load_model(args, vocab)

    # Builds the novelty reference set from every train SMILES — the main
    # cost for geom-drugs (~300k RDKit mols).
    metrics, _stability = util.init_metrics(str(a.data_path), model)

    t0 = time.time()
    molecules, raw_outputs = util.generate_molecules(
        model, dm, a.integration_steps, a.ode_sampling_strategy
    )
    sampling_time = time.time() - t0

    save_rdkit_sdf(args, molecules)
    save_raw_smol(args, raw_outputs, model)

    n_valid = sum(1 for m in molecules if m is not None)
    if n_valid == 0:
        print(
            f"ERROR: 0 valid molecules produced from {a.n_molecules} samples. "
            "Check checkpoint + dataset compatibility.",
            file=sys.stderr,
        )
        return 1

    results = util.calc_metrics_(molecules, metrics)
    metrics_out = {k: _to_float(v) for k, v in results.items()}
    (a.save_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2))

    (a.save_dir / "generation_stats.json").write_text(json.dumps({
        "model_name": a.dataset,
        "dataset": a.dataset,
        "n_requested": a.n_molecules,
        "n_valid": n_valid,
        "integration_steps": a.integration_steps,
        "dataset_split": a.dataset_split,
        "ode_sampling_strategy": a.ode_sampling_strategy,
        "sampling_time_seconds": round(sampling_time, 3),
        "seed": a.seed,
    }, indent=2))

    print(
        f"Wrote {n_valid} / {a.n_molecules} valid molecules to "
        f"{a.save_dir / 'predictions.smol.sdf'} in {sampling_time:.1f} s."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
