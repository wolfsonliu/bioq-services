"""megalodon-server inference wrapper.

Runs Megalodon unconditional 3D molecule generation. Unlike the upstream
`scripts/sample.py` (which bundles generation + metrics in
`MoleculeEvaluationCallback.evaluate_molecules`, hardcoding
`model.sample(current, timesteps=...)` with no `num_atoms` hook and only
printing metrics), this wrapper:

  1. loads the per-job config (statistics paths already repointed at NAS by
     server.configs.build_config) + the checkpoint,
  2. runs its own sampling loop so `--n-atoms-per-mol` can fix molecule size,
  3. writes generated_molecules.sdf,
  4. reuses the upstream metric components (via a MoleculeEvaluationCallback
     built the same way sample.py builds it) to dump metrics.json,
  5. dumps generation_stats.json.

Metric computation is defensive: a failing metric never discards the SDF.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Megalodon generation wrapper")
    p.add_argument("--config-path", type=Path, required=True)
    p.add_argument("--ckpt-path", type=Path, required=True)
    p.add_argument("--stats-dir", type=Path, required=True)
    p.add_argument("--stats-split", default="train")
    p.add_argument("--save-dir", type=Path, required=True)
    p.add_argument("--n-molecules", type=int, default=100)
    p.add_argument("--n-atoms-per-mol", type=int, default=None)
    p.add_argument("--timesteps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _load_model(ckpt_path: Path, cfg):
    import omegaconf
    import torch

    from megalodon.models.module import Graph3DInterpolantModel

    torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])
    model = Graph3DInterpolantModel.load_from_checkpoint(
        str(ckpt_path),
        interpolant_params=cfg.interpolant,
        sampling_params=cfg.sample,
    )
    return model


def _generate(model, n_molecules, batch_size, timesteps, n_atoms_per_mol, scale_coords):
    import torch

    from megalodon.metrics.molecule import get_molecules
    from megalodon.metrics.molecule_evaluation_callback import full_atom_decoder

    mols = []
    with torch.no_grad():
        while len(mols) < n_molecules:
            cur = min(n_molecules - len(mols), batch_size)
            num_atoms = None
            if n_atoms_per_mol is not None:
                num_atoms = torch.full((cur,), n_atoms_per_mol, dtype=torch.long)
            generated = model.sample(cur, timesteps=timesteps, num_atoms=num_atoms)
            generated["x"] = scale_coords * generated["x"]
            mols.extend(get_molecules(generated, {"atom_decoder": full_atom_decoder}))
    return mols


def _write_sdf(mols, sdf_path: Path) -> int:
    from rdkit import Chem

    n_valid = 0
    with open(sdf_path, "w") as fh:
        for m in mols:
            rk = getattr(m, "raw_rdkit_mol", None)
            if rk is None:
                continue
            try:
                fh.write(Chem.MolToMolBlock(rk, kekulize=False) + "\n$$$$\n")
                n_valid += 1
            except Exception as exc:  # noqa: BLE001
                print(f"WARN: skipping unwritable molecule: {exc}", file=sys.stderr)
    return n_valid


def _compute_metrics(mols, callback) -> dict:
    """Replicate MoleculeEvaluationCallback.evaluate_molecules' metric block
    on an already-generated `mols` list. Reuses the callback's dataset_info /
    statistics / flags so wiring matches upstream exactly."""
    from megalodon.metrics.molecule_metrics_2d import Molecule2DMetrics
    from megalodon.metrics.molecule_metrics_3d import Molecule3DMetrics
    from megalodon.metrics.molecule_novelty_similarity import MoleculeTrainDataMetrics
    from megalodon.metrics.molecule_stability_2d import Molecule2DStability

    device = "cpu"
    results: dict = {}

    stab = Molecule2DStability(callback.dataset_info, device=device)
    stability_res, valid_smiles, _valid, _stable = stab(mols)
    results.update(stability_res)

    if callback.compute_2D_metrics:
        results.update(Molecule2DMetrics(callback.dataset_info, device=device).evaluate(valid_smiles))

    if callback.compute_3D_metrics:
        m3d = Molecule3DMetrics(
            callback.dataset_info, device=device, preserve_aromatic=callback.preserve_aromatic
        )
        results.update(m3d([m.rdkit_mol for m in mols]))

    if callback.compute_train_data_metrics and callback.train_smiles is not None:
        tdm = MoleculeTrainDataMetrics(callback.train_smiles, device=device)
        results.update(tdm(valid_smiles))

    return {k: (float(v) if hasattr(v, "__float__") else v) for k, v in results.items()}


def main() -> int:
    a = parse_args()
    a.save_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from omegaconf import OmegaConf

    from megalodon.data.statistics import Statistics
    from megalodon.metrics.molecule_evaluation_callback import MoleculeEvaluationCallback

    if a.seed is not None:
        torch.manual_seed(a.seed)

    cfg = OmegaConf.load(a.config_path)

    model = _load_model(a.ckpt_path, cfg)
    model.to(a.device)
    model.eval()

    # Statistics bundle (metrics + novelty reference). Loaded directly from the
    # flat NAS stats dir (decoupled from cfg.data.dataset_root).
    try:
        statistics = Statistics.load_statistics(
            statistics_dir=str(a.stats_dir), split_name=a.stats_split
        )
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: statistics load failed ({exc}); metrics degraded", file=sys.stderr)
        statistics = None

    scale_coords = float(OmegaConf.select(cfg, "evaluation.scale_coords", default=1.0) or 1.0)

    t0 = time.time()
    mols = _generate(
        model,
        n_molecules=a.n_molecules,
        batch_size=a.batch_size,
        timesteps=a.timesteps,
        n_atoms_per_mol=a.n_atoms_per_mol,
        scale_coords=scale_coords,
    )
    sampling_time = time.time() - t0

    sdf_path = a.save_dir / "generated_molecules.sdf"
    n_valid = _write_sdf(mols, sdf_path)
    if n_valid == 0:
        print("ERROR: 0 valid molecules produced", file=sys.stderr)
        return 1

    # Metrics — reuse the upstream callback's construction (statistics ->
    # train_smiles, dataset_info, compute flags from cfg.evaluation).
    metrics: dict = {}
    try:
        callback = MoleculeEvaluationCallback(
            n_graphs=a.n_molecules,
            batch_size=a.batch_size,
            timesteps=a.timesteps,
            statistics=statistics,
            compute_2D_metrics=bool(OmegaConf.select(cfg, "evaluation.compute_2D_metrics", default=True)),
            compute_3D_metrics=bool(OmegaConf.select(cfg, "evaluation.compute_3D_metrics", default=True)),
            compute_train_data_metrics=bool(
                OmegaConf.select(cfg, "evaluation.compute_train_data_metrics", default=True)
            ),
            compute_energy_metrics=False,
            scale_coords=scale_coords,
            preserve_aromatic=bool(OmegaConf.select(cfg, "evaluation.preserve_aromatic", default=True)),
        )
        metrics = _compute_metrics(mols, callback)
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: metric computation failed: {exc}", file=sys.stderr)
        metrics = {"error": str(exc)}

    (a.save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (a.save_dir / "generation_stats.json").write_text(json.dumps({
        "n_requested": a.n_molecules,
        "n_valid": n_valid,
        "n_atoms_per_mol": a.n_atoms_per_mol,
        "timesteps": a.timesteps,
        "seed": a.seed,
        "sampling_time_seconds": round(sampling_time, 2),
    }, indent=2))

    print(f"✓ {n_valid}/{a.n_molecules} valid molecules -> {sdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
