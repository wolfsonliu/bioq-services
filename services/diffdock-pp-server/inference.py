"""DiffDock-PP inference wrapper.

Solves three impedance-mismatch problems between upstream's `src/main_inf.py`
CLI shape and what an HTTP service needs:

  1. **Dataset-layout shim** — upstream only accepts a `--data_file` CSV +
     `--data_path` dir with `{name}_r_b.pdb` / `{name}_l_b.pdb` conventions.
     We build a temp DB5-style layout for a single pair from two arbitrary
     PDB uploads before dispatching in-process.
  2. **Pickle → PDB post-process** — upstream saves a pickle of all N
     samples with confidence; we sort, take top-K, and write each pose as
     a single `dock_pose_<rank>.pdb` (receptor + ligand concatenated).
  3. **Clean errors** — validate all paths before triggering the heavy
     `import torch / e3nn / torch_geometric` cost, so bad params fail fast
     with useful stderr instead of a 30 s import cascade.

Imports come from the vendored upstream at `/opt/diffdock-pp/src/`
(PYTHONPATH); this file lives at `/opt/diffdock-pp/server/inference.py`
inside the image.  See engineering/decisions/2026-07-03-diffdock-pp-server-design.md.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def _bool(s: str) -> bool:
    return s.lower() in ("true", "1", "yes")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="diffdock-pp inference",
        description="Rigid protein-protein docking via DiffDock-PP: "
        "score-model sampling + confidence-model ranking.",
    )
    p.add_argument("--receptor", type=Path, required=True,
                   help="Receptor protein PDB (typically the larger chain).")
    p.add_argument("--ligand", type=Path, required=True,
                   help="Ligand protein PDB (typically the smaller chain). "
                        "Note: 'ligand' = the protein being rotated/translated, "
                        "not a small molecule (EquiDock convention).")
    p.add_argument("--output", type=Path, required=True,
                   help="Output directory; receives dock_pose_<rank>.pdb, "
                        "confidence_scores.json, and raw_samples.pkl.")
    p.add_argument("--num_samples", type=int, default=40,
                   help="Reverse-diffusion samples. Upstream default 40.")
    p.add_argument("--actual_steps", type=int, default=40,
                   help="Denoising steps actually executed (early-stop of "
                        "num_steps).")
    p.add_argument("--top_k", type=int, default=5,
                   help="How many top-ranked poses to write as PDB.")
    p.add_argument("--use_confidence_model", type=_bool, default=True,
                   help="Rank samples via the confidence model. "
                        "false → skip and rank by draw order (~30% faster).")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for reverse diffusion.")
    p.add_argument("--mirror_ligand", type=_bool, default=False,
                   help="Mirror half the samples (diversity trick).")
    p.add_argument("--no_final_noise", type=_bool, default=True,
                   help="Skip noise on the last denoising step (upstream "
                        "default).")
    p.add_argument("--score_model_dir", type=Path, required=True,
                   help="Directory containing model_best_*.pth and args.yaml "
                        "for the score model.")
    p.add_argument("--confidence_model_dir", type=Path, required=True,
                   help="Directory containing model_best_*.pth and args.yaml "
                        "for the confidence model.")
    p.add_argument("--config", type=Path, required=True,
                   help="Path to bundled single_pair_inference.yaml.")
    p.add_argument("--torchhub_dir", type=Path, required=True,
                   help="torch.hub cache dir on NAS. Must contain "
                        "hub/checkpoints/esm2_t33_650M_UR50D.pt and "
                        "hub/facebookresearch_esm_main/.")
    return p.parse_args()


def validate(args: argparse.Namespace) -> None:
    def _exists_pdb(p: Path, label: str) -> None:
        if not p.exists():
            raise SystemExit(f"{label} does not exist: {p}")
        if p.suffix.lower() != ".pdb":
            raise SystemExit(f"{label} must be .pdb (got {p.suffix})")
        if p.stat().st_size == 0:
            raise SystemExit(f"{label} is empty: {p}")

    _exists_pdb(args.receptor, "receptor")
    _exists_pdb(args.ligand, "ligand")

    if not args.score_model_dir.is_dir():
        raise SystemExit(
            f"score_model_dir not found: {args.score_model_dir}. "
            f"Is /data/models/diffdock-pp/ mounted?"
        )
    if not list(args.score_model_dir.glob("model_best_*.pth")):
        raise SystemExit(
            f"No model_best_*.pth found under {args.score_model_dir}"
        )
    if args.use_confidence_model:
        if not args.confidence_model_dir.is_dir():
            raise SystemExit(
                f"confidence_model_dir not found: {args.confidence_model_dir}"
            )
        if not list(args.confidence_model_dir.glob("model_best_*.pth")):
            raise SystemExit(
                f"No model_best_*.pth found under {args.confidence_model_dir}"
            )

    if not args.config.exists():
        raise SystemExit(f"config yaml not found: {args.config}")

    esm_ckpt = args.torchhub_dir / "hub" / "checkpoints" / "esm2_t33_650M_UR50D.pt"
    if not esm_ckpt.exists():
        raise SystemExit(
            f"ESM-2 checkpoint not found: {esm_ckpt}. "
            f"Run fetch_weights.sh or check NAS mount."
        )

    if args.top_k > args.num_samples:
        raise SystemExit(
            f"top_k ({args.top_k}) > num_samples ({args.num_samples})"
        )

    args.output.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# DB5-style layout shim
# ---------------------------------------------------------------------------


PAIR_NAME = "pair"  # sits under data_path/, files pair_r_b.pdb / pair_l_b.pdb


def prepare_dataset_layout(args: argparse.Namespace, workdir: Path) -> tuple[Path, Path]:
    """Copy receptor/ligand into a DB5-style temp dir + write splits CSV.

    Returns (data_file_csv, data_path).
    """
    data_root = workdir / "diffdock_pp_input"
    data_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.receptor, data_root / f"{PAIR_NAME}_r_b.pdb")
    shutil.copy2(args.ligand, data_root / f"{PAIR_NAME}_l_b.pdb")

    splits_csv = data_root / "splits_test.csv"
    splits_csv.write_text(f"path,split\n{PAIR_NAME},test\n")
    return splits_csv, data_root


def build_upstream_argv(
    args: argparse.Namespace,
    csv_path: Path,
    data_root: Path,
    pickle_path: Path,
    save_path: Path,
) -> list[str]:
    """argv that mimics what `src/db5_inference.sh` would compose."""
    argv = [
        "src/main_inf.py",
        "--mode", "test",
        "--config_file", str(args.config),
        "--run_name", "diffdock_pp_job",
        "--save_path", str(save_path),
        "--batch_size", "1",
        "--num_folds", "1",
        "--num_gpu", "1",
        "--gpu", "0",
        "--seed", str(args.seed),
        "--logger", "tensorboard",   # avoid wandb (upstream has dead WANDB_KEY)
        "--filtering_model_path", str(args.confidence_model_dir),
        "--score_model_path", str(args.score_model_dir),
        "--num_samples", str(args.num_samples),
        "--actual_steps", str(args.actual_steps),
        "--prediction_storage", str(pickle_path),
        "--data_file", str(csv_path),
        "--data_path", str(data_root),
        "--torchhub_path", str(args.torchhub_dir),
        "--visualize_n_val_graphs", "0",  # skip per-step PDB dumps
    ]
    if not args.use_confidence_model:
        argv.append("--run_inference_without_confidence_model")
    if args.mirror_ligand:
        argv += ["--mirror_ligand", "True"]
    if args.no_final_noise:
        argv += ["--no_final_noise"]
    return argv


# ---------------------------------------------------------------------------
# Post-processing: pickle → dock_pose_<rank>.pdb + confidence_scores.json
# ---------------------------------------------------------------------------


def _to_pdb_lines(vis, graph, part: str, atom_offset: int, chain_override: str | None):
    """Adapted from upstream `src/sample.py:to_pdb_lines`.

    Copied (rather than imported) so we don't pull in the heavy diffusion
    imports at post-process time. Kept faithful to upstream's exact
    formatting so downstream tools see the same output.
    """
    lines = []
    this = {k: (v.strip() if isinstance(v, str) else v) for k, v in vis[part].items()}
    for i, resname in enumerate(this["resname"]):
        xyz = graph[part].pos[i]
        atom_no = atom_offset + i + 1
        atom_name = this["atom_name"][i]
        chain = chain_override if chain_override else str(this["chain"][i])
        residue = this["residue"][i]
        element = this["element"][i]
        line = (
            f"ATOM  {atom_no:>5} {atom_name:>4} "
            f"{resname} {chain}{residue:>4}    "
            f"{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}"
            f"  1.00  0.00          {element:>2} 0\n"
        )
        lines.append(line)
    return lines


def _load_visualization_values(args, csv_path: Path, data_root: Path) -> dict:
    """Rebuild a minimal BindingDataset to extract visualization_values.

    Cheap: upstream `load_data` for a single pair takes <1 s and doesn't
    touch GPU. We do it after upstream main returns so we get consistent
    per-atom metadata (resname, chain, element) for PDB writing.
    """
    # Deferred: pulled through upstream main already; still cheap.
    from args import parse_args as upstream_parse
    from data import load_data

    saved_argv = sys.argv
    sys.argv = [
        "reload",
        "--mode", "test",
        "--config_file", str(args.config),
        "--data_file", str(csv_path),
        "--data_path", str(data_root),
        "--score_model_path", str(args.score_model_dir),
        "--filtering_model_path", str(args.confidence_model_dir),
        "--save_path", str(args.output / "_reload_ckpt"),
        "--torchhub_path", str(args.torchhub_dir),
        "--batch_size", "1", "--num_gpu", "1", "--gpu", "0", "--seed", "0",
        "--num_folds", "1", "--logger", "tensorboard",
        "--num_samples", str(args.num_samples),
        "--prediction_storage", str(args.output / "_reload.pkl"),
        "--visualize_n_val_graphs", "0",
    ]
    try:
        reload_args = upstream_parse()
        # Sanity-check that yaml drift didn't reintroduce data_file/data_path
        # overrides — those were stripped from single_pair_inference.yaml in
        # 2026-07-06 after v0.0.4 silently ran on upstream's 1A2K sample data
        # instead of the user's uploaded PDBs.
        if str(reload_args.data_file) != str(csv_path):
            raise SystemExit(
                f"reload_args.data_file drift: got {reload_args.data_file!r}, "
                f"expected {str(csv_path)!r} — check bundled yaml for "
                f"stray data_file / data_path keys."
            )
        data = load_data(reload_args)
        if not data.data:
            raise SystemExit("no data entries loaded from load_data")
        # We only ever run 1 pair per job, so take the sole entry regardless
        # of what key upstream uses (DB5Loader keys by `line['path']`, i.e.
        # PAIR_NAME, but we don't rely on that string matching).
        entry = next(iter(data.data.values()))
        return entry["visualization_values"]
    finally:
        sys.argv = saved_argv


def _extract_ranked_samples(pickle_path: Path, use_confidence: bool) -> list[tuple[object, float | None]]:
    """Read upstream pickle and normalize to `[(graph, score_or_None), ...]`.

    - With confidence: pickle is `[[(gt, inf), (sample, conf), ...]]`
      (one outer list per complex; we only run 1 complex per job).
      We drop the ground-truth entry and keep the (already-sorted-desc)
      confidence-ranked samples.
    - Without confidence: pickle is `[data_list, samples_list]` where
      samples_list has N HeteroData objects in draw order.
    """
    with open(pickle_path, "rb") as f:
        results = pickle.load(f)

    if use_confidence:
        # results is `[complex_0_ranked_list]`; per-complex list is
        # `[(gt, inf), (sample_1, conf_1), (sample_2, conf_2), ...]` sorted desc.
        complex_ranked = results[0]
        # Drop leading ground-truth entry (float("inf") sentinel confidence).
        return [(g, float(c)) for g, c in complex_ranked[1:]]

    # No-confidence path: results = [data_list_input, samples_list]
    # The last element is the samples_list from the final iteration.
    samples_list = results[-1]
    return [(s, None) for s in samples_list]


def _postprocess(args: argparse.Namespace, pickle_path: Path,
                 csv_path: Path, data_root: Path) -> None:
    ranked = _extract_ranked_samples(pickle_path, args.use_confidence_model)
    if not ranked:
        print("[diffdock-pp] ERROR: no samples in pickle", file=sys.stderr)
        sys.exit(1)

    vis = _load_visualization_values(args, csv_path, data_root)

    scores = []
    top_k = min(args.top_k, len(ranked))
    for rank in range(1, top_k + 1):
        graph, conf = ranked[rank - 1]
        out_pdb = args.output / f"dock_pose_{rank}.pdb"
        lines = []
        # Receptor first (chain R), then ligand (chain L) — clean distinct chains.
        lines += _to_pdb_lines(vis, graph, "receptor", atom_offset=0,
                               chain_override="R")
        lines.append("TER\n")
        rec_atoms = len(vis["receptor"]["resname"])
        lines += _to_pdb_lines(vis, graph, "ligand", atom_offset=rec_atoms,
                               chain_override="L")
        lines.append("END\n")
        out_pdb.write_text("".join(lines))
        scores.append({
            "rank": rank,
            "confidence": conf,
            "sample_file": out_pdb.name,
            "note": (
                "ranked by confidence model" if args.use_confidence_model
                else "unranked (use_confidence_model=false); order = sample draw order"
            ),
        })

    (args.output / "confidence_scores.json").write_text(
        json.dumps(scores, indent=2)
    )
    print(f"[diffdock-pp] wrote {top_k} top poses + confidence_scores.json "
          f"to {args.output}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    validate(args)

    # `TORCH_HOME` + `HF_HOME` — belt-and-suspenders for any auxiliary model
    # (ESM-2 is the main consumer; some downstream libs also honor these).
    os.environ["TORCH_HOME"] = str(args.torchhub_dir)
    os.environ["HF_HOME"] = str(args.torchhub_dir / "hf_cache")

    # Prepare temp DB5-style layout under <job_dir>/.
    workdir = args.output.parent
    csv_path, data_root = prepare_dataset_layout(args, workdir)
    pickle_path = args.output / "raw_samples.pkl"
    save_path = args.output / "_ckpt"
    save_path.mkdir(parents=True, exist_ok=True)

    # Upstream has two unconditional imports of deps we deliberately don't
    # install (both are dead — the imported names are never actually
    # referenced in the inference path):
    #   - src/main_inf.py:13: `import wandb` (all wandb.* calls gated by
    #     `if args.wandb_sweep:`, permanently False via config yaml)
    #   - src/evaluation/compute_rmsd.py:10: `from matplotlib import pyplot as plt`
    #     (`plt` is never referenced in the file — dead import)
    # We inject empty stub modules so top-level imports succeed without
    # installing the real packages (~50 MB wandb + ~80 MB matplotlib saved).
    # Any accidental attribute access would raise AttributeError — the
    # intended fail-loud behavior if either dead branch ever gets revived.
    #
    # `from matplotlib import pyplot as plt` requires matplotlib.pyplot to
    # be reachable as an attribute of the parent matplotlib module (Python
    # does `getattr(matplotlib, 'pyplot')` after ensuring both are in
    # sys.modules), so we wire the attribute explicitly.
    import types as _types
    _wandb = sys.modules.setdefault("wandb", _types.ModuleType("wandb"))
    _mpl = sys.modules.setdefault("matplotlib", _types.ModuleType("matplotlib"))
    _plt = sys.modules.setdefault("matplotlib.pyplot",
                                  _types.ModuleType("matplotlib.pyplot"))
    _mpl.pyplot = _plt  # noqa: SLF001

    # Heavy imports gated past validate() so bad params get clean errors.
    from args import parse_args as upstream_parse
    from main_inf import main as upstream_main

    saved_argv = sys.argv
    sys.argv = build_upstream_argv(args, csv_path, data_root, pickle_path,
                                   save_path)
    try:
        upstream_args = upstream_parse()
        # Guard against yaml overriding CLI data_file / data_path — see
        # inference.py:_load_visualization_values for the incident that
        # motivated this check.
        if str(upstream_args.data_file) != str(csv_path):
            raise SystemExit(
                f"upstream_args.data_file drift: got "
                f"{upstream_args.data_file!r}, expected {str(csv_path)!r} "
                f"— bundled yaml is overriding CLI. Check "
                f"single_pair_inference.yaml for stray data_file / data_path."
            )
        if str(upstream_args.data_path) != str(data_root):
            raise SystemExit(
                f"upstream_args.data_path drift: got "
                f"{upstream_args.data_path!r}, expected {str(data_root)!r}"
            )
        upstream_main(upstream_args)
    finally:
        sys.argv = saved_argv

    if not pickle_path.exists():
        print(
            f"[diffdock-pp] ERROR: upstream main_inf did not write "
            f"{pickle_path}",
            file=sys.stderr,
        )
        return 1

    _postprocess(args, pickle_path, csv_path, data_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
