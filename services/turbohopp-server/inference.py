"""TurboHopp inference wrapper.

Re-implements the inference flow of upstream's ``evaluate_consistency.py``
for the **single (protein_pocket, reference_ligand)** case:

  * Upstream's CLI only supports dataset-mode inference over the pre-
    registered ``pdbbind_filtered`` / ``crossdocked`` datasets.  Our
    wrapper builds an in-memory single-sample ``ProteinLigandComplex`` +
    featurization pipeline (identical to DiffHopp's single-input path)
    and runs the consistency-model sampler directly.
  * Adds explicit ``--checkpoint``, ``--num_sampling_steps``, ``--find_best``,
    ``--seed`` flags — upstream reads all of these from a global config.
  * Surfaces clean errors to stderr so ``JobInfo.error_tail`` is
    informative when the FC job fails.

Imports come from the vendored TurboHopp source at ``/opt/turbohopp/``
(installed via ``pip install -e`` in the Dockerfile).  This file lives at
``/opt/turbohopp-server/server/inference.py`` inside the image.

Contract with server code (``server/tools.py``):

    python -m server.inference \
        --input_protein <pocket.pdb> \
        --input_molecule <ref_ligand.sdf|mol2|pdb> \
        --output <dir> \
        --checkpoint <path/to/consistency_model.ckpt> \
        --num_samples N \
        --num_sampling_steps K \
        [--find_best] \
        [--seed S]

Output: ``<dir>/output_<i>.sdf`` for i in 0..N-1 (invalid mols skipped).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="turbohopp inference",
        description=(
            "Consistency-model accelerated scaffold hopping conditioned on a "
            "protein pocket + reference ligand."
        ),
    )
    p.add_argument("--input_protein", type=Path, required=True,
                   help="Protein pocket (.pdb).")
    p.add_argument("--input_molecule", type=Path, required=True,
                   help="Reference ligand (.sdf / .mol2 / .pdb).")
    p.add_argument("--output", type=Path, required=True,
                   help="Output directory; one output_<i>.sdf per sample.")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Absolute path to a TurboHopp consistency-model *.ckpt.")
    p.add_argument("--num_samples", type=int, default=10,
                   help="Number of scaffold candidates (1-100). Default 10.")
    p.add_argument("--num_sampling_steps", type=int, default=40,
                   help="Consistency-model sampling steps (1-100). Default 40.")
    p.add_argument("--find_best", action="store_true",
                   help="Post-hoc rescoring: pick QED+SA-composite argmax "
                        "over the last num_sampling_steps candidates per graph.")
    p.add_argument("--seed", type=int, default=None,
                   help="Sampling RNG seed. None → torch default (non-deterministic).")
    return p.parse_args()


def validate(args: argparse.Namespace) -> None:
    if not args.input_molecule.exists():
        raise SystemExit(f"input_molecule does not exist: {args.input_molecule}")
    if args.input_molecule.suffix.lower() not in (".sdf", ".mol2", ".pdb"):
        raise SystemExit(
            f"input_molecule must be .sdf / .mol2 / .pdb "
            f"(got {args.input_molecule.suffix})"
        )
    if not args.input_protein.exists():
        raise SystemExit(f"input_protein does not exist: {args.input_protein}")
    if args.input_protein.suffix.lower() != ".pdb":
        raise SystemExit(
            f"input_protein must be .pdb (got {args.input_protein.suffix})"
        )
    if not args.checkpoint.exists():
        raise SystemExit(
            f"checkpoint not found: {args.checkpoint}.  "
            f"Is /data/models/turbohopp/checkpoints/v1/ mounted, and does it "
            f"contain the consistency-model .ckpt?  Upstream does not publish "
            f"public checkpoints — see services/turbohopp-server/README.md "
            f"## Weights."
        )
    if not 1 <= args.num_samples <= 100:
        raise SystemExit(
            f"--num_samples out of range (1-100), got {args.num_samples}"
        )
    if not 1 <= args.num_sampling_steps <= 100:
        raise SystemExit(
            f"--num_sampling_steps out of range (1-100), got {args.num_sampling_steps}"
        )
    args.output.mkdir(parents=True, exist_ok=True)


def _disable_wandb() -> None:
    """Upstream imports call ``wandb.init`` at module load in some code paths.

    Set ``WANDB_MODE=disabled`` and ``WANDB_DISABLED=true`` before any
    upstream import so we never block on a wandb login check in FC.
    """
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("WANDB_SILENT", "true")


def main() -> int:
    args = parse_args()
    validate(args)
    _disable_wandb()

    # Imports gated past validation so missing weights / bad paths produce
    # clean error messages instead of obscure import-time failures.
    import torch
    from rdkit import Chem
    from torch_geometric.data import Batch
    from torch_geometric.transforms import Compose

    from diffusion_hopping.analysis.build import MoleculeBuilder
    from diffusion_hopping.data import (
        Ligand,
        Protein,
        ProteinLigandComplex,
    )
    from diffusion_hopping.data.featurization import (
        ProteinLigandSimpleFeaturization,
    )
    from diffusion_hopping.data.transform import (
        ObabelTransform,
        ReduceTransform,
    )

    # Upstream module names — imported as bare (top-level) modules because
    # upstream doesn't package them.  Dockerfile puts /opt/turbohopp on
    # sys.path via `pip install -e /opt/turbohopp`.
    from consistency.models_consistency import (  # type: ignore[import-not-found]
        ConsistencySamplingAndEditing_DiffHopp,
        karras_schedule,
    )
    from utils._util_consistency import get_consistency_models  # type: ignore[import-not-found]

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[turbohopp] device={device} checkpoint={args.checkpoint} "
        f"num_samples={args.num_samples} num_sampling_steps={args.num_sampling_steps} "
        f"find_best={args.find_best} seed={args.seed}",
        flush=True,
    )

    # -- Model ------------------------------------------------------------
    # ``get_consistency_models`` instantiates student + ema + teacher with
    # the fixed GVP architecture used in the paper.  We only use the student
    # for inference; teacher stays on CPU / never runs a forward pass.
    student_model, _ema, teacher_model = get_consistency_models(
        T=args.num_sampling_steps,
        architecture=None,  # keep default (Architecture.GVP)
    ) if False else get_consistency_models(T=args.num_sampling_steps)

    # LitConsistencyModel wraps student + teacher + a sampling schedule.
    # The saved state_dict follows the LightningModule convention with keys
    # prefixed by "student_model." / "teacher_model." — load via the plain
    # torch.load path so we don't need to construct a full LitConsistencyModel
    # around it (upstream's evaluate_consistency.py does this too).
    checkpoint = torch.load(str(args.checkpoint), map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Strip the "student_model." prefix and load into student_model directly.
    student_state = {
        k[len("student_model."):]: v
        for k, v in state_dict.items()
        if k.startswith("student_model.")
    }
    if not student_state:
        # No prefix means the ckpt is already a bare student state_dict.
        student_state = state_dict
    missing, unexpected = student_model.load_state_dict(student_state, strict=False)
    if missing:
        print(f"[turbohopp] WARN: missing keys in student_model: "
              f"{len(missing)} (first 3: {missing[:3]})", flush=True)
    if unexpected:
        print(f"[turbohopp] WARN: unexpected keys: "
              f"{len(unexpected)} (first 3: {unexpected[:3]})", flush=True)

    student_model.to(device)
    student_model.eval()

    # -- Input featurization ----------------------------------------------
    # Format normalization: ligand → .sdf, protein → .pdb + reduce (Hs).
    # Reuse DiffHopp's data pipeline directly — TurboHopp shares this layer.
    ligand_transform_sdf = ObabelTransform(
        from_format=args.input_molecule.suffix[1:].lower(),
        to_format="sdf",
    )
    protein_transform = Compose([ObabelTransform(), ReduceTransform()])

    protein = Protein(protein_transform(args.input_protein))
    ligand = Ligand(ligand_transform_sdf(args.input_molecule))
    pl_complex = ProteinLigandComplex(protein, ligand, identifier="query")

    featurization = ProteinLigandSimpleFeaturization(
        c_alpha_only=True, cutoff=8.0, mode="residue",
    )
    batch = Batch.from_data_list(
        [featurization(pl_complex)] * args.num_samples
    ).to(device)

    # -- Sampling ---------------------------------------------------------
    print(f"[turbohopp] sampling {args.num_samples} candidates × "
          f"{args.num_sampling_steps} steps...", flush=True)

    sampler = ConsistencySamplingAndEditing_DiffHopp(
        final_timesteps=args.num_sampling_steps,
    )
    sigmas_tensor = karras_schedule(
        args.num_sampling_steps,
        sigma_min=0.002, sigma_max=80.0, rho=7.0,
        device=torch.device(device),
    )
    # Upstream evaluate_consistency.py reverses to a decreasing schedule.
    sigmas = list(reversed(sigmas_tensor.tolist()))

    # Sampler returns List[HeteroData batches] — one per timestep.
    # NOTE: the sampler reads ``model.consistency_model`` for helpers
    # (get_mask, centered_complex, normalize, ...).  ``student_model`` on
    # its own doesn't have that attribute.  We attach it inline: the
    # student itself IS the consistency model.
    if not hasattr(student_model, "consistency_model"):
        student_model.consistency_model = student_model  # noqa: SLF001
    sample_results = sampler(student_model, batch, sigmas)

    # -- Molecule building ------------------------------------------------
    builder = MoleculeBuilder(include_invalid=True)
    final_mols: list = builder(sample_results[-1])

    if args.find_best:
        # Build per-timestep mols for the last num_sampling_steps steps,
        # then apply the paper's QED + normalized-SA composite argmax.
        # Upstream's ``postprocess_molecules`` in ConsistencySamplingAnd-
        # Editing_DiffHopp assumes its input is already List[List[Mol]],
        # not List[Batch] — we do the MoleculeBuilder step here.
        per_step_mols = [builder(step_batch) for step_batch in sample_results]
        final_mols = _pick_best_by_qed_sa(per_step_mols, args.num_sampling_steps)

    # -- Persist ----------------------------------------------------------
    written = 0
    for i, mol in enumerate(final_mols):
        if mol is None:
            continue
        out_path = args.output / f"output_{i}.sdf"
        Chem.MolToMolFile(mol, str(out_path))
        written += 1

    print(f"[turbohopp] wrote {written}/{args.num_samples} valid molecules "
          f"to {args.output}", flush=True)
    if written == 0:
        print("[turbohopp] ERROR: no valid molecules generated "
              "(MoleculeBuilder filtered all outputs; try higher "
              "num_sampling_steps or a different reference ligand)",
              file=sys.stderr)
        return 1
    return 0


def _pick_best_by_qed_sa(
    per_step_mols: list,
    look_back: int,
) -> list:
    """Paper's ``ConsistencySamplingAndEditing_DiffHopp.postprocess_molecules``,
    fixed to accept List[List[Mol]] (upstream's version fails on List[Batch])."""
    from rdkit import Chem
    from rdkit.Chem import QED

    try:
        # sascorer ships in TurboHopp upstream at repo root.
        import sascorer  # type: ignore[import-not-found]
    except ImportError:
        # If sascorer unavailable, fall back to last-step samples so
        # find_best gracefully degrades rather than crashing.
        print("[turbohopp] WARN: sascorer not importable; --find_best "
              "degraded to last-step selection", flush=True)
        return per_step_mols[-1]

    n_graphs = len(per_step_mols[-1])
    best_mols: list = []
    for i in range(n_graphs):
        candidates = [step_mols[i] for step_mols in per_step_mols[-look_back:]]
        best_score, best = -1.0, None
        for mol in candidates:
            if mol is None:
                continue
            try:
                smiles = Chem.MolToSmiles(mol)
            except Exception:
                continue
            if "." in smiles:
                continue  # multi-fragment
            try:
                qed_v = float(QED.qed(mol))
                sa_v = float(sascorer.calculateScore(mol))
            except Exception:
                continue
            sa_norm = (10.0 - sa_v) / 9.0
            score = 0.5 * (qed_v + sa_norm)
            if score > best_score:
                best_score = score
                best = mol
        if best is not None:
            best.SetProp("QED", str(QED.qed(best)))
            best.SetProp("SA_Score", str(sascorer.calculateScore(best)))
            best.SetProp("Unified_Score", f"{best_score:.4f}")
            best_mols.append(best)
        else:
            # Fall back to last-step (may still be None → skipped downstream).
            best_mols.append(candidates[-1])
    return best_mols


if __name__ == "__main__":
    sys.exit(main())
