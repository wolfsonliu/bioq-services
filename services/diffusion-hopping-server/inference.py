"""DiffHopp inference wrapper.

Re-implements the inference flow of upstream's `generate_scaffolds.py` so
we can:

  1. accept `--checkpoint <path>` and `--variant <name>` flags (upstream
     hardcodes `Path("checkpoints") / "gvp_conditional.ckpt"`)
  2. pass an explicit output directory without depending on CWD
  3. surface clean errors to subprocess.stderr so JobInfo.error_tail is
     informative

Imports come from the vendored upstream package at /opt/diffusion-hopping/
(installed via `pip install -e` in the Dockerfile).  This file lives at
/opt/diffusion-hopping/server/inference.py inside the image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="diffusion-hopping inference",
        description="Generate scaffold-hopped molecules conditioned on a "
        "protein pocket + reference ligand.",
    )
    p.add_argument("--input_molecule", type=Path, required=True,
                   help="Reference ligand (.sdf / .mol2 / .pdb).")
    p.add_argument("--input_protein", type=Path, required=True,
                   help="Protein pocket (.pdb).")
    p.add_argument("--output", type=Path, required=True,
                   help="Output directory; one output_<i>.sdf per sample.")
    p.add_argument("--num_samples", type=int, default=10,
                   help="Number of scaffold candidates (1–100). Default 10.")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Absolute path to a *.ckpt file.")
    p.add_argument(
        "--variant",
        choices=["gvp_conditional", "gvp_unconditional",
                 "egnn_conditional", "egnn_unconditional"],
        default="gvp_conditional",
        help="Model variant (used only for logging; the actual graph "
             "comes from --checkpoint).",
    )
    return p.parse_args()


def validate(args: argparse.Namespace) -> None:
    if not args.input_molecule.exists():
        raise SystemExit(f"input_molecule does not exist: {args.input_molecule}")
    if args.input_molecule.suffix.lower() not in (".sdf", ".mol2", ".pdb"):
        raise SystemExit(
            f"input_molecule must be .sdf / .mol2 / .pdb (got {args.input_molecule.suffix})"
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
            f"Is /data/models/diffusion-hopping/checkpoints/ mounted?"
        )
    if args.num_samples < 1 or args.num_samples > 100:
        raise SystemExit(
            f"--num_samples out of range (1–100), got {args.num_samples}"
        )
    args.output.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    validate(args)

    # Imports gated past validation so missing weights / bad paths produce
    # clean error messages instead of obscure import-time failures.
    import torch
    from rdkit import Chem
    from torch_geometric.data import Batch

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
    from diffusion_hopping.model import DiffusionHoppingModel
    from torch_geometric.transforms import Compose

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[diffhopp] device={device} variant={args.variant} "
          f"checkpoint={args.checkpoint} num_samples={args.num_samples}",
          flush=True)

    model = DiffusionHoppingModel.load_from_checkpoint(
        str(args.checkpoint), map_location=device,
    ).to(device)
    model.eval()
    model.freeze()

    # Format normalization: ligand → .sdf, protein → .pdb + reduce (Hs).
    ligand_transform_sdf = ObabelTransform(
        from_format=args.input_molecule.suffix[1:].lower(), to_format="sdf",
    )
    protein_transform = Compose([ObabelTransform(), ReduceTransform()])

    protein = Protein(protein_transform(args.input_protein))
    ligand = Ligand(ligand_transform_sdf(args.input_molecule))
    pl_complex = ProteinLigandComplex(protein, ligand, identifier="complex")

    featurization = ProteinLigandSimpleFeaturization(
        c_alpha_only=True, cutoff=8.0, mode="residue",
    )
    batch = Batch.from_data_list(
        [featurization(pl_complex)] * args.num_samples
    ).to(model.device)

    print("[diffhopp] sampling...", flush=True)
    sample_results = model.model.sample(batch)
    final_output = sample_results[-1]

    builder = MoleculeBuilder(include_invalid=False)
    molecules: list[Chem.Mol] = builder(final_output)

    written = 0
    for i, mol in enumerate(molecules):
        if mol is None:
            continue
        out_path = args.output / f"output_{i}.sdf"
        Chem.MolToMolFile(mol, str(out_path))
        written += 1

    print(f"[diffhopp] wrote {written}/{args.num_samples} valid molecules "
          f"to {args.output}", flush=True)
    if written == 0:
        # MoleculeBuilder filtered everything → still a real error from a
        # user-visible perspective; signal failure so JobInfo.status reflects it.
        print("[diffhopp] ERROR: no valid molecules generated", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
