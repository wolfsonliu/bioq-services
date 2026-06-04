"""Standalone ESMFold2 inference script.

Loads the model, reads input JSON, runs structure prediction, and writes
mmCIF output + metrics JSON. Called as a subprocess by the service framework.

Usage:
    python inference.py \
        --input-json input.json \
        --output-dir output/ \
        --model-dir /opt/esmfold2/weights \
        --ccd-path /opt/esmfold2/weights/ccd.pkl \
        --num-loops 3 \
        --num-sampling-steps 50
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ESMFold2 inference")
    p.add_argument("--input-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=str, required=True)
    p.add_argument("--ccd-path", type=Path, default=None)
    p.add_argument("--num-loops", type=int, default=3)
    p.add_argument("--num-sampling-steps", type=int, default=50)
    p.add_argument("--num-diffusion-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--noise-scale", type=float, default=None)
    p.add_argument("--step-scale", type=float, default=None)
    return p.parse_args()


def build_structure_prediction_input(doc: dict, msa_class=None):
    """Convert JSON document to ESMFold2 StructurePredictionInput."""
    from esm.models.esmfold2 import (
        DNAInput,
        LigandInput,
        Modification,
        ProteinInput,
        StructurePredictionInput,
    )
    try:
        from esm.models.esmfold2 import RNAInput
    except ImportError:
        from esm.models.esmfold2.types import RNAInput

    sequences = []
    for entry in doc["sequences"]:
        entry_type = entry["type"]
        entry_id = entry["id"]

        if entry_type == "protein":
            mods = None
            if entry.get("modifications"):
                mods = [
                    Modification(position=m["position"], ccd=m["ccd"])
                    for m in entry["modifications"]
                ]
            msa = None
            if entry.get("msa_path") and msa_class is not None:
                msa = msa_class.from_a3m(entry["msa_path"])
            sequences.append(
                ProteinInput(
                    id=entry_id,
                    sequence=entry["sequence"],
                    modifications=mods,
                    msa=msa,
                )
            )
        elif entry_type == "dna":
            mods = None
            if entry.get("modifications"):
                mods = [
                    Modification(position=m["position"], ccd=m["ccd"])
                    for m in entry["modifications"]
                ]
            sequences.append(
                DNAInput(id=entry_id, sequence=entry["sequence"], modifications=mods)
            )
        elif entry_type == "rna":
            mods = None
            if entry.get("modifications"):
                mods = [
                    Modification(position=m["position"], ccd=m["ccd"])
                    for m in entry["modifications"]
                ]
            sequences.append(
                RNAInput(id=entry_id, sequence=entry["sequence"], modifications=mods)
            )
        elif entry_type == "ligand":
            kwargs: dict = {"id": entry_id}
            if entry.get("ccd"):
                kwargs["ccd"] = entry["ccd"]
            elif entry.get("smiles"):
                kwargs["smiles"] = entry["smiles"]
            sequences.append(LigandInput(**kwargs))

    return StructurePredictionInput(sequences=sequences)


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading input from {args.input_json}", flush=True)
    doc = json.loads(args.input_json.read_text(encoding="utf-8"))

    print("Loading ESMFold2 model...", flush=True)
    t0 = time.time()

    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    model = ESMFold2Model.from_pretrained(args.model_dir).cuda().eval()
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s", flush=True)

    ccd_cache = args.ccd_path.parent if args.ccd_path else None
    from esm.models.esmfold2 import ESMFold2InputBuilder

    try:
        from esm.utils.msa import MSA as msa_class
    except ImportError:
        msa_class = None

    builder = ESMFold2InputBuilder(ccd_cache=ccd_cache)

    spi = build_structure_prediction_input(doc, msa_class=msa_class)

    fold_kwargs: dict = {
        "num_loops": args.num_loops,
        "num_sampling_steps": args.num_sampling_steps,
        "num_diffusion_samples": args.num_diffusion_samples,
    }
    if args.seed is not None:
        fold_kwargs["seed"] = args.seed
    if args.noise_scale is not None:
        fold_kwargs["noise_scale"] = args.noise_scale
    if args.step_scale is not None:
        fold_kwargs["step_scale"] = args.step_scale

    print(
        f"Running inference: loops={args.num_loops}, steps={args.num_sampling_steps}, "
        f"samples={args.num_diffusion_samples}",
        flush=True,
    )
    t0 = time.time()
    result = builder.fold(model, spi, **fold_kwargs)
    infer_time = time.time() - t0
    print(f"Inference completed in {infer_time:.1f}s", flush=True)

    if not isinstance(result, list):
        result = [result]

    metrics: dict = {"samples": []}
    for i, r in enumerate(result):
        cif_path = args.output_dir / f"prediction_{i}.cif"
        cif_path.write_text(r.complex.to_mmcif(), encoding="utf-8")
        print(f"  Saved {cif_path.name}", flush=True)

        sample_metrics: dict = {
            "sample_index": i,
            "output_file": cif_path.name,
        }
        if r.ptm is not None:
            sample_metrics["ptm"] = round(float(r.ptm), 4)
        if r.iptm is not None:
            sample_metrics["iptm"] = round(float(r.iptm), 4)
        if r.plddt is not None:
            sample_metrics["plddt_mean"] = round(float(r.plddt.mean()), 4)
        metrics["samples"].append(sample_metrics)

    metrics["model_load_time_s"] = round(load_time, 1)
    metrics["inference_time_s"] = round(infer_time, 1)

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  Saved {metrics_path.name}", flush=True)

    print(
        f"\nDone! {len(result)} sample(s) in {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
