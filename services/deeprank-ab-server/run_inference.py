#!/usr/bin/env python3
"""Self-contained DeepRank-Ab inference entry point.

Inlines all pipeline logic from the upstream ``scripts/inference.py`` so that
the subprocess never needs to ``importlib``-load that file (which polluted
``sys.path`` and caused module-name collisions with the server package).

Environment variables consumed (all optional, with sane Docker defaults):

  DEEPRANK_AB_ROOT        – parent of the ``DeepRank-Ab/`` checkout
  DEEPRANK_AB_NUM_WORKERS – DataLoader workers (upstream hardcodes 96)
  WEIGHT_PATH             – absolute path to ``esm2_t33_650M_UR50D.pt``
  REG_WEIGHT_PATH         – absolute path to the contact-regression weights
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# sys.path setup — MUST happen before any ``from src.*`` import so that
# ``src/tools/`` (DeepRank-Ab) is found before ``/opt/deeprank-ab/server/``.
# ---------------------------------------------------------------------------
import os
import sys
from pathlib import Path

_DEEPRANK_AB_ROOT = os.environ.get("DEEPRANK_AB_ROOT", "/opt/deeprank-ab")
_PROJECT_DIR = os.path.join(_DEEPRANK_AB_ROOT, "DeepRank-Ab")
_SRC_DIR = os.path.join(_PROJECT_DIR, "src")

# Remove the script's own directory if Python auto-added it, to avoid the
# ``server/argv.py`` ↔ ``src/tools/`` collision entirely.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR in sys.path:
    sys.path.remove(_SCRIPT_DIR)

# Insert at front so ``from src.*`` resolves inside the DeepRank-Ab tree.
# Also add ``src/`` because upstream modules (NeuralNet_focal_EMA, etc.) use
# bare imports like ``from tools.FocalLoss import BMCLoss``.
for _p in (_SRC_DIR, _PROJECT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Imports (safe now that sys.path is correct)
# ---------------------------------------------------------------------------
import argparse
import hashlib
import json
import logging
import shutil
import tempfile
from typing import List, Optional

import h5py
import numpy as np
import pandas as pd
import requests
import torch
from Bio.PDB import PDBParser, PDBIO

from esm import FastaBatchedDataset, pretrained

from src.DataSet import HDF5DataSet, PreCluster
from src.EGNN import egnn
from src.GraphGenMP import GraphHDF5
from src.NeuralNet_focal_EMA import NeuralNet
from src.tools.annotate import annotate_folder_one_by_one_mp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ROOT_DIR = Path(_PROJECT_DIR)

MODEL_PATH = (
    _ROOT_DIR
    / "src"
    / "weights"
    / "af"
    / "top_mse_ep37_mse2.3192e-02_treg_ydockq_b128_e100_lr2e-03.pth.tar"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOKS_PER_BATCH = 4096
REPR_LAYERS = [33]
TRUNCATION_SEQ_LENGTH = 2500
INCLUDE = ["mean", "per_tok"]

MAX_CORES = 50
BATCH_SIZE = 64
NUM_WORKERS = int(os.environ.get("DEEPRANK_AB_NUM_WORKERS", "8"))

TARGET = "dockq"
TASK = "reg"
THRESHOLD = 0.23

NODE_FEATURES = ["atom_type", "polarity", "bsa", "region", "embedding"]
EDGE_FEATURES = ["voro_area", "covalent", "vdw", "orientation"]

ESM_MODEL = "esm2_t33_650M_UR50D"
EXPECTED_CHECKSUMS = [
    "ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c",
    "8ffe6edbd4173dc8d45c2cd5cb27d43aad77ec26b4c768200c58ae1f96693575",
]

log = logging.getLogger("DeepRankAB")
log.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter(" [%(levelname)s] %(message)s"))
log.addHandler(_handler)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _norm_chain(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if s.lower() in {"", "none", "null", "-", "na"}:
        return None
    return s


def clamp_cores(requested: int, task_count: int) -> int:
    avail = os.cpu_count() or 1
    return max(1, min(requested, avail, task_count))


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


def setup_workspace(identificator: str) -> Path:
    workspace = Path.cwd() / identificator
    workspace.mkdir(parents=True, exist_ok=True)
    log.info(f"Workspace → {workspace}")
    return workspace


# ---------------------------------------------------------------------------
# PDB processing
# ---------------------------------------------------------------------------


def split_input_pdb(pdb_file: Path, out_dir: Path) -> List[Path]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)
    out_dir.mkdir(exist_ok=True)

    io = PDBIO()
    saved: List[Path] = []
    is_ensemble = len(list(structure)) > 1

    for model in structure:
        model_id = model.id if is_ensemble else 0
        out = out_dir / f"{pdb_file.stem}_model_{model_id}.pdb"
        io.set_structure(model)
        io.save(str(out))
        saved.append(out)

    log.info(f"Split PDB into {len(saved)} model(s)")
    return saved


_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def get_chain_sequence(structure, chain_id: Optional[str]) -> str:
    if not chain_id:
        return ""
    try:
        chain = structure[0][chain_id]
    except KeyError:
        return ""
    return "".join(_THREE_TO_ONE.get(r.get_resname(), "X") for r in chain)


def create_merged_pdb(
    pdb_file, heavy_chain_id, light_chain_id, antigen_chain_id, output_pdb,
):
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Structure import Structure

    parser = PDBParser(QUIET=True)
    s0 = parser.get_structure("model", pdb_file)
    m0 = next(s0.get_models())

    HID = _norm_chain(heavy_chain_id)
    LID = _norm_chain(light_chain_id)
    AID = _norm_chain(antigen_chain_id)

    chain_ids = {c.id for c in m0.get_chains()}
    if HID not in chain_ids:
        raise ValueError(f"Heavy chain '{HID}' not found (have {sorted(chain_ids)})")
    if AID not in chain_ids:
        raise ValueError(f"Antigen chain '{AID}' not found (have {sorted(chain_ids)})")
    if LID is not None and LID not in chain_ids:
        raise ValueError(f"Light chain '{LID}' not found (have {sorted(chain_ids)})")

    chH = m0[HID]
    chL = m0[LID] if LID is not None else None
    chAg = m0[AID]

    s = Structure("merged")
    m = Model(0)
    s.add(m)
    chAb = Chain("A")
    chBg = Chain("B")
    m.add(chAb)
    m.add(chBg)

    def _sorted_res(chain_obj):
        residues = list(chain_obj.get_residues())

        def key(r):
            hetflag, resseq, icode = r.id
            icode_key = "" if icode == " " else icode
            return (hetflag.strip(), int(resseq), icode_key)

        return sorted(residues, key=key)

    def _copy_into(dst_chain, src_chain, start_resseq=1):
        cur = start_resseq
        for r0 in _sorted_res(src_chain):
            r = r0.copy()
            hetflag, _, _icode = r.id
            r.id = (hetflag, cur, " ")
            dst_chain.add(r)
            cur += 1
        return cur

    next_res = _copy_into(chAb, chH, start_resseq=1)
    if chL is not None:
        _copy_into(chAb, chL, start_resseq=next_res)
    _copy_into(chBg, chAg, start_resseq=1)

    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(s)
    io.save(str(output_pdb))


def convert_pdb_to_fastas(
    pdb_file: Path, fasta_outdir: Path,
    heavy_chain_id="H", light_chain_id="L", antigen_chain_id="A",
):
    fasta_outdir.mkdir(parents=True, exist_ok=True)

    HID = _norm_chain(heavy_chain_id)
    LID = _norm_chain(light_chain_id)
    AID = _norm_chain(antigen_chain_id)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", pdb_file)
    root = pdb_file.stem

    seq_H = get_chain_sequence(structure, HID)
    seq_L = get_chain_sequence(structure, LID)
    seq_Ag = get_chain_sequence(structure, AID)

    fasta_annot = fasta_outdir / f"{root}_HL.fasta"
    with open(fasta_annot, "w") as f:
        if seq_H:
            f.write(f">{root}.H\n{seq_H}\n")
        if seq_L:
            f.write(f">{root}.L\n{seq_L}\n")

    fasta_esm = fasta_outdir / f"{root}_merged_A_B.fasta"
    with open(fasta_esm, "w") as f:
        f.write(f">{root}.A\n{seq_H + seq_L}\n")
        f.write(f">{root}.B\n{seq_Ag}\n")

    log.info(f"FASTA files created for {pdb_file.name}")
    return fasta_annot, fasta_esm


def preprocess_input_pdb(
    work_dir: Path, pdb_file: Path,
    heavy_chain_id: str, light_chain_id: str, antigen_chain_id: str,
):
    fasta_out = work_dir / "fastas"
    fasta_annot, fasta_esm = convert_pdb_to_fastas(
        pdb_file, fasta_out,
        heavy_chain_id=heavy_chain_id,
        light_chain_id=light_chain_id,
        antigen_chain_id=antigen_chain_id,
    )
    merged_pdb = work_dir / "processed" / f"{pdb_file.stem}.pdb"
    create_merged_pdb(
        pdb_file, heavy_chain_id, light_chain_id, antigen_chain_id, merged_pdb,
    )
    return merged_pdb, (fasta_annot, fasta_esm)


# ---------------------------------------------------------------------------
# ESM-2 weights & embeddings
# ---------------------------------------------------------------------------


def calculate_checksum(path: str, algo="sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_weights(url: str, dest: str) -> str:
    resp = requests.get(url, stream=True, timeout=10)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    return dest


def fetch_weights() -> str:
    models = [
        (
            "WEIGHT_PATH",
            f"{ESM_MODEL}.pt",
            f"https://dl.fbaipublicfiles.com/fair-esm/models/{ESM_MODEL}.pt",
            EXPECTED_CHECKSUMS[0],
        ),
        (
            "REG_WEIGHT_PATH",
            f"{ESM_MODEL}-contact-regression.pt",
            f"https://dl.fbaipublicfiles.com/fair-esm/regression/{ESM_MODEL}-contact-regression.pt",
            EXPECTED_CHECKSUMS[1],
        ),
    ]
    for env_var, fname, url, checksum in models:
        path = os.getenv(env_var) or fname
        if not os.path.exists(path):
            download_weights(url, path)
        if calculate_checksum(path) != checksum:
            log.warning(f"Checksum mismatch for {path}, re-downloading.")
            download_weights(url, path)

    return os.getenv("WEIGHT_PATH") or f"{ESM_MODEL}.pt"


def get_model_output(toks, model, layers):
    out = model(toks, repr_layers=layers, return_contacts="contacts" in INCLUDE)
    return {layer: t.cpu() for layer, t in out["representations"].items()}


def process_batch(labels, strs, outdir, representations):
    paths = []
    for i, label in enumerate(labels):
        data = {}
        trunc = min(TRUNCATION_SEQ_LENGTH, len(strs[i]))
        if "per_tok" in INCLUDE:
            data["representations"] = {
                l: t[i, 1: trunc + 1].clone() for l, t in representations.items()
            }
        if "mean" in INCLUDE:
            data["mean_representations"] = {
                l: t[i, 1: trunc + 1].mean(0).clone()
                for l, t in representations.items()
            }
        outpath = outdir / f"{label}.pt"
        torch.save(data, outpath)
        paths.append(outpath)
    return paths


def get_embedding(fasta_file: Path, output_dir: Path) -> List[Path]:
    log.info("Generating ESM embeddings...")

    esm_path = fetch_weights()
    model, alphabet = pretrained.load_model_and_alphabet(esm_path)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(TOKS_PER_BATCH, extra_toks_per_seq=1)
    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(TRUNCATION_SEQ_LENGTH),
        batch_sampler=batches,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    repr_layers = [
        (i + model.num_layers + 1) % (model.num_layers + 1) for i in REPR_LAYERS
    ]

    emb_paths: List[Path] = []
    with torch.no_grad():
        for labels, strs, toks in loader:
            if torch.cuda.is_available():
                toks = toks.cuda(non_blocking=True)
            reps = get_model_output(toks, model, repr_layers)
            emb_paths.extend(process_batch(labels, strs, output_dir, reps))

    return emb_paths


# ---------------------------------------------------------------------------
# Graph generation & clustering
# ---------------------------------------------------------------------------


def cluster(hdf5_path: str):
    dataset = HDF5DataSet(name="Train", root="./", database=hdf5_path)
    PreCluster(dataset, method="mcl")
    log.info("Clustering completed.")


def gen_graph(
    pdb_dir: str, n_cores: int, outfile_name: str,
    use_regions=True, region_json=None, antigen_chainid=None,
    graph_type="atom", use_voro=False, embedding_path=None,
    contact_features=True,
):
    base = Path(pdb_dir).name
    tmpdir = tempfile.mkdtemp(prefix=f"{base}_")
    GraphHDF5(
        pdb_path=pdb_dir,
        graph_type=graph_type,
        outfile=outfile_name,
        nproc=n_cores,
        tmpdir=tmpdir,
        use_regions=use_regions,
        region_json=region_json,
        add_orientation=True,
        contact_features=contact_features,
        antigen_chainid=antigen_chainid,
        use_voro=use_voro,
        embedding_path=embedding_path,
    )
    return outfile_name


def correct_json(work_dir: Path) -> Path:
    region_json = Path(work_dir) / "annotations" / "annotations_cdrs.json"
    if not region_json.is_file():
        raise FileNotFoundError(f"Region JSON not found: {region_json}")
    with open(region_json) as f:
        data = json.load(f)
    new = {k.replace(".pdb", ""): v for k, v in data.items()}
    with open(region_json, "w") as f:
        json.dump(new, f, indent=4)
    return region_json


def add_embedding(work_dir: Path, hdf5_path: str):
    base = work_dir / "processed" / "embeddings"

    def _resolve_pt(mol_name: str, chain: str) -> Path:
        candidates = [mol_name, mol_name.replace(".pdb", ""), mol_name.split("/")[-1]]
        if candidates[0].endswith("_graph"):
            candidates.append(candidates[0].removesuffix("_graph"))
        for cand in candidates:
            pt = base / f"{cand}.{chain}.pt"
            if pt.is_file():
                return pt
        return base / f"{Path(mol_name).stem}.{chain}.pt"

    with h5py.File(hdf5_path, "a") as f:
        for mol in list(f.keys()):
            residues = f[mol]["nodes"][()]
            emb = torch.zeros(len(residues), 1)
            for i, res in enumerate(residues):
                chain, idx = res[0].decode(), int(res[1].decode())
                pt = _resolve_pt(mol, chain)
                if not pt.is_file():
                    raise FileNotFoundError(
                        f"Missing embedding: {pt} (mol={mol}, chain={chain})"
                    )
                data = torch.load(pt, map_location="cpu")
                vecs = data["representations"][33]
                if idx < 1 or idx > vecs.shape[0]:
                    raise IndexError(
                        f"Residue index out of bounds for {pt}: "
                        f"idx={idx}, len={vecs.shape[0]}"
                    )
                emb[i] = vecs[idx - 1].mean()
            grp = f[mol].require_group("node_data")
            if "embedding" in grp:
                del grp["embedding"]
            grp.create_dataset("embedding", data=emb.numpy())

    log.info(f"Added embeddings → {hdf5_path}")


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def deeprank_evaluate_model(
    target_name: str, hdf5_test: str,
    model_path: str = str(MODEL_PATH),
    save_name: str = "eval_predictions.hdf5",
) -> Path:
    net = NeuralNet(
        database=hdf5_test,
        Net=egnn,
        node_feature=NODE_FEATURES,
        edge_feature=EDGE_FEATURES,
        target=None,
        task=TASK,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device_name=DEVICE,
        shuffle=False,
        pretrained_model=model_path,
        cluster_nodes="mcl",
    )
    out = Path(hdf5_test).parent / save_name
    net.predict(database_test=hdf5_test, hdf5=str(out))
    log.info(f"Predictions saved → {out}")
    return out


def hdf5_to_csv(hdf5_path: str) -> str:
    hdf5_path = Path(hdf5_path)
    out_csv = hdf5_path.with_suffix(".csv")
    with h5py.File(hdf5_path, "r") as f:
        group = f["epoch_0000"]["pred"]
        mol = [m.decode("utf-8") for m in group["mol"][()]]
        dockq = group["outputs"][()]
    df = pd.DataFrame({"mol": mol, "dockq": dockq})
    df.to_csv(out_csv, index=False)
    return str(out_csv)


# ---------------------------------------------------------------------------
# Contact check & output parsing
# ---------------------------------------------------------------------------


def ca_coords(pdb_file: Path, chain_id: str, max_resid=125) -> np.ndarray:
    coords = []
    residue_index = 0
    with open(pdb_file, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            chain = line[21].strip()
            if chain != chain_id or atom_name != "CA":
                continue
            if residue_index > max_resid:
                break
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])
            residue_index += 1
    arr = np.array(coords, dtype=float)
    if arr.size == 0:
        arr = arr.reshape((0, 3))
    return arr


def flag_low_contacts(
    pdb_file: Path, heavy_chain_id: str, light_chain_id: str, threshold: int = 15,
):
    coords_H = ca_coords(pdb_file, heavy_chain_id, max_resid=125)
    coords_L = ca_coords(pdb_file, light_chain_id, max_resid=115)
    if coords_H.shape[0] == 0 or coords_L.shape[0] == 0:
        return False
    dists = np.linalg.norm(coords_H[:, None, :] - coords_L[None, :, :], axis=-1)
    n_contacts = int(np.sum(dists < 8.0))
    if n_contacts < threshold:
        log.warning(
            f"[FLAG] LOW H-L CONTACTS → {pdb_file.name} with {n_contacts} contacts"
        )
        return True
    return False


def parse_output(
    csv_output: str, original_pdb_dir: Path,
    heavy_chain_id=None, light_chain_id=None,
) -> None:
    df = pd.read_csv(csv_output)
    df = df.rename(columns={"mol": "pdb_id", "dockq": "predicted_dockq"})

    flags = []
    for pdb_id in df["pdb_id"]:
        matches = list(original_pdb_dir.glob(f"{pdb_id}*.pdb"))
        pdb_file = matches[0]

        if (
            not heavy_chain_id
            or not light_chain_id
            or heavy_chain_id == "-"
            or light_chain_id == "-"
        ):
            flags.append("not_applicable")
            continue

        is_bad = flag_low_contacts(pdb_file, heavy_chain_id, light_chain_id)
        flags.append("low_HL_contacts" if is_bad else "ok")

    df["quality_flag"] = flags
    df = df.sort_values(by="predicted_dockq", ascending=False)
    df.to_csv(csv_output, index=False)
    log.info(f"Output written → {csv_output}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="DeepRank-Ab inference pipeline")
    parser.add_argument("pdb_file")
    parser.add_argument("heavy_chain_id")
    parser.add_argument("light_chain_id")
    parser.add_argument("antigen_chain_id")
    args = parser.parse_args()

    pdb_file = Path(args.pdb_file)
    HID = args.heavy_chain_id
    LID = args.light_chain_id
    AID = args.antigen_chain_id

    identificator = f"{pdb_file.stem}-deeprank_ab_pred_{HID}{LID}_{AID}"
    work = setup_workspace(identificator)

    copied = work / pdb_file.name
    shutil.copy(pdb_file, copied)

    split_dir = work / "original_models"
    split_dir.mkdir(exist_ok=True)
    pdb_models = split_input_pdb(copied, split_dir)

    avail = os.cpu_count() or 1
    cores = max(1, min(MAX_CORES, avail))

    processed_dir = work / "processed"
    processed_dir.mkdir(exist_ok=True)

    # Build merged PDBs + embeddings
    fasta_annot = None
    embed_dir = processed_dir / "embeddings"
    embed_dir.mkdir(exist_ok=True)

    for pdb in pdb_models:
        _merged_pdb, (fasta_annot, fasta_esm) = preprocess_input_pdb(
            work, pdb, HID, LID, AID,
        )
        get_embedding(fasta_esm, embed_dir)

    if fasta_annot is None:
        raise RuntimeError("No FASTA files were produced (unexpected).")

    # Annotation
    anno_dir = work / "annotations"
    anno_dir.mkdir(exist_ok=True)
    n_cores_anno = clamp_cores(cores, len(pdb_models))

    annotate_folder_one_by_one_mp(
        processed_dir,
        Path(fasta_annot.parent),
        output_dir=str(anno_dir),
        n_cores=n_cores_anno,
        antigen_chainid="B",
    )

    region_json = correct_json(work)
    log.info("Region JSON corrected.")

    # Graph generation
    pdbs_for_graph = sorted(processed_dir.glob("*.pdb"))
    n_cores_graph = clamp_cores(cores, len(pdbs_for_graph))
    graph_out = work / f"{identificator}_graph.hdf5"

    gen_graph(
        pdb_dir=str(processed_dir),
        n_cores=n_cores_graph,
        outfile_name=str(graph_out),
        use_regions=True,
        region_json=str(region_json),
        antigen_chainid="B",
        graph_type="atom",
        use_voro=True,
        embedding_path=str(embed_dir),
        contact_features=True,
    )

    # Evaluation
    cluster(str(graph_out))

    deeprank_evaluate_model(
        "dockq",
        str(graph_out),
        model_path=str(MODEL_PATH),
        save_name=f"{identificator}_predictions.hdf5",
    )

    pred_h5 = work / f"{identificator}_predictions.hdf5"
    csv_out = hdf5_to_csv(str(pred_h5))

    # Contact check
    parse_output(csv_out, split_dir, heavy_chain_id=HID, light_chain_id=LID)


if __name__ == "__main__":
    main()
