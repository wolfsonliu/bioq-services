"""Service-wide policy: name + output detection + env + manifest_extras."""

from __future__ import annotations

import json
from pathlib import Path

from bioq_service import EndpointExample, JobAdapter

from .settings import DiffdockSettings


class DiffdockAdapter(JobAdapter):
    name = "diffdock"

    settings: DiffdockSettings

    def __init__(self, settings: DiffdockSettings) -> None:
        super().__init__(settings)

    def detect_outputs(self, job_dir: Path) -> bool:
        """True iff at least one ``rank1.sdf`` exists under output/.

        Upstream writes to ``output/<complex_name>/rank1.sdf`` (top-1 pose,
        no confidence suffix; see inference.py:289).  We don't know
        ``complex_name`` from the disk state alone — read it back from the
        JobInfo sidecar if present, else scan for any ``rank1.sdf`` under
        ``output/``.
        """
        out = self.output_dir(job_dir)
        if not out.is_dir():
            return False

        complex_name = self._infer_complex_name(job_dir)
        if complex_name:
            rank1 = out / complex_name / "rank1.sdf"
            if rank1.exists() and rank1.stat().st_size > 0:
                return True

        # Fallback: scan for any rank1.sdf under output/
        for p in out.rglob("rank1.sdf"):
            if p.is_file() and p.stat().st_size > 0:
                return True
        return False

    @staticmethod
    def _infer_complex_name(job_dir: Path) -> str | None:
        sidecar = job_dir / "job.json"
        if not sidecar.exists():
            return None
        try:
            info = json.loads(sidecar.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        params = info.get("input_params") or {}
        name = params.get("complex_name")
        return name if isinstance(name, str) and name else None

    def subprocess_cwd(self) -> Path | None:
        # Upstream utils/so3.py + utils/torus.py load LUT files via
        # CWD-relative paths (np.load('.so3_omegas_array4.npy')); those
        # `.npy` files are pre-computed at Docker build time and live at
        # settings.root.  Anchor CWD there so the loads succeed.
        return self.settings.root

    def subprocess_env(self) -> dict[str, str]:
        return {
            # Upstream imports (`from utils.xxx import ...`) are relative
            # to the repo root — expose it on PYTHONPATH.
            "PYTHONPATH": str(self.settings.root),
            # fair-esm and torch.hub read ESM-2 / ESMFold weights from
            # $TORCH_HOME/hub/checkpoints/; point that at the NAS cache.
            "TORCH_HOME": str(self.settings.esm_cache_dir),
            # Upstream inference.py:53 uses REPOSITORY_URL for its release
            # zip fallback download.  Deploy env has no outbound network;
            # a null URL makes the fallback fail loud rather than hang.
            "REPOSITORY_URL": "file:///dev/null",
        }

    def manifest_extras(self) -> dict:
        return {
            "model": {
                "name": "DiffDock-L (v1.1)",
                "paper": "Corso, Deng, Polizzi, Barzilay, Jaakkola. "
                "arXiv:2402.18396 (ICLR 2024).  Original: arXiv:2210.01776 "
                "(ICLR 2023).",
                "task": "small-molecule blind docking (score + confidence "
                "diffusion on T(3) × SO(3) × SO(2)^m)",
                "training_data": "PDBBind + BindingMOAD + DockGen + vdM "
                "augmentation",
                "note": "Rigid protein (no torsion / side-chain flexibility). "
                "Ligand torsion is modeled.  ESM-2 embedding is required; "
                "ESMFold folding kicks in only when protein_sequence is "
                "provided instead of a PDB.",
                "license": "MIT",
            },
            "tool_outputs": {
                "dock": (
                    "output/<complex_name>/rank<r>_confidence<c>.sdf for r in "
                    "[1, samples_per_complex]; "
                    "output/<complex_name>/rank1.sdf (top-1, unlabeled duplicate); "
                    "output/<complex_name>/confidence_scores.json (parsed ranking); "
                    "output/<complex_name>/<complex_name>_esmfold.pdb (only if "
                    "protein_sequence input)"
                ),
            },
            "config_tips": {
                "samples_per_complex": "10 = default (fast); 40 = paper accuracy peak; "
                ">40 = marginal returns.",
                "inference_steps": "20 = default; 40 = paper max.  Higher = more "
                "denoise steps ~= slower + slightly better.",
                "actual_steps": "Typically inference_steps - 1; the last step tends "
                "to overfit noise (upstream default 19 vs 20).",
                "batch_size": "10 fits T4 for typical protein 300 aa.  OOM => "
                "reduce to 4.",
                "no_final_step_noise": "Requested value 'false' is IGNORED in "
                "v0.0.1 (upstream is store_true default True; no CLI to disable). "
                "See design doc §Risks §4.",
                "protein_sequence": "Long sequences (>800 aa) may OOM on T4 during "
                "ESMFold.  Recommend A10 24GB or split into chains via ':'.",
                "ligand": "SMILES needs a valid parseable structure; RDKit ETKDG "
                "generates seed conformer.  If RDKit fails to embed, upstream "
                "logs 'Failed on ...' and skips → detect_outputs returns false.",
                "async_payload_limit": "FC async task mode (/api/tasks/dock) "
                "caps the request payload at 128 KiB.  A full protein PDB is "
                "almost always larger, so for /api/tasks/dock pass the protein "
                "via protein_uri (file:// on NAS, oss://, or job://) — NOT a "
                "multipart upload.  Sync /api/dock has no such cap and accepts "
                "multipart uploads directly.",
            },
            "input_uri_schemes": {
                "protein": "multipart .pdb / job:// / oss:// / file:// / http(s):// / "
                "OR protein_sequence text field (str, → ESMFold)",
                "ligand": "multipart .sdf/.mol2 / job:// / oss:// / file:// / "
                "http(s):// OR ligand_description text field (SMILES / SMARTS)",
            },
            "chaining_tip": (
                "Common upstream pipelines: (a) RFantibody / genie3 designed "
                "PDB → diffdock-server /api/dock via `protein_uri=job://<prev>/"
                "target.pdb`; (b) ESMFold-driven end-to-end: pass "
                "protein_sequence directly, DiffDock folds internally."
            ),
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        example_pdb_sdf = EndpointExample(
            title="PDB + SDF file upload (default preset)",
            curl=(
                "curl -X POST $URL/api/dock "
                "-F protein=@target.pdb "
                "-F ligand=@ligand.sdf "
                "-F complex_name=1a0q "
                "-F samples_per_complex=10 -F inference_steps=20"
            ),
            notes="Fastest path when both structures are on hand.  "
            "Output: output/1a0q/rank1.sdf + confidence_scores.json.",
        )
        example_pdb_smiles = EndpointExample(
            title="PDB + SMILES ligand",
            curl=(
                "curl -X POST $URL/api/dock "
                "-F protein=@target.pdb "
                '-F ligand_description="COc1ccc(C#N)cc1" '
                "-F complex_name=abl1_inhibitor"
            ),
            notes="RDKit ETKDG generates seed conformer from SMILES.  "
            "For invalid SMILES the job fails with `no_outputs` "
            "(RDKit embedding failure).",
        )
        example_seq_smiles = EndpointExample(
            title="protein_sequence + SMILES (ESMFold pipeline)",
            curl=(
                "curl -X POST $URL/api/dock "
                '-F protein_sequence="MKWVTFISLLFLFSSAYSRGVFRRDTHKS..." '
                '-F ligand_description="CCOc(cc1)ccc1NC(=O)C" '
                "-F complex_name=novel_target"
            ),
            notes="ESMFold folds the sequence server-side (adds ~30 s "
            "on A10 for a 300 aa protein).  Requires esmfold_3B_v1.pt "
            "in the NAS ESM cache — check /healthz/detail.",
        )
        example_async = EndpointExample(
            title="async task mode (recommended for GPU-heavy jobs)",
            curl=(
                "curl -X POST $URL/api/tasks/dock "
                '-H "X-Fc-Invocation-Type: Async" '
                "-F protein=@target.pdb "
                "-F ligand=@ligand.sdf "
                "-F samples_per_complex=40 -F inference_steps=40"
            ),
            notes="Client polls /api/jobs/<id> for final JobInfo.  "
            "Preferred over sync submit/poll on FC for jobs longer "
            "than ~2 min.",
        )
        return {
            "/api/dock": [example_pdb_sdf, example_pdb_smiles, example_seq_smiles],
            "/api/tasks/dock": [example_async],
        }
