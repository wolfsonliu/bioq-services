"""PLIP service adapter: output detection + subprocess env + manifest + examples.

PLIP is CPU-only, rule-based, and has NO model weights. `detect_outputs`
accepts any non-empty file under `output/` (label-agnostic: xml / txt / pse /
png / protonated pdb). `subprocess_env` injects the vendored upstream root onto
PYTHONPATH so `python -m plip.plipcmd` can `import plip` regardless of cwd.
"""

from __future__ import annotations

import os
from pathlib import Path

from bioagent_service import EndpointExample, JobAdapter

from .settings import PlipSettings


class PlipAdapter(JobAdapter):
    name = "plip"

    settings: PlipSettings  # narrow for IDEs

    def __init__(self, settings: PlipSettings) -> None:
        super().__init__(settings)

    def detect_outputs(self, job_dir: Path) -> bool:
        out = self.output_dir(job_dir)
        if not out.is_dir():
            return False
        for path in out.rglob("*"):
            try:
                if path.is_file() and path.stat().st_size > 0:
                    return True
            except OSError:
                continue
        return False

    def subprocess_env(self) -> dict[str, str]:
        """Put the vendored PLIP source root on PYTHONPATH so `import plip` works.

        Preserve any inherited PYTHONPATH (prepend, don't clobber).
        """
        prefix = self.settings.pythonpath or str(self.settings.upstream_dir)
        existing = os.environ.get("PYTHONPATH", "")
        pythonpath = f"{prefix}:{existing}" if existing else prefix
        return {"PYTHONPATH": pythonpath}

    def manifest_extras(self) -> dict:
        return {
            "protocol": (
                "PLIP is CPU-only and rule-based (no GPU, no model weights). "
                "It reads ONE PDB complex (protein + ligand already present) and "
                "reports the non-covalent interactions it detects. It does NOT "
                "score, predict affinity, or prepare structures."
            ),
            "tool_outputs": {
                "profile": (
                    "output/<name>.xml — structured interaction report (primary "
                    "machine-readable output; see report/bindingsite/interactions). "
                    "output/<name>.txt — human-readable RST report. "
                    "output/*.pse — PyMOL sessions per binding site (pymol_session=True). "
                    "output/*.png — ray-traced images (render_images=True). "
                    "output/<name>_protonated.pdb — PLIP's fixed/protonated structure."
                ),
                "note": (
                    "Enumerate via GET /api/jobs/{id}/files; single file via "
                    "/api/jobs/{id}/file/{relpath}; whole dir via /download."
                ),
            },
            "modes": {
                "default": "Automatic ligand detection (HETATM-based).",
                "peptide": "Treat given chain(s) as peptide ligands / inter-chain contacts (peptide_chains required).",
                "intra": "Intra-chain contacts within one chain (intra_chain required; slow on large structures).",
                "dnareceptor": "Treat DNA/RNA as part of the receptor rather than as a ligand.",
            },
            "interaction_types": [
                "hydrophobic_interaction", "hydrogen_bond", "water_bridge",
                "salt_bridge", "pi_stack", "pi_cation_interaction",
                "halogen_bond", "metal_complex",
            ],
            "input_uri_schemes": {
                "upload": "multipart/form-data (input_pdb).",
                "job://<id>/<file>": "Re-use a PDB from a prior job.",
                "file:///abs/path": "Direct NAS path on the shared mount.",
                "oss://<bucket>/<key>": "Alibaba Cloud OSS object.",
                "http(s)://...": "Generic URL, including OSS signed URLs.",
            },
            "endpoints_summary": {
                "/api/profile": "Profile interactions in one PDB complex.",
                "/api/tasks/profile": "FC async task-mode variant.",
                "/healthz/detail": "Extended health (upstream + openbabel/pymol import).",
            },
            "not_in_scope_v0_0_1": (
                "batch multi-structure; fetch-by-PDB-ID over the network (-i); "
                "--chains / --regions advanced receptor/ligand grouping; geometric "
                "threshold tuning (--hbond_dist_max etc.); stdout reports (-O); "
                "gzip (-z). PLIP does not score or predict affinity."
            ),
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/profile": [
                EndpointExample(
                    title="default ligand-interaction profile (XML + TXT)",
                    curl=(
                        "curl -X POST $URL/api/profile "
                        "-F input_pdb=@complex.pdb "
                        "-F name=complex"
                    ),
                    notes="Results in output/complex.xml and output/complex.txt.",
                ),
                EndpointExample(
                    title="protein-peptide (inter-chain) mode with PyMOL session",
                    curl=(
                        "curl -X POST $URL/api/profile "
                        "-F input_pdb=@complex.pdb "
                        "-F mode=peptide "
                        "-F peptide_chains=I "
                        "-F pymol_session=true "
                        "-F name=complex"
                    ),
                    notes="Chain I is treated as a peptide ligand; also writes .pse per binding site.",
                ),
                EndpointExample(
                    title="re-use a docked pose from a prior job",
                    curl=(
                        "curl -X POST $URL/api/profile "
                        "-F input_pdb_uri=job://<dock_job_id>/output/pose_1.pdb "
                        "-F name=pose1"
                    ),
                    notes="`input_pdb_uri` accepts job:// / oss:// / file:// / http(s)://.",
                ),
            ],
        }


__all__ = ["PlipAdapter"]
