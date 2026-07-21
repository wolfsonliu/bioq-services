"""Service-wide policy: name + output detection + manifest_extras +
endpoint_examples for pocketxmol-server.

The manifest is agent-facing: keep entries copy-pasteable and honest
about what each of the 6 endpoints does.
"""

from __future__ import annotations

from pathlib import Path

from bioq_service import EndpointExample, JobAdapter

from .settings import PocketXMolSettings


class PocketXMolAdapter(JobAdapter):
    name = "pocketxmol"

    settings: PocketXMolSettings

    def __init__(self, settings: PocketXMolSettings) -> None:
        super().__init__(settings)

    def detect_outputs(self, job_dir: Path) -> bool:
        """True if any generation SDF/PDB or confidence ranking CSV exists.

        Upstream sample_use.py lands SDF/PDB files under
        ``output/<exp_name>_<timestamp>/<exp_name>_<timestamp>_SDF/``.
        Upstream believe_use_pdb.py lands CSVs under
        ``output/<exp_name>_<timestamp>/ranking/``.
        Use rglob because the depth is variable.
        """
        out = self.output_dir(job_dir)
        if not out.exists():
            return False

        # Generation outputs — SDF (small mol) or PDB (peptide) inside a
        # *_SDF subdir.
        for pattern in ("*.sdf", "*.pdb"):
            for p in out.rglob(pattern):
                if not p.is_file() or p.stat().st_size == 0:
                    continue
                # Skip the "0_inputs" subdir which contains pocket_block.pdb
                # + input_mol.sdf echoed back from upstream — that's not a
                # real output, and it always exists.
                if "0_inputs" in p.parts:
                    continue
                if "_SDF" in p.parent.name:
                    return True

        # Confidence rankings.
        for p in out.rglob("ranking/*.csv"):
            if p.is_file() and p.stat().st_size > 0:
                return True

        return False

    def subprocess_cwd(self) -> Path | None:
        return self.settings.root

    def manifest_extras(self) -> dict:
        return {
            "model": {
                "name": "PocketXMol",
                "paper": "Peng et al., Cell 2026 (10.1016/j.cell.2026.01.003)",
                "license": "MIT",
                "checkpoint": self.settings.pxm_checkpoint.name,
                "capabilities": [
                    "small-molecule docking", "peptide docking",
                    "structure-based drug design (de novo)",
                    "fragment linking / growing / PROTAC",
                    "molecular optimization",
                    "peptide design (linear / cyclic / inverse-fold / sc-pack)",
                    "tuned-ranker confidence scoring",
                ],
            },
            "tool_outputs": {
                "dock": "output/<exp>_<ts>/<exp>_<ts>_SDF/*.sdf (small-mol) or *.pdb (peptide) + gen_info.csv",
                "sbdd": "output/<exp>_<ts>/<exp>_<ts>_SDF/*.sdf + gen_info.csv (cfd_traj self-confidence)",
                "linking": "output/<exp>_<ts>/<exp>_<ts>_SDF/*.sdf + gen_info.csv",
                "optimize": "output/<exp>_<ts>/<exp>_<ts>_SDF/*.sdf + gen_info.csv",
                "pepdesign": "output/<exp>_<ts>/<exp>_<ts>_SDF/*.sdf (mol) + *.pdb (peptide) + gen_info.csv",
                "confidence": "output/<exp>_<ts>/ranking/*.csv (tuned-ranker scores per molecule)",
            },
            "input_uri_schemes": {
                "protein": "multipart / oss:// / file:// / job:// / http(s)://",
                "ligand": "multipart / oss:// / file:// / job:// / http(s)://",
                "ref_ligand": "multipart / oss:// / file:// / job:// / http(s)://",
                "input_ligand": "multipart / oss:// / file:// / job:// / http(s)://",
                "input_peptide": "multipart / oss:// / file:// / job:// / http(s)://",
            },
            "task_selection_hint": {
                "known_ligand_docking": "use /api/dock",
                "de_novo_sbdd": "use /api/sbdd (mode=ar for quality, mode=simple for speed)",
                "fragment_linker_or_grow_or_protac": "use /api/linking with fragments as list-of-lists",
                "refine_existing_mol": "use /api/optimize with init_step=0.5",
                "peptide_design_linear_or_cyclic": "use /api/pepdesign mode=denovo_linear|denovo_cyclic",
                "peptide_inverse_fold": "use /api/pepdesign mode=inverse_fold with peptide PDB",
                "score_generated_ligands": "chain /api/confidence with source_job_id",
            },
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/dock": [
                EndpointExample(
                    title="dock a small molecule (SDF)",
                    curl=(
                        "curl -X POST $URL/api/dock "
                        "-F protein=@8C7Y_TXV_protein.pdb "
                        "-F ligand=@8C7Y_TXV_ligand.sdf "
                        "-F num_samples=10 "
                        "-F pocket_coord=-8.257 -F pocket_coord=85.181 -F pocket_coord=19.050 "
                        "-F pocket_radius=15"
                    ),
                    notes="For SMILES input pass -F smiles=... instead of ligand file.",
                ),
                EndpointExample(
                    title="dock a peptide (from sequence)",
                    curl=(
                        "curl -X POST $URL/api/dock "
                        "-F protein=@3bik_A.pdb "
                        "-F is_pep=true -F pep_sequence=DTVFALFW "
                        "-F pocket_radius=20 -F num_samples=5"
                    ),
                    notes="For peptide from PDB, pass -F ligand=@peptide.pdb instead of pep_sequence.",
                ),
            ],
            "/api/sbdd": [
                EndpointExample(
                    title="de novo SBDD with AR refinement",
                    curl=(
                        "curl -X POST $URL/api/sbdd "
                        "-F protein=@2ar9_A.pdb "
                        "-F pocket_coord=-8.16 -F pocket_coord=36.70 -F pocket_coord=38.77 "
                        "-F pocket_radius=15 -F num_samples=50 -F mode=ar"
                    ),
                    notes="pocket_coord is required — de novo has no reference "
                    "ligand to derive it from.",
                ),
            ],
            "/api/linking": [
                EndpointExample(
                    title="fragment growing (1 group)",
                    curl=(
                        "curl -X POST $URL/api/linking "
                        "-F protein=@2ar9_A.pdb "
                        "-F input_ligand=@fragment.sdf "
                        "-F fragments='[[0,1,2,3,4,5,6]]' "
                        "-F mol_size_mean=28 -F num_samples=10"
                    ),
                    notes="Single fragment group = growing.  Fragments field "
                    "is a JSON list-of-lists of 0-based atom indices.",
                ),
                EndpointExample(
                    title="fragment linking (2 groups) — good for PROTAC linker design",
                    curl=(
                        "curl -X POST $URL/api/linking "
                        "-F protein=@target.pdb "
                        "-F input_ligand=@warhead_e3.sdf "
                        "-F fragments='[[0,1,2,3,4,5,6],[23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]]' "
                        "-F mol_size_mean=60 -F part1_pert=fixed"
                    ),
                    notes="For PROTAC: warhead atoms in group 1, E3-ligand atoms "
                    "in group 2; linker is generated between them.",
                ),
            ],
            "/api/optimize": [
                EndpointExample(
                    title="local optimization of an existing mol",
                    curl=(
                        "curl -X POST $URL/api/optimize "
                        "-F protein=@target.pdb "
                        "-F input_ligand=@parent.sdf "
                        "-F init_step=0.3 -F num_samples=20"
                    ),
                    notes="Smaller init_step → closer to input; larger → more "
                    "exploration.  Use 0.3–0.6 for lead-opt.",
                ),
            ],
            "/api/pepdesign": [
                EndpointExample(
                    title="de novo linear peptide (10 residues)",
                    curl=(
                        "curl -X POST $URL/api/pepdesign "
                        "-F protein=@3bik_A.pdb "
                        "-F mode=denovo_linear -F pep_length=10 "
                        "-F ref_ligand=@3bik_A_pocket_coord.sdf "
                        "-F pocket_radius=20 -F num_samples=10"
                    ),
                    notes="For cyclic pass mode=denovo_cyclic (same shape).",
                ),
                EndpointExample(
                    title="inverse folding (design sequence for known backbone)",
                    curl=(
                        "curl -X POST $URL/api/pepdesign "
                        "-F protein=@target.pdb "
                        "-F input_peptide=@backbone.pdb "
                        "-F mode=inverse_fold -F num_samples=10"
                    ),
                    notes="Peptide length is taken from the input PDB.",
                ),
            ],
            "/api/confidence": [
                EndpointExample(
                    title="score a completed generation job (tuned ranker)",
                    curl=(
                        "curl -X POST $URL/api/confidence "
                        "-F source_job_id=<uuid-from-earlier-generate-call> "
                        "-F variant=tuned_cfd"
                    ),
                    notes="Reads output/<exp>_<ts>/<exp>_<ts>_SDF/ from the "
                    "source job and writes ranking/*.csv there.  Use "
                    "variant=flex_cfd for flexible-docking-noise scoring.",
                ),
            ],
        }
