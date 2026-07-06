"""QligfepAdapter — per-endpoint detect_outputs dispatch + manifest metadata."""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Callable

from bioagent_service import EndpointExample, JobAdapter

from .settings import QligfepSettings


class QligfepAdapter(JobAdapter):
    name = "qligfep"

    settings: QligfepSettings

    def __init__(self, settings: QligfepSettings) -> None:
        super().__init__(settings)

    def subprocess_cwd(self) -> Path | None:
        return self.settings.root

    def _read_label(self, job_dir: Path) -> str | None:
        try:
            data = json.loads((job_dir / "manifest.json").read_text())
            return data.get("label")
        except (FileNotFoundError, ValueError):
            return None

    def detect_outputs(self, job_dir: Path) -> bool:
        label = self._read_label(job_dir)
        checkers: dict[str, Callable[[Path], bool]] = {
            "ligprep":      self._detect_ligprep,
            "protprep":     self._detect_protprep,
            "cog":          self._detect_cog,
            "setup-ligfep": self._detect_setup_fep,
            "setup-resfep": self._detect_setup_fep,
            "setup-lie":    self._detect_setup_lie,
            "run-fep":      self._detect_run_fep,
            "analyze-fep":  self._detect_analyze,
            "analyze-lie":  self._detect_analyze,
        }
        checker = checkers.get(label or "")
        if checker is None:
            return False
        return checker(self.output_dir(job_dir))

    def _detect_ligprep(self, out: Path) -> bool:
        libs = list(out.glob("*.lib"))
        prms = list(out.glob("*.prm"))
        pdbs = list(out.glob("*.pdb"))
        return bool(libs) and bool(prms) and bool(pdbs) and all(
            p.stat().st_size > 0 for p in libs + prms + pdbs
        )

    def _detect_protprep(self, out: Path) -> bool:
        p = out / "protein.pdb"
        w = out / "water.pdb"
        return p.exists() and w.exists() and p.stat().st_size > 0 and w.stat().st_size > 0

    def _detect_cog(self, out: Path) -> bool:
        r = out / "result.json"
        return r.exists() and r.stat().st_size > 0

    def _detect_setup_fep(self, out: Path) -> bool:
        submit = out / "1.protein" / "FEP_submit.sh"
        if not submit.exists():
            return False
        window_inps = glob.glob(str(out / "1.protein" / "FEP*" / "md_*.inp"))
        return bool(window_inps)

    def _detect_setup_lie(self, out: Path) -> bool:
        s = out / "setup.json"
        if not s.exists() or s.stat().st_size == 0:
            return False
        bound = out / "md_LIE_bound"
        return bound.is_dir() and any(bound.iterdir())

    def _detect_run_fep(self, out: Path) -> bool:
        en_files = glob.glob(str(out / "window_*_rep_*" / "md_*.en"))
        return any(Path(f).stat().st_size > 0 for f in en_files)

    def _detect_analyze(self, out: Path) -> bool:
        r = out / "results.txt"
        return r.exists() and r.stat().st_size > 0

    def manifest_extras(self) -> dict:
        return {
            "tool_outputs": {
                "ligprep":      "output/<ligand>.lib",
                "protprep":     "output/protein.pdb",
                "cog":          "output/result.json",
                "setup-ligfep": "output/1.protein/FEP_submit.sh",
                "setup-resfep": "output/1.protein/FEP_submit.sh",
                "setup-lie":    "output/setup.json",
                "run-fep":      "output/window_<idx>_rep_<r>/md_*.en",
                "analyze-fep":  "output/results.txt",
                "analyze-lie":  "output/results.txt",
            },
            "input_uri_schemes": {"upload": "multipart/form-data", "zip": "zip archive"},
            "workflow_order": [
                "ligprep", "cog (optional)", "protprep",
                "setup-{ligfep|resfep|lie}",
                "run-fep (× N windows × M legs × R replicates)",
                "analyze-{fep|lie}",
            ],
            "forcefields_available": ["OPLS2005", "OPLS2015", "OPLSAAM", "AMBER14sb", "CHARMM36"],
            "q6_binaries": {
                "cpu":  str(self.settings.q_bin_dir / "qdyn"),
                "mpi":  str(self.settings.q_bin_dir / "qdynp"),
                "gpu":  str(self.settings.q_bin_dir / "qdyn_cuda"),
                "prep": str(self.settings.q_bin_dir / "qprep"),
                "fep":  str(self.settings.q_bin_dir / "qfep"),
                "calc": str(self.settings.q_bin_dir / "qcalc"),
            },
            "no_model_weights": True,
            "task_endpoints": "disabled (HPC-primary, no FC)",
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/ligprep": [EndpointExample(
                title="OpenFF ligand parameterization",
                curl=("curl -X POST $URL/api/ligprep "
                      "-F ligand_name=17 -F ligand=@17.mol2"),
                notes="Produces 17.lib, 17.prm, 17.pdb under output/.",
            )],
            "/api/protprep": [EndpointExample(
                title="Spherical boundary prep",
                curl=("curl -X POST $URL/api/protprep "
                      "-F sphere_radius=22 -F sphere_center=0.53:26.77:8.82 "
                      "-F forcefield=OPLSAAM -F protein_pdb=@1h1s.pdb"),
                notes="Requires already-protonated PDB.",
            )],
            "/api/cog": [EndpointExample(
                title="Center of geometry",
                curl="curl -X POST $URL/api/cog -F pdb=@17.pdb",
                notes="Returns {cx, cy, cz} for feeding into protprep.",
            )],
            "/api/setup-ligfep": [EndpointExample(
                title="QligFEP dual-topology setup",
                curl=("curl -X POST $URL/api/setup-ligfep "
                      "-F lig1_name=17 -F lig2_name=18 -F system=protein "
                      "-F ligprep_zip=@ligprep.zip -F protprep_zip=@protprep.zip"),
                notes="Produces 1.protein/ + 2.water/ + FEP_submit.sh.",
            )],
            "/api/setup-resfep": [EndpointExample(
                title="QresFEP mutation setup",
                curl=("curl -X POST $URL/api/setup-resfep "
                      "-F mutation=A24V -F mutchain=A "
                      "-F protprep_zip=@protprep.zip"),
                notes="",
            )],
            "/api/setup-lie": [EndpointExample(
                title="QLIE setup",
                curl=("curl -X POST $URL/api/setup-lie "
                      "-F ligand_name=17 "
                      "-F ligprep_zip=@ligprep.zip -F protprep_zip=@protprep.zip"),
                notes="",
            )],
            "/api/run-fep": [EndpointExample(
                title="Run a single lambda window (HPC-only in practice)",
                curl=("curl -X POST $URL/api/run-fep "
                      "-F window_idx=25 -F leg=protein -F device=gpu "
                      "-F setup_zip=@setup_leg_protein.zip"),
                notes=("Long-running (30 min - 2 h). Use CLI mode via sbatch array, "
                       "not HTTP."),
            )],
            "/api/analyze-fep": [EndpointExample(
                title="Post-process FEP energies to DDG",
                curl=("curl -X POST $URL/api/analyze-fep "
                      "-F temperature=298.15 -F fep_run_zip=@runs.zip"),
                notes="Produces results.txt + DDG.csv.",
            )],
            "/api/analyze-lie": [EndpointExample(
                title="Post-process LIE",
                curl="curl -X POST $URL/api/analyze-lie -F lie_run_zip=@runs.zip",
                notes="Produces results.txt + LIE_energies.csv.",
            )],
        }
