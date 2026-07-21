"""Service-wide policy for haddock3-server."""

from __future__ import annotations

from pathlib import Path

from bioq_service import EndpointExample, JobAdapter

from .settings import Haddock3Settings


class Haddock3Adapter(JobAdapter):
    name = "haddock3"

    settings: Haddock3Settings  # narrow for IDEs

    def __init__(self, settings: Haddock3Settings) -> None:
        super().__init__(settings)

    # ---- Output detection ----

    def detect_outputs(self, job_dir: Path) -> bool:
        """Completed if output/ holds any non-empty, non-log artifact.

        Label-agnostic on purpose (the framework writes no per-job manifest):
        docking drops a whole `run/` tree, score writes `score.json`, restraints
        write a `.tbl`. `*.log` files are excluded so a job that only produced a
        log (i.e. crashed) is still flagged NO_OUTPUTS.
        """
        out = self.output_dir(job_dir)
        if not out.is_dir():
            return False
        for p in out.rglob("*"):
            if p.is_file() and p.suffix != ".log" and p.stat().st_size > 0:
                return True
        return False

    # ---- Subprocess env ----

    def subprocess_env(self) -> dict[str, str]:
        """Point haddock3 at the externally-staged CNS binary + force UTF-8.

        haddock3 discovers CNS via `CNS_EXEC` when no in-package binary exists.
        We only set it when the file is actually present so the env doesn't lie
        — CNS-free restraints jobs run fine without it.
        """
        env = {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
        cns = self.settings.cns_exec
        if cns.exists():
            env["CNS_EXEC"] = str(cns)
        return env

    def subprocess_cwd(self) -> Path | None:
        # The wrapper's `dock` subcommand chdirs into the staged input dir itself
        # (so molecule/.tbl basenames resolve); all other subcommands use
        # absolute paths. cwd here is not load-bearing.
        return None

    # ---- Manifest enrichments (agent protocol) ----

    def manifest_extras(self) -> dict:
        return {
            "model": {
                "name": "HADDOCK3",
                "paper": "Giulini et al., J. Chem. Inf. Model. 2025 "
                "(DOI:10.1021/acs.jcim.5c00969)",
                "method": "Integrative biomolecular docking (CNS-driven rigid-body "
                "sampling + flexible refinement), BonvinLab.",
                "task": "protein-protein / protein-ligand / peptide / nucleic-acid "
                "docking + HADDOCK scoring + restraint generation.",
                "engine": "CNS (license-gated, externalised to NAS at cns_exec). "
                "Restraints utilities are CNS-free.",
                "cpu_only": True,
            },
            "tool_outputs": {
                "dock": "output/run/<NN>_<module>/ step folders; final ranked "
                "models in the last module folder; CAPRI metrics in "
                "<NN>_caprieval/capri_ss.tsv (+ capri_clt.tsv when clustered).",
                "score": "output/score.json — haddock_score (+ vdw/elec/desolv/"
                "air/bsa components when full=true).",
                "restrain-bodies": "output/restraints.tbl — CNS distance restraints "
                "locking chains as rigid bodies.",
                "actpass-to-ambig": "output/ambig.tbl — ambiguous interaction "
                "restraints from two active/passive residue files.",
            },
            "config_tips": {
                "dock_config": "Supply only the workflow body ([module] sections + "
                "optional extra top-level keys). Do NOT set run_dir / molecules / "
                "mode / ncores — the service injects them. Reference uploaded "
                "molecules and .tbl files by bare filename.",
                "sampling": "protein-protein `sampling` defaults to 200 (vs upstream "
                "1000) for a faster/cheaper triage; raise for production runs.",
                "cns_required": "dock + score need CNS staged at cns_exec; check "
                "/healthz/detail cns_available before submitting.",
                "cpu_cost": "Docking is CPU-heavy — prefer a large-vCPU instance or "
                "the CLI batch mode on Slurm; keep sampling low on small instances.",
                "large_inputs": "Use *_uri (oss:// / job:// / file://) for PDBs that "
                "exceed the FC async 128 KiB multipart cap.",
            },
            "input_uri_schemes": {
                "upload": "multipart/form-data — molecule PDBs + .tbl / .actpass",
                "oss://<bucket>/<key>": "fetched at submit-time",
                "job://<prev_job_id>/<file>": "pipeline chaining",
                "file:///<path>": "absolute path on the FC NAS mount",
                "http(s)://...": "streamed at submit-time",
            },
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/dock/protein-protein": [
                EndpointExample(
                    title="two-body docking with ambiguous restraints",
                    curl=(
                        "curl -X POST $URL/api/dock/protein-protein "
                        "-F mol1=@receptor.pdb -F mol2=@ligand.pdb "
                        "-F ambig=@ambig.tbl -F sampling=200 -F top_models=4"
                    ),
                    notes="Needs CNS. Returns a JobInfo; poll /api/jobs/<id>, then "
                    "GET /api/jobs/<id>/files for the run/ tree + capri_ss.tsv.",
                ),
            ],
            "/api/dock": [
                EndpointExample(
                    title="general workflow runner",
                    curl=(
                        "curl -X POST $URL/api/dock "
                        "-F molecules=@e2a.pdb -F molecules=@hpr.pdb "
                        "-F config='[topoaa]\\n\\n[rigidbody]\\n"
                        "ambig_fname = \"air.tbl\"\\nsampling = 200\\n\\n[caprieval]\\n' "
                        "-F tbl=@air.tbl"
                    ),
                    notes="config is the workflow body only (no run_dir/molecules/"
                    "mode/ncores). Reference uploads by bare filename.",
                ),
            ],
            "/api/score": [
                EndpointExample(
                    title="HADDOCK-score a complex",
                    curl="curl -X POST $URL/api/score -F complex=@complex.pdb -F full=true",
                    notes="Needs CNS. Result in output/score.json.",
                ),
            ],
            "/api/restraints/restrain-bodies": [
                EndpointExample(
                    title="lock chains as rigid bodies (CNS-free)",
                    curl=(
                        "curl -X POST $URL/api/restraints/restrain-bodies "
                        "-F structure=@complex.pdb"
                    ),
                    notes="Fast, no CNS. Result in output/restraints.tbl.",
                ),
            ],
            "/api/restraints/active-passive-to-ambig": [
                EndpointExample(
                    title="ambig restraints from two actpass files (CNS-free)",
                    curl=(
                        "curl -X POST $URL/api/restraints/active-passive-to-ambig "
                        "-F actpass1=@a.actpass -F actpass2=@b.actpass "
                        "-F segid1=A -F segid2=B"
                    ),
                    notes="Fast, no CNS. Result in output/ambig.tbl.",
                ),
            ],
        }
