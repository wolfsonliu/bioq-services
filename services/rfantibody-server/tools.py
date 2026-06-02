"""argv builders for the three RFantibody tools.

Each function returns the subprocess argv that an endpoint hands to
`JobRunner.submit`. They never spawn anything — the framework owns subprocess
lifecycle (`SubprocessRunner.run`).

Output paths are hard-coded relative to a job's `output/` directory to match the
on-disk contract exposed by other endpoints (e.g., `job://<id>/1_rfdiffusion.qv`).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .models import ProteinMPNNRequest, RF2Request, RFdiffusionRequest
from .settings import RFantibodySettings

logger = logging.getLogger(__name__)

# Conventional output filenames. Order matters: `RFantibodyAdapter.detect_outputs`
# also reads from this list to recognise a job that finished successfully.
RFDIFFUSION_OUTPUT = "1_rfdiffusion.qv"
PROTEINMPNN_OUTPUT = "2_proteinmpnn.qv"
RF2_OUTPUT = "3_rf2.qv"
ALL_OUTPUT_FILENAMES = (RFDIFFUSION_OUTPUT, PROTEINMPNN_OUTPUT, RF2_OUTPUT)


def _weight_path(tool: str, settings: RFantibodySettings) -> Path:
    """Convention: weights/<TOOL>.pt. Falls back to a generic name if the tool isn't known."""
    known = {
        "rfdiffusion": "RFdiffusion_Ab.pt",
        "proteinmpnn": "ProteinMPNN_v48_noise_0.2.pt",
        "rf2": "RF2_ab.pt",
    }
    return settings.weights_dir / known.get(tool, f"{tool}.pt")


def _maybe_append_weights(cmd: list[str], tool: str, flag: str, settings: RFantibodySettings) -> None:
    """Append `flag=<weight>` (or `[flag, weight]`) iff the checkpoint exists.

    Missing weights are logged but not fatal — the underlying script may have
    its own default. This matches the legacy `_check_weights` behavior.
    """
    weights = _weight_path(tool, settings)
    if not weights.exists():
        logger.warning("weights missing for %s at %s; relying on script default", tool, weights)
        return
    if "=" in flag:
        cmd.append(f"{flag}{weights}")
    else:
        cmd.extend([flag, str(weights)])


def rfdiffusion_argv(
    req: RFdiffusionRequest,
    target_pdb: Path,
    framework_pdb: Path,
    job_dir: Path,
    settings: RFantibodySettings,
) -> list[str]:
    """Build the `rfdiffusion_inference.py` argv (Hydra config overrides)."""
    output_qv = job_dir / "output" / RFDIFFUSION_OUTPUT
    script = settings.scripts_dir / "rfdiffusion_inference.py"

    cmd = [
        sys.executable, str(script),
        "--config-name", "antibody",
        f"hydra.run.dir={job_dir / 'hydra'}",
        f"antibody.target_pdb={target_pdb.resolve()}",
        f"antibody.framework_pdb={framework_pdb.resolve()}",
        f"inference.quiver={output_qv}",
        f"inference.num_designs={req.num_designs}",
        f"diffuser.T={req.diffuser_t}",
        f"inference.final_step={req.final_step}",
    ]

    loops_list = [x.strip() for x in req.design_loops.split(",")]
    cmd.append(f"antibody.design_loops=[{','.join(loops_list)}]")

    if req.hotspots:
        hs = [h.strip() for h in req.hotspots.split(",")]
        cmd.append(f"ppi.hotspot_res=[{','.join(hs)}]")

    _maybe_append_weights(cmd, "rfdiffusion", "inference.ckpt_override_path=", settings)

    if req.deterministic:
        cmd.append("inference.deterministic=True")
    if req.no_trajectory:
        cmd.append("inference.write_trajectory=False")

    return cmd


def proteinmpnn_argv(
    req: ProteinMPNNRequest,
    input_quiver: Path,
    job_dir: Path,
    settings: RFantibodySettings,
) -> list[str]:
    """Build the `proteinmpnn_interface_design.py` argv (flag-style CLI)."""
    output_qv = job_dir / "output" / PROTEINMPNN_OUTPUT
    script = settings.scripts_dir / "proteinmpnn_interface_design.py"

    cmd = [
        sys.executable, str(script),
        "-quiver", str(input_quiver.resolve()),
        "-outquiver", str(output_qv),
        "-loop_string", req.loops,
        "-seqs_per_struct", str(req.seqs_per_struct),
        "-temperature", str(req.temperature),
        "-omit_AAs", req.omit_aas,
    ]

    _maybe_append_weights(cmd, "proteinmpnn", "-checkpoint_path", settings)

    if req.deterministic:
        cmd.append("-deterministic")

    return cmd


def rf2_argv(
    req: RF2Request,
    input_quiver: Path,
    job_dir: Path,
    settings: RFantibodySettings,
) -> list[str]:
    """Build the `rf2_predict.py` argv (Hydra config overrides)."""
    output_qv = job_dir / "output" / RF2_OUTPUT
    script = settings.scripts_dir / "rf2_predict.py"

    cmd = [
        sys.executable, str(script),
        f"hydra.run.dir={job_dir / 'hydra'}",
        f"input.quiver={input_quiver.resolve()}",
        f"output.quiver={output_qv}",
        f"inference.num_recycles={req.num_recycles}",
        f"inference.hotspot_show_proportion={req.hotspot_show_prop}",
        "inference.cautious=False",
    ]

    _maybe_append_weights(cmd, "rf2", "model.model_weights=", settings)

    if req.seed is not None:
        cmd.append(f"+inference.seed={req.seed}")

    return cmd
