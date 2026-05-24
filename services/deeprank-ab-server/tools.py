"""argv builder for deeprank-ab-server.

The upstream inference.py creates its workspace in CWD (`Path.cwd() /
identificator`). Since the framework's `submit(cwd=...)` resolves before
`build_argv` runs (so we can't pass a per-job cwd), we wrap the command in
`bash -c "cd <output_dir> && exec python ..."` to switch into the job's
output directory before launching the script.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from .models import ScoreRequest
from .settings import DeepRankAbSettings


def score_argv(
    req: ScoreRequest,
    *,
    job_dir: Path,
    pdb_path: Path,
    settings: DeepRankAbSettings,
) -> list[str]:
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    parts = [
        f"cd {shlex.quote(str(output_dir))}",
        "&&",
        "exec",
        shlex.quote(settings.python),
        shlex.quote(settings.inference_script),
        shlex.quote(str(pdb_path)),
        shlex.quote(req.heavy_chain_id),
        shlex.quote(req.light_chain_id),
        shlex.quote(req.antigen_chain_id),
    ]
    return ["bash", "-c", " ".join(parts)]
