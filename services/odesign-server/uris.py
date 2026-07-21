"""odesign-server specific input helpers.

Generic URI resolution (upload / job:// / file:// / oss:// / http(s)://) lives in
`bioq_service.uris`; this module only holds ODesign's JSON ref_file rewriting.
"""

from __future__ import annotations

import json
from pathlib import Path


def rewrite_ref_files(json_path: Path, input_dir: Path) -> None:
    """Rewrite ref_file paths in JSON spec to absolute paths under input_dir."""
    with open(json_path) as f:
        samples = json.load(f)
    modified = False
    for sample in samples:
        ref = sample.get("ref_file")
        if ref:
            filename = Path(ref).name
            sample["ref_file"] = str(input_dir / filename)
            modified = True
    if modified:
        with open(json_path, "w") as f:
            json.dump(samples, f, indent=2)
