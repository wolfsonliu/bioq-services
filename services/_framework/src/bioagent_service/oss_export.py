"""Output-sink: mirror a completed job's NAS dir to a mounted OSS path.

When a downstream FC service has the data-plane OSS bucket mounted at
`settings.oss_output_mount` (default /mnt/oss) AND the gateway passed an OSS
prefix (users/<principal>/<job_id>/), the framework copies the whole NAS job
dir (minus input/, already uploaded by the client) to <mount>/<prefix> and
writes results.zip of the output dir. No mount or no prefix => no-op. This lets
the gateway serve downloads straight from OSS without invoking the downstream.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


def _zip_dir(src: Path, dest_zip: Path) -> None:
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(src))


def mirror_job_dir_to_oss(
    *,
    job_dir: Path,
    output_dir: Path,
    oss_prefix: str,
    mount: str,
    skip: tuple[str, ...] = ("input",),
) -> str | None:
    """Copy `job_dir` (minus `skip`) to `<mount>/<oss_prefix>` + write results.zip.

    Returns the destination dir path, or None if skipped (no prefix / no mount).
    """
    if not oss_prefix:
        return None
    mount_path = Path(mount)
    if not mount_path.is_dir():
        return None
    dest = mount_path / oss_prefix.strip("/")
    dest.mkdir(parents=True, exist_ok=True)
    for child in sorted(job_dir.iterdir()):
        if child.name in skip:
            continue
        target = dest / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)
    if output_dir.is_dir() and any(output_dir.iterdir()):
        _zip_dir(output_dir, dest / "results.zip")
    return str(dest)
