"""HADDOCK3 config (.cfg) builders — pure text, no upstream import.

HADDOCK3 configs are TOML-like but allow repeated section headers (e.g.
`[caprieval]` twice), which is invalid TOML — so we emit text directly rather
than round-tripping a dict through a TOML lib. Keeping these functions free of
`haddock` imports lets them be unit-tested in the plain dev env.
"""

from __future__ import annotations

import re
from pathlib import Path

# Top-level keys the service always controls; stripped from a caller-supplied
# general workflow body before we prepend our own header.
_MANAGED_TOP_KEYS = ("run_dir", "molecules", "mode", "ncores")


def _quote(s: str) -> str:
    """TOML-quote a string path/value (backslashes + quotes escaped)."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _molecules_block(molecules: list[str]) -> str:
    inner = ",\n".join(f"    {_quote(m)}" for m in molecules)
    return f"molecules = [\n{inner}\n]"


def _header(*, run_dir: str, ncores: int, molecules: list[str]) -> str:
    return "\n".join([
        f"run_dir = {_quote(run_dir)}",
        'mode = "local"',
        f"ncores = {int(ncores)}",
        _molecules_block(molecules),
        "",
    ])


def build_protein_protein_cfg(
    *,
    molecules: list[str],
    run_dir: str,
    ncores: int,
    sampling: int,
    do_flexref: bool,
    do_emref: bool,
    clustering: bool,
    top_models: int,
    ambig_fname: str | None = None,
    reference_fname: str | None = None,
) -> str:
    """Assemble a canonical two-body protein-protein docking workflow.

    topoaa -> rigidbody -> [flexref] -> [emref] -> [clustfcc -> seletopclusts]
    -> caprieval. All file paths should be absolute (self-contained; no cwd
    dependence).
    """
    lines: list[str] = [_header(run_dir=run_dir, ncores=ncores, molecules=molecules)]

    lines.append("[topoaa]\n")

    rb = ["[rigidbody]", f"sampling = {int(sampling)}"]
    if ambig_fname:
        rb.append(f"ambig_fname = {_quote(ambig_fname)}")
    lines.append("\n".join(rb) + "\n")

    if do_flexref:
        fr = ["[flexref]"]
        if ambig_fname:
            fr.append(f"ambig_fname = {_quote(ambig_fname)}")
        lines.append("\n".join(fr) + "\n")

    if do_emref:
        er = ["[emref]"]
        if ambig_fname:
            er.append(f"ambig_fname = {_quote(ambig_fname)}")
        lines.append("\n".join(er) + "\n")

    if clustering:
        lines.append("[clustfcc]\n")
        lines.append(f"[seletopclusts]\ntop_models = {int(top_models)}\n")

    cap = ["[caprieval]"]
    if reference_fname:
        cap.append(f"reference_fname = {_quote(reference_fname)}")
    lines.append("\n".join(cap) + "\n")

    return "\n".join(lines)


def _strip_managed_top_keys(body: str) -> str:
    """Remove any top-level run_dir/molecules/mode/ncores the caller set.

    Only lines *before the first section header* are considered top-level. A
    `molecules = [ ... ]` array may span multiple lines, so we drop through the
    closing bracket.
    """
    out: list[str] = []
    in_sections = False
    skipping_array = False
    for line in body.splitlines():
        if skipping_array:
            if "]" in line:
                skipping_array = False
            continue
        stripped = line.lstrip()
        if not in_sections and stripped.startswith("["):
            in_sections = True
        if not in_sections:
            m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=", stripped)
            if m and m.group(1) in _MANAGED_TOP_KEYS:
                if m.group(1) == "molecules" and "]" not in line:
                    skipping_array = True
                continue
        out.append(line)
    return "\n".join(out)


def finalize_general_cfg(
    config_body: str, *, molecules: list[str], run_dir: str, ncores: int,
) -> str:
    """Prepend a service-controlled header to a caller-supplied workflow body.

    The caller supplies only the workflow (module sections + optional extra
    top-level keys), referencing uploaded molecules / .tbl files by bare
    filename (resolved against the staged input dir via the wrapper's chdir).
    """
    cleaned = _strip_managed_top_keys(config_body).strip("\n")
    header = _header(run_dir=run_dir, ncores=ncores, molecules=molecules)
    return f"{header}\n{cleaned}\n"


def write_cfg(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
