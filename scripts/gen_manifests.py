#!/usr/bin/env python3
"""Materialize static `describe` contracts (see
docs/specs/2026-08-20-describe-cold-start-static-manifest-design.md).

Modes:
  generate   `python scripts/gen_manifests.py`            all services in services.yaml
  check      `python scripts/gen_manifests.py --check`     regenerate to tmp + diff
  dump-one   internal leaf, called per service in its own venv

`generate`/`check` run the orchestrator in the *gateway* venv (which has
bioq-service-framework + pyyaml); for each service it re-enters that service's
own venv via `uv run --project services/<svc>-server`. `dump-one` registers the
service dir as the `server` package (same trick as the service tests' conftest)
and dumps build_manifest() + app.openapi().

Import side effect: each service's `app.py` runs `create_app()` at import time,
which does `settings.jobs_base_dir.mkdir(...)`. On read-only/immutable build
hosts that mkdir fails, so `dump_one` redirects `<PREFIX>JOBS_BASE_DIR` to a
throwaway dir for the import, then restores the declared default in the emitted
`nas_layout.jobs_base_dir` so the committed contract stays deterministic.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = Path(__file__).resolve()


def register_server_package(service_dir: Path) -> None:
    if "server" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        "server",
        service_dir / "__init__.py",
        submodule_search_locations=[str(service_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot build `server` package spec from {service_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["server"] = module
    spec.loader.exec_module(module)


def _service_settings_cls():
    """Find the service's ServiceSettings subclass in `server.settings`."""
    from bioq_service import ServiceSettings
    mod = importlib.import_module("server.settings")
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if obj is ServiceSettings:
            continue
        if issubclass(obj, ServiceSettings) and obj.__module__ == "server.settings":
            return obj
    raise RuntimeError("no ServiceSettings subclass found in server.settings")


@contextlib.contextmanager
def _writable_jobs_base_dir():
    settings_cls = _service_settings_cls()
    env_var = f"{settings_cls.model_config.get('env_prefix', '')}JOBS_BASE_DIR"
    default = settings_cls.model_fields["jobs_base_dir"].default
    with tempfile.TemporaryDirectory(prefix=".bioq-jobs-", dir=REPO_ROOT) as tmp:
        previous = os.environ.get(env_var)
        os.environ[env_var] = tmp
        try:
            yield default
        finally:
            if previous is None:
                os.environ.pop(env_var, None)
            else:
                os.environ[env_var] = previous


def dump_one(svc: str, service_dir: Path, out_dir: Path) -> None:
    register_server_package(service_dir)
    with _writable_jobs_base_dir() as default_jobs_base_dir:
        from bioq_service.manifest import build_manifest
        from server.app import adapter, app, settings

        manifest = build_manifest(app, adapter, settings)
        openapi = app.openapi()
    data = manifest.model_dump(mode="json")
    # create_app() mkdir'd the overridden jobs_base_dir at import; restore the
    # declared default so the committed manifest is host-independent.
    nas_layout = data.get("nas_layout")
    if isinstance(nas_layout, dict) and "jobs_base_dir" in nas_layout:
        nas_layout["jobs_base_dir"] = str(default_jobs_base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{svc}.manifest.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / f"{svc}.openapi.json").write_text(
        json.dumps(openapi, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def service_names() -> list[str]:
    from bioq_service.service_registry import load_services
    return sorted(load_services(REPO_ROOT / "services.yaml"))


def service_dir(svc: str) -> Path:
    return REPO_ROOT / "services" / svc


def render_one(svc: str, out_dir: Path) -> None:
    d = service_dir(svc)
    if not (d / "Dockerfile").is_file():
        raise SystemExit(f"missing service dir {d}")
    subprocess.run(
        ["uv", "run", "--project", str(d), "python", str(SCRIPT),
         "--dump-one", "--svc", svc, "--service-dir", str(d), "--out-dir", str(out_dir)],
        check=True,
    )


def generate() -> None:
    out_dir = REPO_ROOT / "manifests"
    for svc in service_names():
        print(f"gen-manifests: {svc}")
        render_one(svc, out_dir)


def check() -> int:
    rc = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for svc in service_names():
            committed_m = REPO_ROOT / "manifests" / f"{svc}.manifest.json"
            committed_o = REPO_ROOT / "manifests" / f"{svc}.openapi.json"
            if not committed_m.is_file() or not committed_o.is_file():
                print(f"check-manifests: MISSING manifest for {svc}", file=sys.stderr)
                rc = 1
                continue
            render_one(svc, tmp_dir)
            for kind, committed in (("manifest", committed_m), ("openapi", committed_o)):
                fresh = tmp_dir / f"{svc}.{kind}.json"
                if committed.read_text(encoding="utf-8") != fresh.read_text(encoding="utf-8"):
                    print(f"check-manifests: STALE {committed} (re-run make gen-manifests)",
                          file=sys.stderr)
                    rc = 1
    return rc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dump-one", action="store_true")
    parser.add_argument("--svc")
    parser.add_argument("--service-dir")
    parser.add_argument("--out-dir")
    args = parser.parse_args()

    if args.dump_one:
        dump_one(args.svc, Path(args.service_dir).resolve(), Path(args.out_dir))
        return 0
    if args.check:
        return check()
    generate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
