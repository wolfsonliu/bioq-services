"""CLI batch-mode entry point for bioagent services.

Lets the same Docker image run as either an HTTP service (``uvicorn``, the
default CMD) or a one-shot batch job (``python -m server <endpoint> ...``).
The CLI path reuses each service's ``tools.py`` argv builders, ``adapter.py``
output detection, and ``SubprocessRunner`` — but skips FastAPI, the thread pool,
the job store, MCP, and FC keepalive entirely.

Typical Slurm usage::

    apptainer exec --nv image.sif python -m server score \\
        --model /data/model.pdb --native /data/native.pdb \\
        --output-dir /scratch/$SLURM_JOB_ID/

Framework contract:

    Each service writes a ``__main__.py`` that instantiates its settings +
    adapter, builds a dict of ``CLIEndpoint`` descriptors, and calls
    ``create_cli(adapter, settings, endpoints)``.  The framework handles
    argparse construction (auto-derived from the pydantic request model),
    input-file resolution, subprocess execution, output detection, and exit-code
    mapping.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from bioagent_service.adapter import JobAdapter
from bioagent_service.runner import SubprocessRunner
from bioagent_service.settings import ServiceSettings

logger = logging.getLogger(__name__)

BuildCLIArgv = Callable[
    [BaseModel, dict[str, Path], Path, ServiceSettings],
    list[str],
]
"""Per-endpoint callback: (parsed_request, input_paths, job_dir, settings) → argv."""


@dataclass
class CLIEndpoint:
    """Describes one CLI sub-command backed by a service endpoint.

    ``name``
        Sub-command name (e.g. ``"score"``).
    ``help``
        One-line description shown in ``--help``.
    ``request_model``
        The pydantic model whose fields become CLI flags.
    ``build_argv``
        Callback that receives the validated request, a dict of resolved input
        file paths, the job directory, and settings — returns the subprocess
        argv.  Typically a thin wrapper around the service's ``tools.py``
        function.
    ``inputs``
        Mapping of input-file argument names to ``(help_text, required)``.
        Each becomes a ``--<name>`` flag that accepts a local file path.
    """

    name: str
    help: str
    request_model: type[BaseModel]
    build_argv: BuildCLIArgv
    inputs: dict[str, tuple[str, bool]] = field(default_factory=dict)


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Return (inner_type, is_optional) for a type annotation."""
    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0], True
    return annotation, False


_UNSET = object()


def _add_model_args(
    parser: argparse.ArgumentParser,
    model: type[BaseModel],
    *,
    skip: frozenset[str] = frozenset(),
) -> None:
    """Generate argparse flags from a pydantic model's fields.

    All flags default to ``_UNSET`` so that ``_build_request`` can distinguish
    "user passed this flag" from "not passed" — letting ``--params-json``
    values and pydantic model defaults take effect when a flag is omitted.
    Boolean flags are the exception: they use store_true/store_false and
    default to ``_UNSET`` via ``argparse.SUPPRESS`` (not present in namespace
    unless the flag is used).
    """
    for field_name, field_info in model.model_fields.items():
        if field_name in skip:
            continue
        flag = f"--{field_name.replace('_', '-')}"
        annotation = field_info.annotation
        if annotation is None:
            continue

        inner_type, is_optional = _unwrap_optional(annotation)
        default = field_info.default
        if default is PydanticUndefined:
            default = None
        help_text = _field_help(field_info, default)

        if inner_type is bool:
            if default is True:
                parser.add_argument(
                    f"--no-{field_name.replace('_', '-')}",
                    dest=field_name,
                    action="store_false",
                    default=_UNSET,
                    help=help_text,
                )
            else:
                parser.add_argument(
                    flag,
                    dest=field_name,
                    action="store_true",
                    default=_UNSET,
                    help=help_text,
                )
        elif inner_type is int:
            parser.add_argument(flag, dest=field_name, type=int, default=_UNSET, help=help_text)
        elif inner_type is float:
            parser.add_argument(flag, dest=field_name, type=float, default=_UNSET, help=help_text)
        else:
            parser.add_argument(flag, dest=field_name, type=str, default=_UNSET, help=help_text)


def _field_help(info: FieldInfo, default: Any) -> str:
    """Build a help string from the field's description and default."""
    parts: list[str] = []
    if info.description:
        parts.append(info.description)
    if default is not None:
        parts.append(f"(default: {default})")
    return " ".join(parts) if parts else ""


def _build_request(
    model: type[BaseModel],
    namespace: argparse.Namespace,
    params_json: str | None,
) -> BaseModel:
    """Construct a pydantic request from CLI args, optionally merged with JSON.

    Priority: explicit CLI flag > params_json > pydantic model default.
    """
    data: dict[str, Any] = {}

    if params_json is not None:
        if Path(params_json).is_file():
            data = json.loads(Path(params_json).read_text(encoding="utf-8"))
        else:
            data = json.loads(params_json)

    for field_name in model.model_fields:
        val = getattr(namespace, field_name, _UNSET)
        if val is not _UNSET:
            data[field_name] = val

    return model.model_validate(data)


def _resolve_inputs(
    namespace: argparse.Namespace,
    input_specs: dict[str, tuple[str, bool]],
) -> dict[str, Path]:
    """Resolve input file paths from the parsed namespace."""
    resolved: dict[str, Path] = {}
    for name, (_help, required) in input_specs.items():
        raw = getattr(namespace, name, None)
        if raw is None:
            if required:
                print(f"error: --{name.replace('_', '-')} is required", file=sys.stderr)
                sys.exit(2)
            continue
        path = Path(raw)
        if not path.exists():
            print(f"error: input file not found: {path}", file=sys.stderr)
            sys.exit(2)
        resolved[name] = path.resolve()
    return resolved


def create_cli(
    adapter: JobAdapter,
    settings: ServiceSettings,
    endpoints: dict[str, CLIEndpoint],
    *,
    version: str = "0.0.0",
) -> None:
    """Build an argparse CLI from endpoint descriptors and run one job.

    This is the main entry point for each service's ``__main__.py``.  It parses
    ``sys.argv``, builds the subprocess argv via the matched endpoint's
    ``build_argv`` callback, runs it synchronously, and exits with the
    subprocess return code (or 1 if outputs are missing).
    """
    parser = argparse.ArgumentParser(
        prog=f"python -m server",
        description=f"{adapter.name} — CLI batch mode",
    )
    parser.add_argument("--version", action="version", version=version)

    subparsers = parser.add_subparsers(dest="endpoint", help="Endpoint to run")

    for ep in endpoints.values():
        sub = subparsers.add_parser(ep.name, help=ep.help)

        sub.add_argument(
            "--output-dir", type=str, default="./output",
            help="Output directory (default: ./output)",
        )
        sub.add_argument(
            "--params-json", type=str, default=None,
            help="JSON file or inline string with request parameters",
        )
        sub.add_argument(
            "--write-job-json", action="store_true", default=False,
            help="Write job.json status sidecar to output-dir",
        )
        sub.add_argument(
            "--json", dest="json_output", action="store_true", default=False,
            help="Print result as JSON to stdout",
        )
        sub.add_argument(
            "--log-file", type=str, default=None,
            help="Subprocess log path (default: <output-dir>/run.log)",
        )

        for input_name, (input_help, required) in ep.inputs.items():
            sub.add_argument(
                f"--{input_name.replace('_', '-')}",
                dest=input_name,
                type=str,
                required=required,
                help=input_help,
            )

        _add_model_args(sub, ep.request_model, skip=frozenset(ep.inputs))

    args = parser.parse_args()

    if args.endpoint is None:
        parser.print_help()
        sys.exit(2)

    ep = endpoints[args.endpoint]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    job_dir = output_dir.parent if output_dir.name == "output" else output_dir
    actual_output = job_dir / "output"
    actual_output.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_file) if args.log_file else job_dir / "logs" / "run.log"

    request = _build_request(ep.request_model, args, args.params_json)
    inputs = _resolve_inputs(args, ep.inputs)

    argv = ep.build_argv(request, inputs, job_dir, settings)
    if not argv:
        print("error: build_argv returned an empty argv", file=sys.stderr)
        sys.exit(1)

    env = adapter.subprocess_env()
    cwd = adapter.subprocess_cwd()

    logger.info("running: %s", " ".join(argv))
    start = time.monotonic()
    rc = SubprocessRunner.run(argv, log_path, env=env, cwd=cwd)
    elapsed = time.monotonic() - start

    has_outputs = adapter.detect_outputs(job_dir)

    result = {
        "status": "completed" if (rc == 0 and has_outputs) else "failed",
        "return_code": rc,
        "has_outputs": has_outputs,
        "duration_seconds": round(elapsed, 2),
        "output_dir": str(actual_output),
        "log_file": str(log_path),
    }

    if rc != 0:
        result["failure_reason"] = "subprocess_error"
    elif not has_outputs:
        result["failure_reason"] = "no_outputs"

    if args.write_job_json:
        job_json_path = job_dir / "job.json"
        job_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.json_output:
        print(json.dumps(result, indent=2))
    else:
        status = result["status"]
        dur = result["duration_seconds"]
        if status == "completed":
            logger.info("completed in %.1fs — outputs in %s", dur, actual_output)
        else:
            reason = result.get("failure_reason", "unknown")
            logger.error("failed (%s, rc=%d) in %.1fs — log: %s", reason, rc, dur, log_path)

    sys.exit(0 if result["status"] == "completed" else rc if rc != 0 else 1)


__all__ = ["BuildCLIArgv", "CLIEndpoint", "create_cli"]
