"""gateway config CLI: `python -m server <cmd>` (container) / `python -m gateway ...` (repo).

    generate --target ecs|compose|openfaas|all [--write|--check]
        Render the complete per-target config file(s) from the schema.
        default: print to stdout; --write: overwrite deploy/config/gateway.<t>.env;
        --check: exit non-zero if the committed file differs (CI drift gate).
    config [--json]
        Print the EFFECTIVE resolved config of this process (secrets redacted).
    check
        Validate the effective config; exit non-zero on fatal problems.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import config_gen as gen
from . import config_spec as spec


def _cmd_generate(args) -> int:
    targets = spec.TARGETS if args.target == "all" else (args.target,)
    rc = 0
    for t in targets:
        text = gen.render_target(t)
        if args.write:
            path = gen.target_path(t)
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path}")
        elif args.check:
            path = gen.target_path(t)
            current = path.read_text(encoding="utf-8") if path.exists() else ""
            if current != text:
                print(f"DRIFT: {path} is out of date — run `make gen-config`", file=sys.stderr)
                rc = 1
            else:
                print(f"ok: {path}")
        else:
            if len(targets) > 1:
                print(f"# ---- {t} ----")
            print(text, end="")
    return rc


def _live_value(settings, key: str):
    if key.startswith("auth."):
        return getattr(settings.auth, key[len("auth."):])
    return getattr(settings, key)


def _cmd_config(args) -> int:
    from .settings import GatewaySettings
    settings = GatewaySettings()
    rows: list[tuple[str, str, str]] = []  # (env, value, origin)
    for _title, keys in spec.SECTIONS:
        for key in keys:
            env = gen.env_name(key)
            val = _live_value(settings, key)
            if key in spec.SECRETS:
                shown = "***set***" if val else "***unset***"
            elif key == "db_url" and "@" in str(val):
                shown = "***set (has password)***"
            else:
                shown = gen.fmt(val)
            origin = "default" if gen.fmt(val) == gen.fmt(gen._schema_default(key)) else "overridden"
            rows.append((env, shown, origin))
    if args.json:
        print(json.dumps({env: {"value": v, "origin": o} for env, v, o in rows}, indent=2))
    else:
        width = max(len(env) for env, _, _ in rows)
        for env, v, o in rows:
            print(f"{env:<{width}}  {v}   [{o}]")
    return 0


def _cmd_check(args) -> int:
    from .config_validate import validate_settings
    from .settings import GatewaySettings
    fatals, warnings = validate_settings(GatewaySettings())
    for w in warnings:
        print(f"WARNING: {w}")
    for f in fatals:
        print(f"FATAL: {f}", file=sys.stderr)
    if fatals:
        return 1
    print("config ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gateway", description="gateway config tooling")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="render per-target config file(s)")
    g.add_argument("--target", choices=(*spec.TARGETS, "all"), default="all")
    mode = g.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="overwrite the committed file(s)")
    mode.add_argument("--check", action="store_true", help="exit non-zero on drift")
    g.set_defaults(func=_cmd_generate)

    c = sub.add_parser("config", help="print effective resolved config (redacted)")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=_cmd_config)

    ck = sub.add_parser("check", help="validate effective config")
    ck.set_defaults(func=_cmd_check)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
