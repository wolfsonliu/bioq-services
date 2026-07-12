"""Service registry — load `services/services.yaml` (svc -> ServiceRecord).

General-purpose utilities (production code may resolve downstream service URLs
via these), so this module has no test-only dependencies.
`bioagent_service.fc_testing` re-exports the helpers for backward compatibility.

`services/services.yaml` schema::

    services:
      <name>-server:
        url: https://fc-...-vpc.fcapp.run   # required — VPC HTTP trigger URL
        region: cn-hangzhou                 # default: cn-hangzhou
        tier: warm                          # hot | warm | cold (default: warm)
        function: fc-...                    # optional — FC function name (for
                                            #   FC OpenAPI / GetAsyncTask)
        gpu: fc.gpu.tesla.1                 # optional — GPU card class

Only `url` is required. `function` / `gpu` are optional and may be omitted until
the values are known (they live in the Aliyun FC console, not the repo).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class ServiceRecord(BaseModel):
    """One downstream service's deployment metadata."""

    model_config = ConfigDict(extra="forbid")

    url: str
    region: str = "cn-hangzhou"
    tier: str = "warm"            # hot | warm | cold
    function: str | None = None   # FC function name (optional)
    gpu: str | None = None        # GPU card class (optional)


def find_services_yaml(start: Path | None = None) -> Path:
    """Walk up from `start` (or cwd) until `services/services.yaml` is found.

    Raises FileNotFoundError if not found — typically means the caller is
    running from outside the bioagent repo.
    """
    base = (start or Path.cwd()).resolve()
    for candidate in [base, *base.parents]:
        target = candidate / "services" / "services.yaml"
        if target.is_file():
            return target
    raise FileNotFoundError(f"services/services.yaml not found above {base!r}")


def load_services(
    path: Path | None = None, *, start: Path | None = None
) -> dict[str, "ServiceRecord"]:
    """Load `services.yaml` into a `{service_name: ServiceRecord}` dict.

    Pass `path` to read a specific file, or `start` to anchor the upward walk
    (e.g. `Path(__file__)` from a test). Trailing slashes on `url` are stripped
    so callers can do `f"{url}/api/..."`.
    """
    yaml_path = Path(path) if path is not None else find_services_yaml(start=start)
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    raw = data.get("services") or {}
    out: dict[str, ServiceRecord] = {}
    for name, rec in raw.items():
        rec = dict(rec or {})
        if isinstance(rec.get("url"), str):
            rec["url"] = rec["url"].rstrip("/")
        out[name] = ServiceRecord(**rec)
    return out


def fc_url(service_name: str, *, start: Path | None = None) -> str:
    """Resolve the deployed URL for `service_name` from `services.yaml`.

    `start` anchors the upward walk — pass `Path(__file__)` from a test.
    """
    services = load_services(start=start)
    if service_name not in services:
        raise KeyError(
            f"service {service_name!r} not in services.yaml; "
            f"known services: {sorted(services)}"
        )
    return services[service_name].url


__all__ = ["ServiceRecord", "fc_url", "find_services_yaml", "load_services"]
