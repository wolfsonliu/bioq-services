"""gateway-server settings.

Inherits bioq_service.ServiceSettings (jobs_base_dir / NAS conventions),
adds: auth (VPC bypass + JWT), a SQLite DB URL, an OSS bucket/region for
presign, the downstream service registry path, and an HTTP dispatch timeout.
API keys live in the DB (not settings) — see db/.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict

from bioq_service import ServiceSettings


class AuthSettings(BaseModel):
    """VPC bypass + JWT verification knobs (ENSEMBLE parity, account-based)."""

    bypass_vpc: bool = True
    vpc_account_id: str = "internal_vpc"
    vpc_is_admin: bool = True   # VPC bypass 身份是否直接视为 admin（内网运维）

    jwt_jwks_url: str = ""                 # empty = JWT disabled
    jwt_audience: str = "gateway-server"
    jwt_issuer: str = ""                   # empty = don't validate iss
    jwt_jwks_cache_ttl_sec: int = 3600


class GatewaySettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        env_file=".env",
        extra="ignore",
        env_nested_delimiter="__",
    )

    jobs_base_dir: Path = Field(default=Path("/data/gateway_jobs"))
    task_endpoints_enabled: bool = False   # gateway has no /api/tasks of its own

    # user/credential + job DB. Default is single-file SQLite (single instance).
    # For HA / multi-instance, point GATEWAY_DB_URL at cloud PostgreSQL, e.g.
    #   postgresql+psycopg://<user>:<pw>@<host>:5432/<db>?sslmode=require
    # The store auto-applies connection-pool tuning for non-sqlite URLs.
    db_url: str = Field(default="sqlite:////data/gateway/gateway.db")

    # storage backend: "oss" (presigned direct-to-object) or "file" (gateway
    # /v1/files IO over a shared volume — local Compose/K8s). Env:
    # GATEWAY_STORAGE_BACKEND / GATEWAY_FILE_BASE_DIR.
    storage_backend: str = Field(default="oss")
    file_base_dir: Path = Field(default=Path("/shared"))

    # OSS for presigned upload/download
    oss_bucket: str = Field(default="bioagent-inputs")
    oss_region: str = Field(default="cn-hangzhou")
    presign_expiry_sec: int = Field(default=900, ge=60)
    downstream_oss_mount: str = Field(default="/mnt/oss")

    # downstream service registry (svc -> ServiceRecord)
    registry_path: Path = Field(default=Path("/opt/gateway/services.yaml"))

    # downstream HTTP dispatch
    dispatch_timeout_sec: float = Field(default=60.0, ge=5)

    # execution backend: "fc" (Alibaba FC async task mode), "http" (plain
    # submit/poll against each service's own in-process runner — local
    # Compose/K8s), or "openfaas" (async via the OpenFaaS gateway). Env:
    # GATEWAY_DISPATCH_BACKEND.
    dispatch_backend: str = Field(default="fc")

    # OpenFaaS gateway base URL (dispatch_backend="openfaas"), e.g.
    # http://gateway.openfaas:8080. Env: GATEWAY_OPENFAAS_GATEWAY_URL.
    openfaas_gateway_url: str = Field(default="")

    # anyio threadpool capacity. All /v1 handlers are sync `def` → FastAPI runs
    # them in the threadpool (anyio default = 40 tokens). Gateway work is
    # I/O-bound proxy work (httpx to downstream/FC + OSS + SQLite) — threads sit
    # blocked on sockets (GIL released), so this can exceed the vCPU count. On
    # the current 4-vCPU ECS host, 100 lifts concurrency well past the default 40
    # without much memory cost, and matches httpx's default 100-connection pool
    # (so threads don't just queue on the client pool). Env: GATEWAY_THREAD_POOL_SIZE.
    thread_pool_size: int = Field(default=100, ge=8)

    # FC OpenAPI (GetAsyncTask status polling). AK/SK read from ALI_AK/ALI_SK
    # (FCDispatcher convention). When unset, the gateway falls back to HTTP
    # status polling. fc_endpoint overrides "{region}.fc.aliyuncs.com" (e.g. a
    # VPC/internal endpoint) when the ECS host lacks public egress to FC OpenAPI.
    ali_access_key_id: str = Field(default="", validation_alias="ALI_AK")
    ali_access_key_secret: str = Field(default="", validation_alias="ALI_SK")
    fc_endpoint: str = Field(default="")

    auth: AuthSettings = Field(default_factory=AuthSettings)
