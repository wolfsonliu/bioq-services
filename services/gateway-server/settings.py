"""gateway-server settings.

Inherits bioagent_service.ServiceSettings (jobs_base_dir / NAS conventions),
adds: auth (VPC bypass + JWT), a SQLite DB URL, an OSS bucket/region for
presign, the downstream service registry path, and an HTTP dispatch timeout.
API keys live in the DB (not settings) — see db/.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict

from bioagent_service import ServiceSettings


class AuthSettings(BaseModel):
    """VPC bypass + JWT verification knobs (ENSEMBLE parity, principal-based)."""

    bypass_vpc: bool = True
    vpc_principal: str = "internal_vpc"

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

    # user/credential + job DB
    db_url: str = Field(default="sqlite:////data/gateway/gateway.db")

    # OSS for presigned upload/download
    oss_bucket: str = Field(default="bioagent-inputs")
    oss_region: str = Field(default="cn-hangzhou")
    presign_expiry_sec: int = Field(default=900, ge=60)

    # downstream service registry (svc -> ServiceRecord)
    registry_path: Path = Field(default=Path("/opt/gateway/services.yaml"))

    # downstream HTTP dispatch
    dispatch_timeout_sec: float = Field(default=60.0, ge=5)

    # FC OpenAPI (GetAsyncTask status polling). AK/SK read from ALI_AK/ALI_SK
    # (FCDispatcher convention). When unset, the gateway falls back to HTTP
    # status polling. fc_endpoint overrides "{region}.fc.aliyuncs.com" (e.g. a
    # VPC/internal endpoint) when the ECS host lacks public egress to FC OpenAPI.
    ali_access_key_id: str = Field(default="", validation_alias="ALI_AK")
    ali_access_key_secret: str = Field(default="", validation_alias="ALI_SK")
    fc_endpoint: str = Field(default="")

    auth: AuthSettings = Field(default_factory=AuthSettings)
