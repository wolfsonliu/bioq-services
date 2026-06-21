"""ensemble-server settings.

Inherits from bioagent_service.ServiceSettings for jobs_base_dir / NAS
conventions, then adds aggregator-specific config:
  - Per-method FC function connection (URL + function name + region)
  - Multi-layer auth: VPC bypass + JWT verification + static API keys
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict

from bioagent_service import ServiceSettings


class FCMethodConfig(BaseModel):
    """Per-method connection to an underlying FC service."""

    function: str
    region: str = "cn-hangzhou"
    http_base_url: str
    task_endpoint: str
    enabled: bool = True
    timeout_seconds: int = 7200


class APIKeyConfig(BaseModel):
    """Static API key entry (Phase 1)."""

    key_id: str
    secret_hash: str
    customer_id: str
    plan: str = "internal"
    monthly_quota_calls: int = 1000


class AuthSettings(BaseModel):
    """Multi-layer auth configuration.

    See engineering/decisions/2026-06-21-ensemble-server-auth.md for the
    fallthrough chain (VPC bypass → JWT → API Key).
    """

    # ----- VPC bypass -----
    bypass_vpc: bool = True
    vpc_customer_id: str = "internal_vpc"

    # ----- JWT verification (disabled when jwt_jwks_url is empty) -----
    jwt_jwks_url: str = ""
    jwt_audience: str = "ensemble-server"
    jwt_issuer: str = ""               # empty = don't validate iss claim
    jwt_jwks_cache_ttl_sec: int = 3600
    jwt_sub_is_customer: bool = True   # if False, look up via jwt_sub_to_customer
    jwt_sub_to_customer: dict[str, str] = Field(default_factory=dict)


class EnsembleSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ENSEMBLE_",
        extra="ignore",
        env_nested_delimiter="__",
    )

    service_name: str = "ensemble"

    fc_access_key_id: str = ""
    fc_access_key_secret: str = ""

    fc_methods: dict[str, FCMethodConfig] = Field(default_factory=dict)

    # Auth: VPC bypass + JWT + static API keys.  api_keys kept as a top-level
    # field for env-var ergonomics (ENSEMBLE_API_KEYS__0__*) and Phase-1
    # backward compatibility; auth.* groups the new VPC/JWT knobs.
    auth: AuthSettings = Field(default_factory=AuthSettings)
    api_keys: list[APIKeyConfig] = Field(default_factory=list)

    folding_default_ranking_metric: str = "mean_plddt"
