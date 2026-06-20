"""ensemble-server settings.

Inherits from bioagent_service.ServiceSettings for jobs_base_dir / NAS
conventions, then adds aggregator-specific config:
  - Per-method FC function connection (URL + function name + region)
  - Static API key allowlist (Phase 1 MVP; replaced by Tablestore in Phase 3)
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict

from bioagent_service import ServiceSettings


class FCMethodConfig(BaseModel):
    """Per-method connection to an underlying FC service."""

    function: str           # FC function name, e.g. "alphafold-server"
    region: str = "cn-hangzhou"
    http_base_url: str      # fcapp.run URL
    task_endpoint: str      # e.g. "/api/tasks/fold"
    enabled: bool = True
    timeout_seconds: int = 7200   # per-method polling timeout


class APIKeyConfig(BaseModel):
    """Static API key entry (Phase 1)."""

    key_id: str             # "ek_test_001"
    secret_hash: str        # sha256 hex digest of the secret
    customer_id: str
    plan: str = "internal"
    monthly_quota_calls: int = 1000


class EnsembleSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ENSEMBLE_",
        extra="ignore",
        env_nested_delimiter="__",
    )

    service_name: str = "ensemble"

    # FC OpenAPI credentials (the aggregator needs them to invoke other services).
    fc_access_key_id: str = ""
    fc_access_key_secret: str = ""

    # Per-method FC config — populated from env vars or .env, key = method name.
    # Example env: ENSEMBLE_FC_METHODS__ALPHAFOLD__FUNCTION=alphafold-server
    fc_methods: dict[str, FCMethodConfig] = Field(default_factory=dict)

    # Phase-1 hardcoded API keys.  Replaced by Tablestore in Phase 3.
    api_keys: list[APIKeyConfig] = Field(default_factory=list)

    # Cross-method default ranking metric for folding ensemble.
    folding_default_ranking_metric: str = "mean_plddt"
