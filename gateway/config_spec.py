"""Declarative spec for generating the per-target gateway config files.

The generator (`config_gen`) reads this + the `GatewaySettings` schema to render
one COMPLETE, defaulted, commented non-secret env file per deploy target. Keeping
the doc comments here (rather than `Field(description=...)`) keeps `settings.py`
churn-free and lets a completeness test force every surfaced field to be
documented + explicitly kept-or-omitted (so a new knob can never go invisible).

Field keys: top-level = the settings field name; nested auth = "auth.<name>".
"""
from __future__ import annotations

TARGETS = ("ecs", "compose", "openfaas")

# Ordered (title, [field keys]) — the visible layout of the generated file.
SECTIONS: list[tuple[str, list[str]]] = [
    ("Execution / dispatch", [
        "dispatch_backend", "dispatch_timeout_sec", "openfaas_gateway_url",
        "thread_pool_size",
    ]),
    ("Storage", [
        "storage_backend", "file_base_dir", "oss_bucket", "oss_region",
        "presign_expiry_sec", "downstream_oss_mount",
    ]),
    ("Auth — VPC bypass / JWT / OIDC", [
        "auth.bypass_vpc", "auth.vpc_account_id", "auth.vpc_is_admin",
        "auth.jwt_jwks_url", "auth.jwt_issuer", "auth.jwt_audience",
        "auth.jwt_groups_claim", "auth.jwt_admin_group",
        "auth.jwt_jwks_cache_ttl_sec",
        "auth.oidc_issuer", "auth.oidc_client_id", "auth.oidc_client_secret",
    ]),
    ("Database", ["db_url"]),
    ("FC control-plane (Alibaba OpenAPI)", [
        "fc_endpoint", "ali_access_key_id", "ali_access_key_secret",
    ]),
    ("Advanced", ["registry_path", "jobs_base_dir", "port", "session_secret"]),
]

# Fields never surfaced (inherited worker-runner knobs the gateway doesn't act on,
# plus the `auth` container itself which is expanded field-by-field above).
OMIT: set[str] = {
    "auth",
    "uploads_base_dir", "oss_output_mount", "disk_limit_mb", "keep_alive_sec",
    "max_concurrent_jobs", "error_tail_chars", "keepalive_interval_s",
    "keepalive_url", "session_header_name", "task_endpoints_enabled",
    "task_job_id_header",
}

# Schema fields that carry secrets — rendered as commented placeholders, never a
# value. (db_url is handled per-target: see DB_URL below.)
SECRETS: set[str] = {
    "session_secret", "ali_access_key_id", "ali_access_key_secret",
    "auth.oidc_client_secret",
}

# External (non-schema) secret env names to list as commented placeholders at the
# end of each target's file, for discoverability. Read by the OSS SDK / postgres,
# not by GatewaySettings.
EXTERNAL_SECRETS: dict[str, list[str]] = {
    "ecs": ["OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET", "POSTGRES_PASSWORD"],
    "compose": [],
    "openfaas": [],
}

# Where each target's secrets live (for the "set in ..." hint).
SECRET_FILE: dict[str, str] = {
    "ecs": ".env", "compose": ".env", "openfaas": ".env.local",
}

# Values shared by all targets (fold-in of the old gateway.common.env).
COMMON: dict[str, object] = {
    "thread_pool_size": 100,
    "auth.jwt_audience": "gateway-server",
    "auth.jwt_groups_claim": "groups",
    "auth.jwt_admin_group": "bioq-admins",
}

# Per-target deltas from schema defaults (⊕ COMMON). Only non-default values.
PROFILES: dict[str, dict[str, object]] = {
    "ecs": {
        "dispatch_backend": "fc",
        "storage_backend": "oss",
        "oss_bucket": "bio-gateway",
        "oss_region": "cn-hangzhou",
        "downstream_oss_mount": "/mnt/oss",
        "auth.bypass_vpc": False,
        "auth.jwt_jwks_url":
            "http://keycloak:8080/realms/bioq/protocol/openid-connect/certs",
        "auth.oidc_client_id": "bioq-gateway",
    },
    "compose": {
        "dispatch_backend": "http",
        "storage_backend": "file",
        "file_base_dir": "/shared",
        "auth.bypass_vpc": True,
        "auth.jwt_jwks_url":
            "http://keycloak:8080/realms/bioq/protocol/openid-connect/certs",
        "auth.jwt_issuer": "http://localhost:8081/realms/bioq",
        "auth.oidc_issuer": "http://keycloak:8080/realms/bioq",
        "auth.oidc_client_id": "bioq-gateway",
    },
    "openfaas": {
        "dispatch_backend": "openfaas",
        "openfaas_gateway_url": "http://gateway.openfaas.svc.cluster.local:8080",
        "storage_backend": "file",
        "file_base_dir": "/shared",
        "jobs_base_dir": "/shared/gw_jobs",
        "registry_path": "/etc/bioq/services.yaml",
        "auth.bypass_vpc": True,
        "auth.jwt_jwks_url":
            "http://keycloak.bioq.svc.cluster.local:8080/realms/bioq/protocol/openid-connect/certs",
        "auth.jwt_issuer": "http://localhost:8081/realms/bioq",
        "auth.oidc_issuer":
            "http://keycloak.bioq.svc.cluster.local:8080/realms/bioq",
        "auth.oidc_client_id": "bioq-gateway",
    },
}

# db_url is per-target: a real value where safe (sqlite), else a commented note
# because it either carries a password (composed via compose interpolation) or is
# injected via a k8s Secret.
DB_URL: dict[str, str | None] = {
    "ecs": None,
    "compose": "sqlite:////data/gateway/gateway.db",
    "openfaas": None,
}
DB_URL_NOTE: dict[str, str] = {
    "ecs": "composed from POSTGRES_* in docker-compose.yml (interpolation); "
           "or set GATEWAY_DB_URL in .env for an external DB",
    "openfaas": "injected via the gateway-secrets Secret "
                "(built from POSTGRES_* in .env.local)",
}

# One-line docs per surfaced field key.
DOCS: dict[str, str] = {
    "dispatch_backend":
        "Execution backend: fc (Alibaba FC async) | http (in-process runner) | openfaas.",
    "dispatch_timeout_sec": "HTTP dispatch timeout to downstream (seconds).",
    "openfaas_gateway_url":
        "OpenFaaS gateway base URL (required when dispatch_backend=openfaas).",
    "thread_pool_size":
        "anyio threadpool size for the sync /v1 handlers (I/O-bound; can exceed vCPUs).",
    "storage_backend": "Upload/result storage: oss (presigned direct) | file (gateway-proxied).",
    "file_base_dir": "Shared-volume root for file storage (storage_backend=file).",
    "oss_bucket":
        "OSS bucket for presigned upload/download (storage_backend=oss). Must EXIST "
        "under the OSS AK below, in oss_region.",
    "oss_region": "OSS region (also the FC region for status polling).",
    "presign_expiry_sec": "Lifetime of presigned OSS URLs (seconds).",
    "downstream_oss_mount": "Path where downstream FC services mount the data-plane bucket.",
    "db_url": "SQLAlchemy URL for the user/job store.",
    "fc_endpoint":
        "FC OpenAPI endpoint override — a BARE host, e.g. "
        "<account-id>.cn-hangzhou-internal.fc.aliyuncs.com (no scheme, no path). "
        "Empty = SDK default {region}.fc.aliyuncs.com.",
    "ali_access_key_id": "FC OpenAPI access key id (env ALI_AK).",
    "ali_access_key_secret": "FC OpenAPI access key secret (env ALI_SK).",
    "registry_path": "Path to the downstream service registry (services.yaml).",
    "jobs_base_dir": "Root dir for per-job scratch.",
    "port": "HTTP listen port (usually driven by the image PORT env / compose).",
    "session_secret": "Signing key for the admin-console session cookie; set for multi-instance.",
    "auth.bypass_vpc":
        "Allow internal/localhost/*-vpc.fcapp.run callers to bypass JWT (break-glass). "
        "Set false for a public-facing gateway.",
    "auth.vpc_account_id": "Account id attributed to VPC-bypassed requests.",
    "auth.vpc_is_admin": "Treat VPC-bypassed callers as admin (internal ops).",
    "auth.jwt_jwks_url": "IdP JWKS URL for verifying Bearer tokens. Empty = JWT disabled.",
    "auth.jwt_issuer": "Expected token issuer (iss). Empty = don't validate iss.",
    "auth.jwt_audience": "Expected token audience (aud).",
    "auth.jwt_groups_claim": "Token claim carrying the user's groups.",
    "auth.jwt_admin_group": "Membership in this group ⇒ role=admin.",
    "auth.jwt_jwks_cache_ttl_sec": "How long to cache fetched JWKS (seconds).",
    "auth.oidc_issuer": "OIDC issuer for the admin-console SSO (Authorization Code). Empty = SSO off.",
    "auth.oidc_client_id": "OIDC client id for the admin console.",
    "auth.oidc_client_secret": "OIDC client secret for the admin console.",
}
