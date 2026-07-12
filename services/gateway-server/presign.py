"""Per-object OSS presigned URLs (alibabacloud-oss-v2).

MVP: sign with the gateway's own OSS credentials
(EnvironmentVariableCredentialsProvider — OSS_ACCESS_KEY_ID / _SECRET). Each
presigned URL is scoped to exactly one object key, so no STS is needed for
tenant isolation. Keys are per-user + content-addressed:
    users/<principal>/inputs/<sha256>/<filename>
"""

from __future__ import annotations

from datetime import timedelta

from .models import PresignResponse


def build_oss_client(region: str):
    import alibabacloud_oss_v2 as oss  # lazy: keeps import light + testable

    cfg = oss.config.load_default()
    cfg.credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()
    cfg.region = region
    return oss.Client(cfg)


class Presigner:
    def __init__(self, *, client, bucket: str, region: str, expiry_sec: int) -> None:
        self._client = client
        self._bucket = bucket
        self._region = region
        self._expiry = expiry_sec

    def input_key(self, principal: str, filename: str, sha256: str) -> str:
        return f"users/{principal}/inputs/{sha256}/{filename}"

    def uri_for(self, key: str) -> str:
        return f"oss://{self._bucket}/{key}"

    def _exists(self, key: str) -> bool:
        import alibabacloud_oss_v2 as oss
        try:
            self._client.head_object(oss.models.HeadObjectRequest(bucket=self._bucket, key=key))
            return True
        except Exception:  # noqa: BLE001 — any error (incl. NoSuchKey) => treat as absent
            return False

    def presign_put(self, principal: str, filename: str, sha256: str) -> PresignResponse:
        import alibabacloud_oss_v2 as oss
        key = self.input_key(principal, filename, sha256)
        uri = self.uri_for(key)
        if self._exists(key):
            return PresignResponse(uri=uri, exists=True, url=None)
        # `expires` = a timedelta duration. (`expiration` would be an absolute
        # datetime — passing a timedelta there raises inside the SDK.)
        result = self._client.presign(
            oss.models.PutObjectRequest(bucket=self._bucket, key=key),
            expires=timedelta(seconds=self._expiry),
        )
        return PresignResponse(uri=uri, exists=False, url=result.url)
