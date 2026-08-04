"""Per-object OSS presigned URLs (alibabacloud-oss-v2).

MVP: sign with the gateway's own OSS credentials
(EnvironmentVariableCredentialsProvider — OSS_ACCESS_KEY_ID / _SECRET). Each
presigned URL is scoped to exactly one object key, so no STS is needed for
tenant isolation. Keys are job-centric: input key
users/<account_id>/<job_id>/input/<filename>, output key
users/<account_id>/<job_id>/<filename>.
"""

from __future__ import annotations

from datetime import timedelta

from .models import UploadTarget


def _oss_error_info(exc: BaseException) -> tuple[int | None, str | None]:
    """Best-effort (http_status, error_code) from an alibabacloud-oss-v2 error.

    The SDK wraps the real error inside OperationError / RequestError (via
    ``.unwrap()``); a ServiceError carries ``status_code`` + ``code`` (a missing
    object is 404 / NoSuchKey), while a transport RequestError has no status.
    """
    cur: BaseException | None = exc
    for _ in range(4):
        if cur is None:
            break
        status = getattr(cur, "status_code", None)
        code = getattr(cur, "code", None)
        if isinstance(status, int) and status:
            return status, code
        unwrap = getattr(cur, "unwrap", None)
        cur = unwrap() if callable(unwrap) else (cur.__cause__ or cur.__context__)
    return None, None


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

    def input_key(self, account_id: str, job_id: str, filename: str) -> str:
        return f"users/{account_id}/{job_id}/input/{filename}"

    def output_key(self, account_id: str, job_id: str, filename: str) -> str:
        return f"users/{account_id}/{job_id}/{filename}"

    def uri_for(self, key: str) -> str:
        return f"oss://{self._bucket}/{key}"

    def _exists(self, key: str) -> bool:
        import alibabacloud_oss_v2 as oss
        try:
            self._client.head_object(oss.models.HeadObjectRequest(bucket=self._bucket, key=key))
            return True
        except Exception as exc:  # noqa: BLE001
            # Only a genuine "object absent" (HTTP 404 / NoSuchKey) means False.
            # Any OTHER error — auth (403), OSS 5xx, throttling, transport
            # (RequestError, no status), or a NoSuchBucket misconfig — must
            # propagate so callers never mistake an OSS outage for a missing
            # object (which would silently mask the failure).
            status, code = _oss_error_info(exc)
            if status == 404 and code != "NoSuchBucket":
                return False
            raise

    def prepare_upload(self, account_id: str, job_id: str, filename: str,
                       sha256: str | None = None) -> UploadTarget:
        import alibabacloud_oss_v2 as oss
        key = self.input_key(account_id, job_id, filename)
        uri = self.uri_for(key)
        if self._exists(key):
            return UploadTarget(uri=uri, exists=True, put_url=None)
        result = self._client.presign(
            oss.models.PutObjectRequest(bucket=self._bucket, key=key),
            expires=timedelta(seconds=self._expiry),
        )
        return UploadTarget(uri=uri, exists=False, put_url=result.url)

    def result_url_if_exists(self, account_id: str, job_id: str, filename: str) -> str | None:
        import alibabacloud_oss_v2 as oss
        key = self.output_key(account_id, job_id, filename)
        if not self._exists(key):
            return None
        result = self._client.presign(
            oss.models.GetObjectRequest(bucket=self._bucket, key=key),
            expires=timedelta(seconds=self._expiry),
        )
        return result.url
