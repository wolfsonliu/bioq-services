"""FC control-plane job status via GetAsyncTask.

Polling an async job's status over HTTP (`GET /api/jobs/<id>`) cold-starts a
downstream FC *function instance* on every poll — against scale-to-zero GPU
services that means an instance storm and unreliable status. `GetAsyncTask`
hits the FC **control plane** instead: reliable, and it spins no function
instance. Requires AK/SK + the FC function name (from services.yaml).

Self-contained (no dependency on pipelines/) so the gateway image stays lean;
mirrors the proven logic in pipelines/framework/fc_dispatcher.py.
"""

from __future__ import annotations

# FC async task state -> gateway job status string (JobView.status).
_FC_STATUS_MAP: dict[str, str] = {
    "Enqueued": "pending",
    "Dequeued": "pending",
    "Running": "running",
    "Retrying": "running",
    "Stopping": "running",
    "Succeeded": "completed",
    "Failed": "failed",
    "Stopped": "failed",
}


class FcStatusClient:
    """Resolve async-task status via FC OpenAPI GetAsyncTask.

    `enabled` is False when no AK/SK (and no injected client) is configured, so
    the gateway can fall back to HTTP status polling.
    """

    def __init__(
        self,
        *,
        access_key_id: str = "",
        access_key_secret: str = "",
        default_region: str = "cn-hangzhou",
        endpoint: str = "",
        client=None,
    ) -> None:
        self._ak = access_key_id
        self._sk = access_key_secret
        self._default_region = default_region
        self._endpoint = endpoint  # override; else "{region}.fc.aliyuncs.com"
        self._injected = client  # for tests: a pre-built FC client
        self._clients: dict[str, object] = {}

    @property
    def enabled(self) -> bool:
        return self._injected is not None or bool(self._ak and self._sk)

    def _client(self, region: str):
        if self._injected is not None:
            return self._injected
        if region not in self._clients:
            from alibabacloud_fc20230330.client import Client as FCClient
            from alibabacloud_tea_openapi.models import Config as FCConfig

            cfg = FCConfig(
                access_key_id=self._ak,
                access_key_secret=self._sk,
                endpoint=self._endpoint or f"{region}.fc.aliyuncs.com",
            )
            self._clients[region] = FCClient(cfg)
        return self._clients[region]

    def get_status(self, *, function: str, task_id: str, region: str | None = None) -> str:
        """Return a gateway status string for the async task, via GetAsyncTask."""
        from alibabacloud_fc20230330 import models as fc_models

        client = self._client(region or self._default_region)
        resp = client.get_async_task(
            function_name=function,
            task_id=task_id,
            request=fc_models.GetAsyncTaskRequest(qualifier="LATEST"),
        )
        fc_state = getattr(resp.body, "status", None) or ""
        return _FC_STATUS_MAP.get(fc_state, "running")
