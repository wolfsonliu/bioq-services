"""Pluggable dispatch backends behind the gateway.

`make_dispatcher(settings)` selects the backend from GATEWAY_DISPATCH_BACKEND:
`fc` (default, Alibaba FC async) or `http` (local submit/poll). New backends
(OpenFaaS, KEDA) plug in here without touching the app wiring.
"""

from __future__ import annotations

import httpx

from ..fc_status import FcStatusClient
from .base import Dispatcher, encode_form, stream_download
from .fc import SESSION_AFFINITY_HEADER, FCDispatcher
from .local import LocalHttpDispatcher
from .openfaas import OpenFaaSDispatcher

__all__ = [
    "SESSION_AFFINITY_HEADER",
    "Dispatcher",
    "FCDispatcher",
    "LocalHttpDispatcher",
    "OpenFaaSDispatcher",
    "encode_form",
    "make_dispatcher",
    "stream_download",
]


def make_dispatcher(settings) -> Dispatcher:
    timeout = settings.dispatch_timeout_sec
    if settings.dispatch_backend == "http":
        return LocalHttpDispatcher(httpx.Client(timeout=timeout))
    if settings.dispatch_backend == "openfaas":
        if not settings.openfaas_gateway_url:
            raise ValueError("GATEWAY_OPENFAAS_GATEWAY_URL is required for the openfaas backend")
        return OpenFaaSDispatcher(settings.openfaas_gateway_url, httpx.Client(timeout=timeout))
    if settings.dispatch_backend == "fc":
        fc = FcStatusClient(
            access_key_id=settings.ali_access_key_id,
            access_key_secret=settings.ali_access_key_secret,
            default_region=settings.oss_region,
            endpoint=settings.fc_endpoint,
        )
        return FCDispatcher(fc, httpx.Client(timeout=timeout))
    raise ValueError(f"unknown GATEWAY_DISPATCH_BACKEND: {settings.dispatch_backend!r}")
