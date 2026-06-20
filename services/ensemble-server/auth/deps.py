"""FastAPI dependency for API key authentication.

Routes use `Depends(require_api_key)` to require + identify the caller.
The dependency returns the matched APIKeyConfig so the route handler can
attribute the request to the customer_id.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, Request

from ..settings import APIKeyConfig
from .api_key import verify_api_key


def require_api_key(
    request: Request,
    x_api_key: str = Header(...),
) -> APIKeyConfig:
    """Validate the X-API-Key header against settings.api_keys.

    Raises HTTPException(401) if the header is missing or doesn't match.
    """
    settings = request.app.state.settings
    key = verify_api_key(x_api_key, settings.api_keys)
    if key is None:
        raise HTTPException(401, "invalid or missing X-API-Key")
    return key
