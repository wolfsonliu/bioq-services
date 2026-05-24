"""JWT token distribution service.

Lightweight FastAPI app that issues RS256-signed JWT tokens using a
pre-generated RSA key pair. Designed for Alibaba Cloud FC deployment.

Endpoints:
  POST /api/token              — Issue a signed JWT
  GET  /.well-known/jwks.json  — Public JWKS for verification
  GET  /healthz                — Health check
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import jwt
from cryptography.hazmat.primitives import serialization
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

_HERE = Path(__file__).resolve().parent
_JWKS_PATH = _HERE / "jwks.json"
_PRIVATE_KEYS_DIR = _HERE / "private_keys"

_DEFAULT_KID = os.getenv("JWT_KID", "self")
_DEFAULT_EXPIRES_HOURS = float(os.getenv("JWT_DEFAULT_EXPIRES_HOURS", "24"))
_API_KEY: str | None = os.getenv("JWT_API_KEY")


def _load_private_key(kid: str):
    key_path = _PRIVATE_KEYS_DIR / f"{kid}.key"
    pem = key_path.read_bytes()
    return serialization.load_pem_private_key(pem, password=None)


_private_key = _load_private_key(_DEFAULT_KID)
_jwks: dict = json.loads(_JWKS_PATH.read_text())

app = FastAPI(title="JWT Token Service", version="0.0.1")


# -- Models ------------------------------------------------------------------

class TokenRequest(BaseModel):
    sub: str = Field(..., description="Subject — who the token is issued to")
    expires_in_hours: Optional[float] = Field(
        None, description=f"Token lifetime in hours (default: {_DEFAULT_EXPIRES_HOURS})"
    )
    claims: Optional[dict[str, Any]] = Field(
        None, description="Extra claims merged into the JWT payload"
    )


class TokenResponse(BaseModel):
    token: str
    expires_at: str
    kid: str


# -- Auth ---------------------------------------------------------------------

def _check_api_key(x_api_key: Optional[str] = Header(None)):
    if _API_KEY and x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# -- Endpoints ----------------------------------------------------------------

@app.post("/api/token", response_model=TokenResponse)
def issue_token(req: TokenRequest, _=Depends(_check_api_key)):
    """Issue a signed JWT with the requested claims."""
    now = datetime.now(timezone.utc)
    exp = now + timedelta(hours=req.expires_in_hours or _DEFAULT_EXPIRES_HOURS)

    payload: dict[str, Any] = {
        "sub": req.sub,
        "iat": now,
        "exp": exp,
    }
    if req.claims:
        payload.update(req.claims)

    token = jwt.encode(
        payload, _private_key, algorithm="RS256",
        headers={"kid": _DEFAULT_KID},
    )

    return TokenResponse(
        token=token,
        expires_at=exp.isoformat(),
        kid=_DEFAULT_KID,
    )


@app.get("/.well-known/jwks.json")
def jwks_endpoint():
    """Serve the public JWKS for token verification."""
    return _jwks


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
