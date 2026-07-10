"""API key secret hashing (MVP: sha256; bcrypt later)."""

from __future__ import annotations

import hashlib


def hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()
