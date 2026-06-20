"""API key verification for ensemble-server.

Phase-1 MVP uses SHA256 of the presented secret to match against the
allowlist's stored hash.  Phase 3 will replace this with bcrypt + Tablestore
storage.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from ..settings import APIKeyConfig


def verify_api_key(presented: str, allowlist: list[APIKeyConfig]) -> Optional[APIKeyConfig]:
    """Match `presented` against `allowlist` by SHA256 hash.

    Returns the matching APIKeyConfig on success, None on miss / empty input.
    """
    if not presented:
        return None
    presented_hash = hashlib.sha256(presented.encode("utf-8")).hexdigest()
    for entry in allowlist:
        if entry.secret_hash == presented_hash:
            return entry
    return None
