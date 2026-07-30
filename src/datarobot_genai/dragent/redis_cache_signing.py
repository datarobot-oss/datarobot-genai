# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HMAC signing for Redis-backed dragent caches."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

_SIGNING_KEY_ENV = "AGENT_CARD_REGISTRY_REDIS_SIGNING_KEY"
_IDP_JWK_ENV = "IDP_AGENT_PRIVATE_KEY_JWK"
_SESSION_SECRET_ENV = "SESSION_SECRET_KEY"

_SIGNING_KEY_REQUIRED_MSG = (
    "Redis cache backends require a deployment-specific signing secret. Set "
    "AGENT_CARD_REGISTRY_REDIS_SIGNING_KEY, or replicate IDP_AGENT_PRIVATE_KEY_JWK "
    "or SESSION_SECRET_KEY for this deployment/workload."
)


def resolve_redis_signing_key(explicit: str | None = None) -> bytes:
    """Return the HMAC key used to sign Redis cache payloads."""
    for candidate in (
        explicit,
        os.getenv(_SIGNING_KEY_ENV),
        os.getenv(_IDP_JWK_ENV),
        os.getenv(_SESSION_SECRET_ENV),
    ):
        if candidate and candidate.strip():
            return hashlib.sha256(candidate.encode()).digest()
    raise ValueError(_SIGNING_KEY_REQUIRED_MSG)


def sign_redis_payload(payload: str, signing_key: bytes) -> str:
    """Return a hex HMAC-SHA256 signature for *payload*."""
    return hmac.new(signing_key, payload.encode(), hashlib.sha256).hexdigest()


def verify_redis_payload(payload: str, signature: str, signing_key: bytes) -> bool:
    """Return whether *signature* matches *payload*."""
    if not signature:
        return False
    expected = sign_redis_payload(payload, signing_key)
    return hmac.compare_digest(expected, signature)


def seal_redis_model(
    model: BaseModel, *, signing_key: bytes, signature_field: str = "signature"
) -> str:
    """Serialize *model*, sign the unsigned JSON, and return sealed JSON."""
    unsigned = model.model_dump_json(exclude={signature_field})
    signature = sign_redis_payload(unsigned, signing_key)
    sealed = model.model_copy(update={signature_field: signature})
    return sealed.model_dump_json()


def open_redis_model(
    model_type: type[T],
    payload: str,
    *,
    signing_key: bytes,
    signature_field: str = "signature",
) -> T | None:
    """Parse and verify a sealed Redis JSON payload."""
    try:
        parsed = model_type.model_validate_json(payload)
    except Exception:
        logger.warning("Redis cache entry failed validation; ignoring entry.")
        return None

    signature = getattr(parsed, signature_field, None)
    unsigned = parsed.model_dump_json(exclude={signature_field})
    if not verify_redis_payload(unsigned, signature or "", signing_key):
        logger.warning("Redis cache entry failed signature verification; ignoring entry.")
        return None
    return parsed
