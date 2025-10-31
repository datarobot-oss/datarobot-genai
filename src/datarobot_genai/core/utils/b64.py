# Copyright 2025 DataRobot, Inc. and its affiliates.
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

import base64
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_MAX_JSON_BYTES = 1024 * 1024  # 1 MiB


def encode_json_to_b64(
    data: Any,
    *,
    max_json_bytes: int | None = DEFAULT_MAX_JSON_BYTES,
    ensure_ascii: bool = False,
    separators: tuple[str, str] = (",", ":"),
) -> str | None:
    """
    Encode any JSON-serializable Python object to a URL-safe Base64 string.

    Args:
        data: JSON-serializable object to encode.
        max_json_bytes: Optional limit on JSON byte length to avoid oversized payloads.
        ensure_ascii: Passed to json.dumps.
        separators: Passed to json.dumps.

    Returns
    -------
        URL-safe Base64 string, or None if encoding/parsing fails or exceeds size.
    """
    try:
        json_str = json.dumps(data, separators=separators, ensure_ascii=ensure_ascii)
        json_bytes = json_str.encode("utf-8")
        if max_json_bytes is not None and len(json_bytes) > max_json_bytes:
            logger.debug("JSON byte size %d exceeds max %d", len(json_bytes), max_json_bytes)
            return None
        return base64.urlsafe_b64encode(json_bytes).decode("utf-8")
    except (TypeError, ValueError) as exc:
        logger.debug("Failed to encode value: %s", exc, exc_info=True)
        return None


def decode_b64_to_json(b64_str: str) -> dict[str, Any] | None:
    """
    Decode a Base64-encoded JSON string back to a Python dict.

    Returns
    -------
        dict or None: Decoded dict, or None if decoding/parsing fails.
    """
    if not b64_str or not isinstance(b64_str, str):
        return None

    b64_str = b64_str.strip()
    if not b64_str:
        return None

    padding_needed = len(b64_str) % 4
    if padding_needed:
        b64_str += "=" * (4 - padding_needed)

    try:
        decoded_bytes = base64.b64decode(b64_str)
        parsed = json.loads(decoded_bytes.decode("utf-8"))

        # Validate result type
        if not isinstance(parsed, dict):
            logger.debug("Decoded JSON is not an object: %r", parsed)
            return None

        return parsed
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.debug("Failed to decode value: %s", exc, exc_info=False)
        return None
