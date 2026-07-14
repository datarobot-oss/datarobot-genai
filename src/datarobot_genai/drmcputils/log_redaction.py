# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared secret-redaction patterns for MCP server logs.

The user-MCP logging formatter (``drmcp.core.logging``) and global-mcp's logging
filter apply the same redaction, so the canonical pattern list and helper live
here and both import them.
"""

import re

REDACTED = "[REDACTED]"

# Applied in order by ``redact_secrets`` (whole match -> "[REDACTED]").  Deliberately
# targeted: a previous catch-all for any 20+-char alphanumeric string also redacted
# ObjectIds, request/trace ids and class names -- making logs unusable while still
# missing secrets that contain ``-`` or ``_``.
SECRET_PATTERNS = [
    r"sk-[A-Za-z0-9_-]+",  # OpenAI-style keys (incl. sk-proj-…, any length)
    r"AKIA[0-9A-Z]{16}",  # AWS Access Key pattern
    r"((?!\.)[\w\-_.]*[^.])(@\w+)(\.\w+(\.\w+)?[^.\W])",  # Email
    # JWTs — three base64url segments (DataRobot/Okta access + S2S tokens)
    r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
    # Authorization header values: "Bearer <token>" / "Basic <base64>".  The value
    # must be credential-shaped — contain a digit/base64 symbol, or 10+ chars of
    # mixed case — so prose like "Basic authentication disabled" survives.  The
    # (?i:) scope keeps the shape checks case-sensitive.
    r"(?i:\b(?:bearer|basic))\s+"
    r"(?:(?=\S*[0-9+/=])[A-Za-z0-9._~+/=-]{6,}"
    r"|(?=\S*[a-z])(?=\S*[A-Z])[A-Za-z0-9._~+/=-]{10,})",
    # Assignments of secret-shaped keys, incl. prefixed forms: token=…, api_key: …,
    # DATAROBOT_API_TOKEN=…, client_secret=…, X-DataRobot-Authorization: …
    # The stem must END the key (``tokens=1500`` survives) and the value must be
    # credential-shaped: not a bare number (``completion_token=300`` survives) and
    # not a short lowercase diagnostic word (``api_key: missing`` survives).
    r"(?i:\b[\w-]*(?:token|secret|password|passwd|pwd|api[_-]?key|apikey|"
    r"access[_-]?key|authorization|credentials?))\b\s*[=:]\s*"
    r"(?!\d+\b)(?:(?=\S*[0-9+/=])\S{6,}|(?=\S*[a-z])(?=\S*[A-Z])\S{10,}|\S{16,})",
]

_COMPILED_SECRET_PATTERNS = [re.compile(pattern) for pattern in SECRET_PATTERNS]


def redact_secrets(text: str) -> str:
    """Replace every secret-shaped substring in ``text`` with ``[REDACTED]``."""
    for pattern in _COMPILED_SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text
