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
"""Shared utilities for drtools and drmcp."""

import re
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel

from datarobot_genai.drmcputils.constants import MAX_INLINE_SIZE
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind

_OBJECT_ID_PATTERN = re.compile(r"^[0-9a-fA-F]{24}$")
_TRACE_ID_PATTERN = re.compile(r"^[0-9a-fA-F]{32}$")


def require_id(value: str, name: str) -> str:
    if not value or not value.strip():
        raise ToolError(
            f"Argument validation error: '{name}' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )
    return value.strip()


def require_object_id(value: str, name: str) -> str:
    """Validate a 24-character hex DataRobot object id (mirrors ``MongoIdField``).

    Raised up front so a malformed id fails with a readable message instead of
    a server 422.
    """
    if not isinstance(value, str):
        raise ToolError(
            f"Argument validation error: '{name}' must be a string, got {type(value).__name__}.",
            kind=ToolErrorKind.VALIDATION,
        )
    stripped = require_id(value, name)
    if not _OBJECT_ID_PATTERN.fullmatch(stripped):
        raise ToolError(
            f"Argument validation error: '{name}' must be a 24-character hex ID, got {value!r}.",
            kind=ToolErrorKind.VALIDATION,
        )
    return stripped


def require_trace_id(value: str) -> str:
    """Validate an exactly-32-character hex OTel trace id.

    Mirrors ``TracingRetrieveParamValidator.trace_id``.
    """
    if not isinstance(value, str):
        raise ToolError(
            f"Argument validation error: 'trace_id' must be a string, got {type(value).__name__}.",
            kind=ToolErrorKind.VALIDATION,
        )
    stripped = require_id(value, "trace_id")
    if not _TRACE_ID_PATTERN.fullmatch(stripped):
        raise ToolError(
            f"Argument validation error: 'trace_id' must be a 32-character hex ID, got {value!r}.",
            kind=ToolErrorKind.VALIDATION,
        )
    return stripped


def is_valid_url(url: str) -> bool:
    """Check if a string is a valid URL."""
    try:
        result = urlparse(url)
        return bool(result.scheme and result.netloc)
    except Exception:
        return False


class PredictionResponse(BaseModel):
    type: str
    data: str | None = None
    resource_id: str | None = None
    show_explanations: bool | None = None


def predictions_result_response(df: Any, show_explanations: bool = False) -> PredictionResponse:
    csv_str = df.to_csv(index=False)
    encoded_len = len(csv_str.encode("utf-8"))
    if encoded_len < MAX_INLINE_SIZE:
        return PredictionResponse(type="inline", data=csv_str, show_explanations=show_explanations)
    raise ToolError(
        f"Prediction CSV is {encoded_len} bytes, which exceeds the inline limit "
        f"of {MAX_INLINE_SIZE} bytes. "
        "Use batch prediction (for example predict_batch_predictions_from_dataset) "
        "for large outputs, "
        "or reduce rows or explanations.",
        kind=ToolErrorKind.VALIDATION,
    )
