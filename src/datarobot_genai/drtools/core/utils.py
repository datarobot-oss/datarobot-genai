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

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel

from datarobot_genai.drmcputils.constants import MAX_INLINE_SIZE
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind


def require_id(value: str, name: str) -> str:
    if not value or not value.strip():
        raise ToolError(
            f"Argument validation error: '{name}' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )
    return value.strip()


def read_spec_file(path: Path) -> dict[str, Any] | None:
    """Try to read and parse a local YAML or JSON spec file; return None if not found."""
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(raw)
    return json.loads(raw)


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
