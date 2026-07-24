# Copyright 2026 DataRobot, Inc. and its affiliates.
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
"""Shared helpers for DataRobot NAT evaluators."""

from __future__ import annotations

import json
from typing import Any

from nat.data_models.evaluator import EvalInputItem


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def interactions_json_from_eval_item(item: EvalInputItem) -> str | None:
    """Return serialized pipeline interactions from the dataset row when present."""
    entry = item.full_dataset_entry
    if not isinstance(entry, dict):
        return None
    for key in ("pipeline_interactions", "pipelineInteractions"):
        raw = entry.get(key)
        if not raw:
            continue
        if isinstance(raw, str):
            return raw
        return json.dumps(raw)
    return None


def citations_from_eval_item(item: EvalInputItem) -> list[str]:
    """Extract retrieval context for faithfulness / guideline metrics."""
    entry = item.full_dataset_entry
    if not isinstance(entry, dict):
        return []

    context = entry.get("context")
    if context is not None:
        if isinstance(context, list):
            return [str(c) for c in context if c]
        return [str(context)]

    citations: list[str] = []
    for key, value in entry.items():
        if str(key).startswith("CITATION_CONTENT_") and value:
            citations.append(str(value))
    return citations
