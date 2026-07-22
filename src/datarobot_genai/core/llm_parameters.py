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

"""Framework-agnostic LiteLLM request parameter helpers.

Shared by dragent workflow YAML and the LangGraph / LlamaIndex / CrewAI
``get_*_llm()`` factories. For parsing reasoning **output** blocks from model
responses, see ``datarobot_genai.core.agents.reasoning``.
"""

from __future__ import annotations

import re
from typing import Any

# Matches OpenAI/Azure reasoning models (gpt-5, o-series). Avoids gpt-4o and similar.
_OPENAI_REASONING_MODEL_RE = re.compile(
    r"(?:^|/)(?:o[1-9](?:-mini|-preview)?|gpt-5)",
    re.IGNORECASE,
)

# OpenAI o-series reasoning models (o1, o3, o4-mini, ...) reject the
# parallel_tool_calls request param. gpt-5 is a reasoning model but accepts it,
# so it is intentionally excluded here (unlike _OPENAI_REASONING_MODEL_RE above).
_OPENAI_NO_PARALLEL_TOOL_CALLS_RE = re.compile(r"(?:^|/)o[1-9]", re.IGNORECASE)

_ANTHROPIC_SONNET_EXTRA_BODY = {
    "thinking": {"type": "enabled", "budget_tokens": 1024},
}
_ANTHROPIC_OPUS_EXTRA_BODY = {
    "thinking": {"type": "adaptive"},
}
_GEMINI_EXTRA_BODY = {
    "thinking_config": {"thinking_budget": 1024},
}
_OPENAI_EXTRA_BODY = {
    "reasoning_effort": "low",
}


def _normalize_model_name(model_name: str | None) -> str:
    return (model_name or "").removeprefix("datarobot/").lower()


def default_reasoning_extra_body(model_name: str | None) -> dict[str, Any]:
    """Return provider-specific ``extra_body`` for ``reasoning=True``."""
    normalized = _normalize_model_name(model_name)

    if "gemini" in normalized:
        return dict(_GEMINI_EXTRA_BODY)

    if "opus" in normalized:
        return dict(_ANTHROPIC_OPUS_EXTRA_BODY)

    if _OPENAI_REASONING_MODEL_RE.search(normalized):
        return dict(_OPENAI_EXTRA_BODY)

    return dict(_ANTHROPIC_SONNET_EXTRA_BODY)


def supports_parallel_tool_calls(model_name: str | None) -> bool:
    """Whether ``model_name`` accepts the ``parallel_tool_calls`` request param.

    OpenAI o-series reasoning models reject it; everything else accepts it.
    """
    return not _OPENAI_NO_PARALLEL_TOOL_CALLS_RE.search(_normalize_model_name(model_name))


def apply_reasoning_to_parameters(
    parameters: dict[str, Any] | None,
    *,
    reasoning: bool,
    model_name: str | None,
    explicit_extra_body: bool = False,
) -> dict[str, Any]:
    """Merge ``reasoning`` into LiteLLM ``parameters`` (``extra_body``, temperature).

    When ``explicit_extra_body`` is true (workflow YAML set ``extra_body``), or
    ``parameters`` already contains ``extra_body``, only temperature is cleared
    for ``reasoning=True``. Otherwise ``reasoning=True`` adds a provider default
    ``extra_body`` derived from ``model_name``.
    """
    params = dict(parameters or {})

    if explicit_extra_body or "extra_body" in params:
        if reasoning:
            params.pop("temperature", None)
        return params

    if reasoning:
        params.pop("temperature", None)
        params["extra_body"] = default_reasoning_extra_body(model_name)

    return params
