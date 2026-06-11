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
"""Reconstruct prior AG-UI turns as structured LangChain messages.

The default chat-history path flattens prior turns into the text ``{chat_history}``
prompt variable, which drops tool calls. This converter rebuilds real
``HumanMessage`` / ``AIMessage(tool_calls=...)`` / ``ToolMessage`` so the model sees
structured tool history. Reasoning, when carried, is already plain text in
``content`` (folded at ingest), so it needs no special handling here.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage

from datarobot_genai.core.agents.history import NormalizedHistoryMessage


def _field(obj: Any, key: str) -> Any:
    """Read a field from an AG-UI pydantic object or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _to_langchain_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    """Map AG-UI ToolCall objects/dicts to LangChain tool-call dicts.

    AG-UI shape: ``{id, function: {name, arguments(str)}}``. LangChain wants
    ``{id, name, args(dict), type}``; ``arguments`` is a JSON string that must be
    parsed to a dict (defaulting to ``{}`` on malformed/empty input).
    """
    result: list[dict[str, Any]] = []
    for tc in tool_calls or []:
        fn = _field(tc, "function") or {}
        raw_args = _field(fn, "arguments")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                args = {}
        else:
            args = raw_args
        if not isinstance(args, dict):  # langchain ToolCall.args must be a dict
            args = {}
        result.append(
            {
                "id": str(_field(tc, "id") or ""),
                "name": _field(fn, "name") or "",
                "args": args,
                "type": "tool_call",
            }
        )
    return result


def ag_ui_history_to_langchain(
    history: list[NormalizedHistoryMessage],
) -> list[BaseMessage]:
    """Convert normalized AG-UI history (from ``BaseAgent.history_messages``) to
    LangChain messages, preserving tool calls and tool results.
    """
    messages: list[BaseMessage] = []
    for msg in history:
        role = msg["role"]
        # content must be "" (never None) for tool-call-only assistant turns.
        content = msg.get("content") or ""
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(
                AIMessage(
                    content=content,
                    tool_calls=_to_langchain_tool_calls(msg.get("tool_calls")),
                )
            )
        elif role == "tool":
            messages.append(
                ToolMessage(content=content, tool_call_id=str(msg.get("tool_call_id") or ""))
            )
        elif role == "system":
            messages.append(SystemMessage(content=content))
    return messages
