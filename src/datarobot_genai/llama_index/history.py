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
"""Reconstruct prior AG-UI turns as structured LlamaIndex ``ChatMessage`` objects.

Fed to ``AgentWorkflow.run(chat_history=...)`` so the model sees real tool calls and
tool results instead of the flattened ``{chat_history}`` text. Tool calls are carried
in ``additional_kwargs["tool_calls"]`` (OpenAI wire shape) rather than a
``ToolCallBlock``: the LiteLLM message adapter renders ``additional_kwargs`` verbatim
but does not serialize ``ToolCallBlock``. Reasoning, when carried, is plain text in
``content``.
"""

from __future__ import annotations

import json
from typing import Any

from llama_index.core.base.llms.types import ChatMessage

from datarobot_genai.core.agents.history import NormalizedHistoryMessage


def _field(obj: Any, key: str) -> Any:
    """Read a field from an AG-UI pydantic object or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _to_openai_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    """Map AG-UI ToolCall objects/dicts to OpenAI ``tool_calls`` dicts.

    OpenAI wire shape keeps ``arguments`` as a JSON *string* (not a parsed dict);
    a non-string/non-dict value falls back to ``"{}"``.
    """
    result: list[dict[str, Any]] = []
    for tc in tool_calls or []:
        fn = _field(tc, "function") or {}
        args = _field(fn, "arguments")
        if isinstance(args, dict):
            args = json.dumps(args)
        elif not isinstance(args, str):
            args = "{}"
        result.append(
            {
                "id": str(_field(tc, "id") or ""),
                "type": "function",
                "function": {"name": _field(fn, "name") or "", "arguments": args},
            }
        )
    return result


def ag_ui_history_to_chat_messages(
    history: list[NormalizedHistoryMessage],
) -> list[ChatMessage]:
    """Convert normalized AG-UI history (from ``BaseAgent.history_messages``) to
    LlamaIndex ``ChatMessage`` objects, preserving tool calls and tool results.
    """
    messages: list[ChatMessage] = []
    for msg in history:
        role = msg["role"]
        content = msg.get("content") or ""
        if role == "user":
            messages.append(ChatMessage(role="user", content=content))
        elif role == "assistant":
            tool_calls = _to_openai_tool_calls(msg.get("tool_calls"))
            additional_kwargs = {"tool_calls": tool_calls} if tool_calls else {}
            messages.append(
                ChatMessage(role="assistant", content=content, additional_kwargs=additional_kwargs)
            )
        elif role == "tool":
            messages.append(
                ChatMessage(
                    role="tool",
                    content=content,
                    additional_kwargs={"tool_call_id": str(msg.get("tool_call_id") or "")},
                )
            )
        elif role == "system":
            messages.append(ChatMessage(role="system", content=content))
    return messages
