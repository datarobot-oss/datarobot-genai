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
"""Per-framework message converters for multi-turn conversation support.

Converts ag_ui Message types into each agent framework's native message
format so that LLMs see proper structured tool_call/tool_result history.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ag_ui.core import Message

logger = logging.getLogger(__name__)


def truncate_messages(
    messages: list[Message],
    max_history: int,
    *,
    exclude_current: bool = False,
) -> list[Message]:
    """Truncate messages to keep the last user message + at most max_history prior messages.

    If max_history <= 0, returns only the last user message (no history).
    Ensures truncation does not orphan tool messages from their assistant tool_call.

    When *exclude_current* is ``True`` only the history portion is returned
    (everything before the last user message).  This is useful for frameworks
    that pass the current turn separately (e.g. CrewAI kickoff inputs,
    LlamaIndex ``user_msg`` parameter).
    """
    if not messages:
        return []

    # Find the last user message
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if getattr(messages[i], "role", None) == "user":
            last_user_idx = i
            break

    if last_user_idx is None:
        if exclude_current:
            return messages[-max_history:] if max_history > 0 else []
        return messages[-max_history:] if max_history > 0 else []

    history = messages[:last_user_idx]
    current = messages[last_user_idx:]

    if max_history <= 0:
        return [] if exclude_current else list(current)

    if len(history) > max_history:
        history = history[-max_history:]

    # Drop orphan tool messages at the start of history (no preceding assistant tool_call)
    while history and getattr(history[0], "role", None) == "tool":
        history = history[1:]

    # Drop leading assistant whose tool_calls have no matching tool results
    if history and getattr(history[0], "role", None) == "assistant":
        tc = getattr(history[0], "tool_calls", None)
        if tc:
            tool_ids = {t.id for t in tc}
            result_ids = {
                getattr(m, "tool_call_id", None)
                for m in history[1:]
                if getattr(m, "role", None) == "tool"
            }
            if not tool_ids.issubset(result_ids):
                history = history[1:]

    if exclude_current:
        return history
    return history + list(current)


# ---------------------------------------------------------------------------
# LangGraph / LangChain
# ---------------------------------------------------------------------------


def to_langchain_messages(
    messages: list[Message],
) -> list[Any]:
    """Convert ag_ui messages to LangChain BaseMessage types.

    Mapping:
    - UserMessage     -> HumanMessage
    - AssistantMessage -> AIMessage (with tool_calls if present)
    - ToolMessage     -> LCToolMessage
    - SystemMessage   -> LCSystemMessage
    """
    from langchain_core.messages import AIMessage
    from langchain_core.messages import HumanMessage
    from langchain_core.messages import SystemMessage as LCSystemMessage
    from langchain_core.messages import ToolMessage

    result: list[Any] = []
    for msg in messages:
        role = getattr(msg, "role", None)
        content = getattr(msg, "content", None) or ""

        if role == "user":
            result.append(HumanMessage(content=content))
        elif role == "assistant":
            tool_calls_raw = getattr(msg, "tool_calls", None)
            lc_tool_calls = []
            if tool_calls_raw:
                for tc in tool_calls_raw:
                    args_str = tc.function.arguments
                    try:
                        args = json.loads(args_str)
                    except (json.JSONDecodeError, TypeError):
                        args = {"_raw": args_str}
                    lc_tool_calls.append({"name": tc.function.name, "args": args, "id": tc.id})
            result.append(AIMessage(content=content, tool_calls=lc_tool_calls))
        elif role == "tool":
            tool_call_id = getattr(msg, "tool_call_id", "")
            result.append(ToolMessage(content=content, tool_call_id=tool_call_id))
        elif role == "system":
            result.append(LCSystemMessage(content=content))
        else:
            logger.debug("Skipping unknown message role: %s", role)

    return result


# ---------------------------------------------------------------------------
def _summarize_tool_calls_with_args(tool_calls: list[Any]) -> str:
    """Render tool calls with function names and arguments for text-based frameworks."""
    parts = []
    for tc in tool_calls:
        fn = getattr(tc, "function", None)
        if fn:
            name = getattr(fn, "name", "unknown")
            args = getattr(fn, "arguments", "{}")
            parts.append(f"{name}({args})")
        else:
            parts.append("unknown()")
    return "[Called tools: " + ", ".join(parts) + "]"


# ---------------------------------------------------------------------------
# NAT
# ---------------------------------------------------------------------------


def to_nat_messages(
    messages: list[Message],
) -> list[Any]:
    """Convert ag_ui messages to NAT Message types.

    NAT's Message model only supports user/assistant/system roles.
    Tool messages are mapped as system messages (injected context) since NAT
    has no tool role. Assistant tool_calls are serialized as text.
    """
    from nat.data_models.api_server import Message as NatMessage
    from nat.data_models.api_server import UserMessageContentRoleType

    result: list[Any] = []
    for msg in messages:
        role = getattr(msg, "role", None)
        content = getattr(msg, "content", None) or ""

        if role == "user":
            result.append(NatMessage(content=content, role=UserMessageContentRoleType.USER))
        elif role == "assistant":
            tool_calls_raw = getattr(msg, "tool_calls", None)
            if tool_calls_raw and not content:
                content = _summarize_tool_calls_with_args(tool_calls_raw)
            elif tool_calls_raw and content:
                content = f"{content}\n{_summarize_tool_calls_with_args(tool_calls_raw)}"
            result.append(NatMessage(content=content, role=UserMessageContentRoleType.ASSISTANT))
        elif role == "system":
            result.append(NatMessage(content=content, role=UserMessageContentRoleType.SYSTEM))
        elif role == "tool":
            tool_call_id = getattr(msg, "tool_call_id", "unknown")
            result.append(
                NatMessage(
                    content=f"[Tool result for {tool_call_id}]: {content}",
                    role=UserMessageContentRoleType.SYSTEM,
                )
            )
        else:
            logger.debug("Skipping unknown message role for NAT: %s", role)

    return result


# ---------------------------------------------------------------------------
# LlamaIndex
# ---------------------------------------------------------------------------


def to_llama_index_messages(
    messages: list[Message],
) -> list[Any]:
    """Convert ag_ui messages to LlamaIndex ChatMessage types.

    Tool call metadata is stored in additional_kwargs.
    """
    from llama_index.core.llms import ChatMessage
    from llama_index.core.llms import MessageRole

    result: list[Any] = []
    for msg in messages:
        role = getattr(msg, "role", None)
        content = getattr(msg, "content", None) or ""

        if role == "user":
            result.append(ChatMessage(role=MessageRole.USER, content=content))
        elif role == "assistant":
            additional_kwargs: dict[str, Any] = {}
            tool_calls_raw = getattr(msg, "tool_calls", None)
            if tool_calls_raw:
                additional_kwargs["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls_raw
                ]
            result.append(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=content,
                    additional_kwargs=additional_kwargs,
                )
            )
        elif role == "tool":
            tool_call_id = getattr(msg, "tool_call_id", "")
            result.append(
                ChatMessage(
                    role=MessageRole.TOOL,
                    content=content,
                    additional_kwargs={"tool_call_id": tool_call_id},
                )
            )
        elif role == "system":
            result.append(ChatMessage(role=MessageRole.SYSTEM, content=content))
        else:
            logger.debug("Skipping unknown message role for LlamaIndex: %s", role)

    return result


# ---------------------------------------------------------------------------
# CrewAI
# ---------------------------------------------------------------------------


def to_crewai_chat_messages(
    messages: list[Message],
) -> list[dict[str, str]]:
    """Convert ag_ui messages to CrewAI crew_chat_messages format.

    Returns list of {"role": str, "content": str} dicts.
    Tool calls are summarized as text since CrewAI flattens to plain text.
    """
    result: list[dict[str, str]] = []
    for msg in messages:
        role = getattr(msg, "role", None) or "user"
        content = getattr(msg, "content", None) or ""

        if role == "assistant":
            tool_calls_raw = getattr(msg, "tool_calls", None)
            if tool_calls_raw and not content:
                content = _summarize_tool_calls_with_args(tool_calls_raw)
            elif tool_calls_raw and content:
                content = f"{content}\n{_summarize_tool_calls_with_args(tool_calls_raw)}"

        if role == "tool":
            tool_call_id = getattr(msg, "tool_call_id", "unknown")
            content = f"[Tool result for {tool_call_id}]: {content}"

        result.append({"role": role, "content": content})

    return result
