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
"""Chat history extraction and normalization utilities.

This module provides helpers for extracting and summarizing prior chat
messages from ``RunAgentInput`` so that agent templates can inject
conversation history into their prompts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from typing import TypedDict

from ag_ui.core import RunAgentInput


class _NormalizedHistoryMessageRequired(TypedDict):
    """Required fields for a normalized prior chat message."""

    role: str
    content: str


class NormalizedHistoryMessage(_NormalizedHistoryMessageRequired, total=False):
    """Normalized representation of a single prior chat message.

    This structure is intentionally minimal but preserves enough optional
    metadata for tool-heavy agents to reconstruct richer history when needed.

    Required fields:
    - role: str
    - content: str

    Optional fields (best-effort, may be absent):
    - tool_call_id: str | None
    - name: str | None
    - tool_calls: Any | None  # e.g. OpenAI-style tool_calls payload
    """

    tool_call_id: str | None
    name: str | None
    tool_calls: Any | None


def _get_message_field(message: Any, key: str, default: Any = None) -> Any:
    """Best-effort field access for pydantic-like objects or dict messages."""
    if isinstance(message, Mapping):
        return message.get(key, default)
    return getattr(message, key, default)


def _summarize_tool_calls(tool_calls: Any) -> str:
    """Render a minimal, stable summary for tool-call-only assistant messages."""
    if not tool_calls:
        return "[tool_calls]"

    names: list[str] = []
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            name: Any = None
            if isinstance(tc, Mapping):
                fn = tc.get("function")
                if isinstance(fn, Mapping):
                    name = fn.get("name")
                name = name or tc.get("name") or tc.get("tool_name")
            else:
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", None) if fn is not None else None
                name = name or getattr(tc, "name", None) or getattr(tc, "tool_name", None)
            if name:
                names.append(str(name))

    if names:
        return "[tool_calls] " + ", ".join(names)
    return "[tool_calls]"


def extract_history_messages(
    run_agent_input: RunAgentInput,
    max_history: int,
) -> list[NormalizedHistoryMessage]:
    r"""Return normalized prior messages to use as chat history.

    Behaviour:
    - Considers ``run_agent_input.messages`` in order.
    - If the *final* message is a ``"user"`` message, treats everything *before*
      it as history (so the latest user turn can be handled separately).
    - Otherwise treats all provided messages as history.
    - Converts messages into ``{role, content}`` dicts with string content.
    - Truncates to the most recent ``max_history`` entries.

    Special cases:
    - When there are no messages, returns an empty list.
    - When ``max_history <= 0``, history is disabled and an empty list is returned.
      This matches the documented semantics where 0 means "no history".
    """
    raw_messages = list(run_agent_input.messages)
    if not raw_messages or max_history <= 0:
        return []

    # Only exclude the final user message when it is actually the last message
    # in the provided list. This avoids dropping trailing assistant/tool messages
    # that some runtimes include after the last user message.
    last_role = _get_message_field(raw_messages[-1], "role")
    history_slice = raw_messages[:-1] if last_role == "user" else raw_messages

    # Keep only the most recent N messages in history to avoid unbounded growth
    # when callers provide long transcripts.
    if len(history_slice) > max_history:
        history_slice = history_slice[-max_history:]

    history: list[NormalizedHistoryMessage] = []
    for message in history_slice:
        role = _get_message_field(message, "role")
        content = _get_message_field(message, "content")
        tool_call_id = _get_message_field(message, "tool_call_id")
        name = _get_message_field(message, "name")
        tool_calls = _get_message_field(message, "tool_calls")

        text = str(content) if content is not None else ""
        if not text and tool_calls is not None:
            # Preserve assistant tool-call messages even when content is empty/None.
            text = _summarize_tool_calls(tool_calls)
        if not text and str(role or "") == "tool":
            # Tool outputs should generally have content, but if they don't,
            # keep a minimal placeholder so downstream adapters don't silently
            # drop tool steps.
            label = str(name) if name is not None else ""
            if not label and tool_call_id is not None:
                label = str(tool_call_id)
            text = f"[tool] {label}".strip() if label else "[tool]"
        if not text:
            continue

        entry: NormalizedHistoryMessage = {
            "role": str(role or "user"),
            "content": text,
        }

        # Preserve optional tool metadata when present so downstream agents can
        # reconstruct richer tool histories if desired.
        if tool_call_id is not None:
            entry["tool_call_id"] = str(tool_call_id)
        if name is not None:
            entry["name"] = str(name)
        if tool_calls is not None:
            entry["tool_calls"] = tool_calls

        history.append(entry)

    return history


def build_history_summary_from_messages(
    run_agent_input: RunAgentInput,
    max_history: int,
) -> str:
    r"""Build a plain-text summary of prior turns for prompts.

    This is a convenience helper around ``extract_history_messages`` that:
    - Uses the same history selection semantics as ``extract_history_messages``
    - Truncates to the most recent ``max_history`` entries
    - Normalizes them into ``{role, content}`` dicts
    and then renders a newline-separated transcript of the form
    ``role: content``.

    Returns an empty string when there is no history to include. Callers that
    embed the result in a prompt section (e.g. "Prior conversation:\\n{summary}")
    should check for an empty return value and omit the section entirely to
    avoid dangling headers.
    """
    history = extract_history_messages(run_agent_input, max_history)
    if not history:
        return ""

    lines = [f"{msg['role']}: {msg['content']}" for msg in history]
    return "\n".join(lines)
