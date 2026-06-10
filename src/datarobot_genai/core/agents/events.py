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
"""Fold a sequence of AG-UI events back into messages.

Reducing the streamed events of a run into the coarse-grained ``Message`` objects
that ``RunAgentInput.messages`` carries is a basic AG-UI client operation -- any
client that supports multi-turn has to do it to replay a finished turn as history.
The AG-UI TypeScript client implements it as ``defaultApplyEvents``; the Python
SDK ships only the type defs and the SSE encoder, so this is the genai port of the
*messages* slice of that operation (state / state-deltas / snapshots are out of
scope here).

(ag-ui-protocol/ag-ui):TS ``defaultApplyEvents``:
  https://github.com/ag-ui-protocol/ag-ui/blob/main/sdks/typescript/packages/client/src/apply/default.ts

The fold preserves the structure flattening used to drop -- in particular it keeps
an assistant step's text and the tool calls it made on the *same*
``AssistantMessage``, followed by the paired ``ToolMessage`` results, so the
reconstructed transcript is valid to send back to any chat-completions provider.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import Any

from ag_ui.core import AssistantMessage
from ag_ui.core import EventType
from ag_ui.core import FunctionCall
from ag_ui.core import Message
from ag_ui.core import ReasoningMessage
from ag_ui.core import ToolCall
from ag_ui.core import ToolMessage

_TEXT_DELTA_TYPES = frozenset({EventType.TEXT_MESSAGE_CONTENT, EventType.TEXT_MESSAGE_CHUNK})
_REASONING_DELTA_TYPES = frozenset(
    {
        EventType.REASONING_MESSAGE_CONTENT,
        EventType.REASONING_MESSAGE_CHUNK,
        EventType.THINKING_TEXT_MESSAGE_CONTENT,
    }
)
_TOOL_START_TYPES = frozenset({EventType.TOOL_CALL_START, EventType.TOOL_CALL_CHUNK})


def events_to_messages(events: Iterable[Any]) -> list[Message]:
    """Fold a sequence of AG-UI events into ordered ``Message`` objects.

    Parameters
    ----------
    events:
        AG-UI events (pydantic event objects) in emission order -- e.g. the
        flattened ``.events`` of a streamed run.

    Returns
    -------
    list[Message]
        ``AssistantMessage`` / ``ToolMessage`` / ``ReasoningMessage`` in emission
        order. An assistant step carries its text and tool calls on one
        ``AssistantMessage``, immediately followed by the ``ToolMessage`` result(s)
        for those calls; reasoning becomes a ``ReasoningMessage`` ahead of the step
        it preceded. Empty text/reasoning turns are dropped.
    """
    messages: list[Message] = []
    assistants: dict[str, AssistantMessage] = {}  # assistant message by its id
    reasonings: dict[str, ReasoningMessage] = {}  # reasoning message by its id
    tool_calls: dict[str, ToolCall] = {}  # ToolCall by tool_call_id (for arg accumulation)
    # The assistant step currently being assembled: text accumulates here, and a
    # tool call with no resolvable parent_message_id attaches here. Cleared by a
    # tool result, so the next text/tool opens a fresh step.
    current: AssistantMessage | None = None

    def _text_assistant(message_id: str | None) -> AssistantMessage:
        nonlocal current
        key = message_id or "\x00text"
        assistant = assistants.get(key)
        if assistant is None:
            assistant = AssistantMessage(id=message_id or uuid.uuid4().hex, content="")
            assistants[key] = assistant
            messages.append(assistant)
        current = assistant
        return assistant

    def _tool_assistant(parent_message_id: str | None, tool_call_id: str) -> AssistantMessage:
        # AG-UI resolution: prefer an existing assistant message named by
        # parent_message_id; else the open step; else create one keyed by the
        # parent (or the tool call) id.
        nonlocal current
        if parent_message_id and parent_message_id in assistants:
            target = assistants[parent_message_id]
        elif current is not None:
            target = current
        else:
            target = AssistantMessage(id=parent_message_id or tool_call_id, content=None)
            assistants[target.id] = target
            messages.append(target)
        if target.tool_calls is None:
            target.tool_calls = []
        current = target
        return target

    def _reasoning(message_id: str | None, delta: str) -> None:
        key = message_id or "\x00reasoning"
        reasoning = reasonings.get(key)
        if reasoning is None:
            reasoning = ReasoningMessage(id=message_id or uuid.uuid4().hex, content="")
            reasonings[key] = reasoning
            messages.append(reasoning)
        if delta:
            reasoning.content += delta

    for event in events:
        event_type = getattr(event, "type", None)

        if event_type == EventType.TEXT_MESSAGE_START:
            _text_assistant(getattr(event, "message_id", None))
        elif event_type in _TEXT_DELTA_TYPES:
            delta = getattr(event, "delta", None)
            role = getattr(event, "role", None)
            if delta and role in (None, "assistant"):
                assistant = _text_assistant(getattr(event, "message_id", None))
                assistant.content = (assistant.content or "") + delta
        elif event_type in _TOOL_START_TYPES:
            tool_call_id = getattr(event, "tool_call_id", None)
            if tool_call_id:
                target = _tool_assistant(getattr(event, "parent_message_id", None), tool_call_id)
                call = tool_calls.get(tool_call_id)
                name = getattr(event, "tool_call_name", None)
                if call is None:
                    call = ToolCall(
                        id=tool_call_id, function=FunctionCall(name=name or "", arguments="")
                    )
                    target.tool_calls.append(call)  # type: ignore[union-attr]
                    tool_calls[tool_call_id] = call
                elif name:
                    call.function.name = name
                # ToolCallChunk may also carry an argument delta.
                delta = getattr(event, "delta", None)
                if delta:
                    call.function.arguments += delta
        elif event_type == EventType.TOOL_CALL_ARGS:
            tool_call_id = getattr(event, "tool_call_id", None)
            delta = getattr(event, "delta", None)
            if tool_call_id and delta:
                call = tool_calls.get(tool_call_id)
                if call is None:
                    target = _tool_assistant(None, tool_call_id)
                    call = ToolCall(id=tool_call_id, function=FunctionCall(name="", arguments=""))
                    target.tool_calls.append(call)  # type: ignore[union-attr]
                    tool_calls[tool_call_id] = call
                call.function.arguments += delta
        elif event_type == EventType.TOOL_CALL_RESULT:
            tool_call_id = getattr(event, "tool_call_id", None)
            if tool_call_id:
                tool_message = ToolMessage(
                    id=getattr(event, "message_id", None) or uuid.uuid4().hex,
                    tool_call_id=tool_call_id,
                    content=getattr(event, "content", None) or "",
                )
                # Insert the result immediately after the assistant message that
                # issued the call (skipping past any results already recorded for it
                # so parallel results keep order), mirroring AG-UI's reducer. This
                # keeps the transcript valid even if a follow-up answer streamed
                # before the result arrived; fall back to append if no owner found.
                owner_index = next(
                    (
                        i
                        for i, message in enumerate(messages)
                        if isinstance(message, AssistantMessage)
                        and message.tool_calls
                        and any(call.id == tool_call_id for call in message.tool_calls)
                    ),
                    -1,
                )
                if owner_index == -1:
                    messages.append(tool_message)
                else:
                    insert_at = owner_index + 1
                    while insert_at < len(messages) and messages[insert_at].role == "tool":
                        insert_at += 1
                    messages.insert(insert_at, tool_message)
                current = None  # the tool-calling step is done; next text opens a new one
        elif event_type == EventType.REASONING_MESSAGE_START:
            _reasoning(getattr(event, "message_id", None), "")
        elif event_type in _REASONING_DELTA_TYPES:
            delta = getattr(event, "delta", None)
            if delta:
                _reasoning(getattr(event, "message_id", None), delta)
        # All other events (TEXT/TOOL/REASONING *_END, RUN_*, STEP_*, STATE_*, ...)
        # carry no message content to fold.

    def _keep(message: Message) -> bool:
        if isinstance(message, AssistantMessage):
            return bool(message.content) or bool(message.tool_calls)
        if isinstance(message, ReasoningMessage):
            return bool(message.content)
        return True

    return [message for message in messages if _keep(message)]
