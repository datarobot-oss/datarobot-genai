# Copyright 2025 DataRobot, Inc. and its affiliates.
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

from __future__ import annotations

import logging
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from uuid import uuid4

from ag_ui.core import AssistantMessage
from ag_ui.core import Message
from ag_ui.core import RunAgentInput
from ag_ui.core import RunErrorEvent
from ag_ui.core import RunFinishedEvent
from ag_ui.core import SystemMessage
from ag_ui.core import TextMessageChunkEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageStartEvent
from ag_ui.core import Tool
from ag_ui.core import ToolMessage
from ag_ui.core import UserMessage
from ag_ui.core.types import FunctionCall
from ag_ui.core.types import ToolCall
from openai.types.chat import CompletionCreateParams

from datarobot_genai.core.agents import InvokeReturn
from datarobot_genai.core.agents import default_usage_metrics
from datarobot_genai.core.agents.base import BaseAgent
from datarobot_genai.core.agents.base import UsageMetrics

if TYPE_CHECKING:
    from datarobot_genai.core.pipeline_interactions import MultiTurnSample

logger = logging.getLogger(__name__)


def _optional_str(mapping: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        val = mapping.get(key)
        if val is not None and val != "":
            return str(val)
    return None


def _resolve_thread_and_run_ids(chat_completion_params: Mapping[str, Any]) -> tuple[str, str]:
    """Prefer AG-UI thread/run ids when present so LangGraph can resume across requests.

    Callers may pass ``thread_id`` / ``run_id`` at the top level or inside OpenAI client
    ``extra_body`` (both shapes appear depending on gateway and SDK behavior).
    """
    thread_id = _optional_str(chat_completion_params, "thread_id", "threadId")
    run_id = _optional_str(chat_completion_params, "run_id", "runId")
    extra = chat_completion_params.get("extra_body")
    if isinstance(extra, Mapping):
        if thread_id is None:
            thread_id = _optional_str(extra, "thread_id", "threadId")
        if run_id is None:
            run_id = _optional_str(extra, "run_id", "runId")
    return (
        thread_id if thread_id is not None else str(uuid4()),
        run_id if run_id is not None else str(uuid4()),
    )


def _merge_mcp_tools_with_agent_tools(mcp_tools: Any, agent: BaseAgent) -> list[Any]:
    """Return MCP tools followed by any tools already set on the agent (e.g. workflow tools)."""
    return [*list(mcp_tools), *list(agent.tools)]


def is_streaming(completion_create_params: CompletionCreateParams | Mapping[str, Any]) -> bool:
    """Return True when the request asks for streaming, False otherwise.

    Accepts both pydantic types and plain dictionaries.
    """
    params = cast(Mapping[str, Any], completion_create_params)
    value = params.get("stream", False)
    # Handle non-bool truthy values defensively (e.g., "true")
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def backfill_model(current: str | None, fallback: str | None) -> str | None:
    """Replace NAT's ``"unknown-model"`` placeholder (or ``None``) with ``fallback``.

    NAT defaults ``ChatResponse.model`` / ``ChatResponseChunk.model`` to the literal
    ``"unknown-model"`` whenever the workflow output didn't carry one. Callers pass the
    agent's configured model (:func:`default_response_model`) as ``fallback`` so the
    response reports the model the agent actually ran. A real model the workflow
    produced — or a deliberately-set one such as moderation's ``MODERATION_MODEL_NAME``
    — is preserved.
    """
    if fallback and current in (None, "unknown-model"):
        return fallback
    return current


class FinalAssistantTextAccumulator:
    """Reassemble assistant text from AG-UI events, keeping only the *final* message.

    A chat completion carries a single ``content`` string, so a run that produced more
    than one assistant message has to collapse to one of them. The one the caller asked
    for is the last: in a multi-node LangGraph graph (the recipe template's
    ``researcher_node -> responder_node`` shape) every node emits its own assistant
    message, but only the responder's is the answer. Concatenating them yields the
    reported ``"ParisParis"`` corruption. The same rule drops a tool-calling agent's
    pre-tool preamble in favour of its final answer.

    A message boundary is either an explicit ``TextMessageStartEvent`` or a change of
    ``message_id`` on a delta (chunk-only streams never emit a START). Streams with a
    single message are unaffected: every delta lands in the same message.
    """

    # Distinct from ``None``, which is a legitimate ``TextMessageChunkEvent.message_id``.
    _UNSET = object()

    def __init__(self) -> None:
        self._text = ""
        self._message_id: Any = self._UNSET

    def add(self, event: Any) -> None:
        """Feed one AG-UI event; non-text events are ignored."""
        if isinstance(event, TextMessageStartEvent):
            self._begin(event.message_id)
        elif isinstance(event, (TextMessageContentEvent, TextMessageChunkEvent)):
            message_id = getattr(event, "message_id", None)
            if self._message_id is self._UNSET:
                self._message_id = message_id
            elif message_id is not None and message_id != self._message_id:
                self._begin(message_id)
            # ``TextMessageChunkEvent.delta`` is optional.
            self._text += event.delta or ""

    def _begin(self, message_id: str | None) -> None:
        self._text = ""
        self._message_id = message_id

    @property
    def text(self) -> str:
        """The text of the most recently opened assistant message."""
        return self._text


def final_assistant_text(events: Iterable[Any]) -> str:
    """Return the final assistant message's text from a complete AG-UI event sequence.

    See :class:`FinalAssistantTextAccumulator` for why earlier messages are discarded.
    """
    accumulator = FinalAssistantTextAccumulator()
    for event in events:
        accumulator.add(event)
    return accumulator.text


def convert_chat_completion_params_to_run_agent_input(
    chat_completion_params: CompletionCreateParams | Mapping[str, Any],
) -> RunAgentInput:
    """Convert a chat completion parameters to a run agent input."""
    tools = [
        Tool(
            name=tool.get("function").get("name"),
            description=tool.get("function").get("description"),
            parameters=tool.get("function").get("parameters"),
        )
        for tool in chat_completion_params.get("tools", []) or []
        if tool.get("type") == "function"  # type: ignore[union-attr]
    ]
    messages: list[Message] = []
    for i, message in enumerate(chat_completion_params.get("messages", [])):  # type: ignore[arg-type]
        id = f"message_{i}"
        if message.get("role") == "user":
            messages.append(UserMessage(id=id, content=message.get("content")))
        elif message.get("role") == "assistant":
            tool_calls = []
            for tool_call in message.get("tool_calls", []) or []:
                function = tool_call.get("function") or {}
                tool_calls.append(
                    ToolCall(
                        id=tool_call.get("id"),
                        type=tool_call.get("type", "function"),
                        function=FunctionCall(
                            name=function.get("name"),
                            arguments=function.get("arguments", "{}"),
                        ),
                    )
                )
            messages.append(
                AssistantMessage(
                    id=id,
                    content=message.get("content"),
                    tool_calls=tool_calls or None,
                )
            )
        elif message.get("role") == "tool":
            messages.append(
                ToolMessage(
                    id=id,
                    content=message.get("content"),
                    tool_call_id=message.get("tool_call_id"),
                    error=message.get("error"),
                )
            )
        elif message.get("role") == "system":
            messages.append(SystemMessage(id=id, content=message.get("content")))

    forwarded_props: dict[str, Any] = {
        "model": chat_completion_params.get("model"),
        "authorization_context": chat_completion_params.get("authorization_context"),
        "forwarded_headers": chat_completion_params.get("forwarded_headers"),
    }

    thread_id, run_id = _resolve_thread_and_run_ids(chat_completion_params)

    return RunAgentInput(
        messages=messages,
        tools=tools,
        forwarded_props=forwarded_props,
        thread_id=thread_id,
        run_id=run_id,
        state={},
        context=[],
    )


async def agent_chat_completion_wrapper(
    agent: BaseAgent,
    chat_completion_params: CompletionCreateParams | Mapping[str, Any],
    mcp_tools_factory: Callable[[], Any],
) -> InvokeReturn | tuple[str, MultiTurnSample | None, UsageMetrics]:
    """Wrap the agent's invoke method in a chat completion wrapper.

    MCP tools from ``mcp_tools_factory`` are combined with any tools already on
    ``agent`` (MCP first, then existing ``agent.tools``).

    Returns
    -------
    InvokeReturn
        When streaming is requested - the raw async event generator
    tuple[str, MultiTurnSample | None, UsageMetrics]
        When non-streaming - the reassembled final text, pipeline
        interactions, and accumulated usage metrics.
    """
    run_agent_input = convert_chat_completion_params_to_run_agent_input(chat_completion_params)

    if is_streaming(chat_completion_params):

        async def _stream_with_mcp() -> InvokeReturn:
            async with mcp_tools_factory() as mcp_tools:
                agent.set_tools(_merge_mcp_tools_with_agent_tools(mcp_tools, agent))
                async for item in agent.invoke(run_agent_input):
                    yield item

        return _stream_with_mcp()
    else:
        async with mcp_tools_factory() as mcp_tools:
            agent.set_tools(_merge_mcp_tools_with_agent_tools(mcp_tools, agent))
            # When we work in non-streaming mode, we only send back the final message.
            # It is because of limitation of completions interface we can not send back the
            # intermediate messages.
            response_text = FinalAssistantTextAccumulator()
            pipeline_interactions = None
            usage_metrics = default_usage_metrics()
            received_run_finished = False
            async for event, iter_interactions, iter_metrics in agent.invoke(run_agent_input):
                response_text.add(event)
                if isinstance(event, RunFinishedEvent):
                    received_run_finished = True
                    pipeline_interactions = iter_interactions
                    usage_metrics = iter_metrics
                elif isinstance(event, RunErrorEvent):
                    raise RuntimeError(event.message)

            if not received_run_finished:
                logger.warning("Agent stream ended without RunFinishedEvent")

            return response_text.text, pipeline_interactions, usage_metrics
