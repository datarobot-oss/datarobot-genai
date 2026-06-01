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

"""Streaming counterpart to NAT's ``auto_memory_agent``.

NAT's upstream ``auto_memory_agent`` calls ``inner_agent_fn.ainvoke(...)`` in
its ``inner_agent_node``, which collapses the inner stream into a single
string.  For AG-UI consumers that means only one
``CustomEvent("DEFAULT_NAT_RESPONSE")`` ever reaches the frontend; the token
deltas and tool-call deltas the inner agent's ``stream_fn`` would have
emitted are lost.

``streaming_memory_agent`` performs the same mem0 capture/retrieve operations
(save user message → retrieve and inject as system context → save AI
response) but ``astream``s the inner agent and yields its native
``DRAgentEventResponse`` events straight through, so token deltas and
tool-call deltas surface as proper AG-UI ``TextMessage*`` / ``ToolCall*``
events.

Registered with ``register_per_user_function`` so the wrapper itself builds
lazily inside a ``PerUserWorkflowBuilder``. Without that, the wrapper's
``builder.get_function(inner_agent_name)`` call would run at workflow-build
time and miss per-user inner agents (which are built lazily on first user
invocation). Per-user inner agents end up in the per-user builder's cache
before the workflow is built, and shared inner agents still resolve via
fall-through to the shared builder — both cases work transparently.

The configuration surface (``memory_name``, ``inner_agent_name``,
``save_user_messages_to_memory``, ``retrieve_memory_for_every_response``,
``save_ai_messages_to_memory``, ``search_params``, ``add_params``) is inherited
from upstream ``AutoMemoryAgentConfig`` so the two wrappers can be swapped
without reauthoring ``workflow.yaml``.
"""

import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Annotated
from typing import Any

from ag_ui.core import RunAgentInput
from ag_ui.core import SystemMessage as AgUiSystemMessage
from ag_ui.core import UserMessage as AgUiUserMessage
from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.builder.function_info import FunctionInfo
from nat.builder.function_info import Streaming
from nat.cli.register_workflow import register_per_user_function
from nat.memory.models import MemoryItem
from nat.plugins.langchain.agent.auto_memory_wrapper.register import AutoMemoryAgentConfig

from datarobot_genai.dragent.frontends.converters import aggregate_dragent_event_responses
from datarobot_genai.dragent.frontends.converters import convert_dragent_event_response_to_str
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse

logger = logging.getLogger(__name__)


class StreamingMemoryAgentConfig(  # type: ignore[call-arg, misc]
    AutoMemoryAgentConfig,
    name="streaming_memory_agent",
):
    """Streaming, per-user variant of ``auto_memory_agent``.

    Inherits ``memory_name``, ``inner_agent_name``,
    ``save_user_messages_to_memory``, ``retrieve_memory_for_every_response``,
    ``save_ai_messages_to_memory``, ``search_params``, and ``add_params``
    from :class:`AutoMemoryAgentConfig`.  Only the ``_type`` discriminator
    differs, so a workflow can switch between the two by renaming ``_type``.
    """


def _user_id_from_context() -> str:
    """Resolve user_id with the same priority chain as auto_memory_agent."""
    context = Context.get()
    user_manager = getattr(context, "user_manager", None)
    if user_manager is not None and hasattr(user_manager, "get_id"):
        try:
            uid = user_manager.get_id()
            if uid:
                return uid
        except Exception as exc:  # noqa: BLE001
            logger.debug("user_manager.get_id() failed: %s", exc)
    metadata = getattr(context, "metadata", None)
    headers = getattr(metadata, "headers", None) if metadata else None
    if headers:
        uid = headers.get("x-user-id")
        if uid:
            return uid
    return "default_user"


def _last_user_text(messages: list[Any]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AgUiUserMessage) and msg.content:
            content = msg.content
            if isinstance(content, str):
                return content
            return "".join(getattr(part, "text", "") for part in content if part is not None)
    return ""


def _with_memory_context(messages: list[Any], memory_text: str) -> list[Any]:
    """Return a new list with a system message inserted before the last user message."""
    payload = f"Relevant context from memory:\n{memory_text}"
    sys_msg = AgUiSystemMessage(id=str(uuid.uuid4()), content=payload)
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if isinstance(out[i], AgUiUserMessage) and out[i].content:
            out.insert(i, sys_msg)
            return out
    out.insert(0, sys_msg)
    return out


@register_per_user_function(  # type: ignore[untyped-decorator]
    config_type=StreamingMemoryAgentConfig,
    input_type=RunAgentInput,
    streaming_output_type=DRAgentEventResponse,
)
async def streaming_memory_agent(
    config: StreamingMemoryAgentConfig, builder: Builder
) -> AsyncGenerator[Any, None]:
    """Build the streaming memory agent workflow."""
    memory_editor = await builder.get_memory_client(config.memory_name)
    inner_agent_fn = await builder.get_function(config.inner_agent_name)

    async def _stream_fn(
        input_message: RunAgentInput,
    ) -> Annotated[
        AsyncGenerator[DRAgentEventResponse, None],
        Streaming(convert=aggregate_dragent_event_responses),
    ]:
        user_text = _last_user_text(input_message.messages)
        user_id = _user_id_from_context()

        # 1. Capture the user's latest message.
        if config.save_user_messages_to_memory and user_text:
            try:
                await memory_editor.add_items(
                    [
                        MemoryItem(
                            conversation=[{"role": "user", "content": user_text}],
                            user_id=user_id,
                        )
                    ],
                    **config.add_params,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("memory.add_items(user) failed: %s", exc)

        # 2. Retrieve relevant memory and inject it as a system message.
        messages = list(input_message.messages)
        if config.retrieve_memory_for_every_response and user_text:
            try:
                memory_items = await memory_editor.search(
                    query=user_text,
                    user_id=user_id,
                    **config.search_params,
                )
                texts = [item.memory for item in memory_items if item.memory]
                if texts:
                    messages = _with_memory_context(messages, "\n".join(texts))
            except Exception as exc:  # noqa: BLE001
                logger.warning("memory.search failed: %s", exc)

        inner_input = input_message.model_copy(update={"messages": messages})

        # 3. astream the inner agent. Inner per-user agents (or shared inner
        #    agents) yield DRAgentEventResponse natively; we pass them straight
        #    through to the caller while collecting them for the post-stream
        #    assistant memory save.
        collected: list[DRAgentEventResponse] = []
        try:
            async for event_response in inner_agent_fn.astream(inner_input):
                collected.append(event_response)
                yield event_response
        finally:
            # 4. Persist the assistant response. Run regardless of partial
            #    errors so partial output still lands in memory.
            if config.save_ai_messages_to_memory and collected:
                aggregated = aggregate_dragent_event_responses(collected)
                ai_text = convert_dragent_event_response_to_str(aggregated)
                if ai_text:
                    try:
                        await memory_editor.add_items(
                            [
                                MemoryItem(
                                    conversation=[{"role": "assistant", "content": ai_text}],
                                    user_id=user_id,
                                )
                            ],
                            **config.add_params,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("memory.add_items(assistant) failed: %s", exc)

    async def _stream_to_str(
        responses: AsyncGenerator[DRAgentEventResponse],
    ) -> str:
        aggregated = aggregate_dragent_event_responses([r async for r in responses])
        return convert_dragent_event_response_to_str(aggregated)

    yield FunctionInfo.create(
        stream_fn=_stream_fn,
        stream_to_single_fn=_stream_to_str,
        description=config.description,
    )
