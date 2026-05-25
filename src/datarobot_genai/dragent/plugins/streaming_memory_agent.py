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
response) but ``astream``s the inner agent and pipes its
``ChatResponseChunk`` stream through
:func:`convert_chunks_to_agui_events`, so token deltas and tool-call deltas
surface as proper AG-UI ``TextMessage*`` / ``ToolCall*`` events.

The configuration surface (``memory_name``, ``inner_agent_name``,
``save_user_messages_to_memory``, ``retrieve_memory_for_every_response``,
``save_ai_messages_to_memory``, ``search_params``, ``add_params``) is inherited
from upstream ``AutoMemoryAgentConfig`` so the two wrappers can be swapped
without reauthoring ``workflow.yaml``.
"""

import logging
from collections.abc import AsyncGenerator
from typing import Annotated
from typing import Any

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.builder.function_info import FunctionInfo
from nat.builder.function_info import Streaming
from nat.cli.register_workflow import register_function
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import Message
from nat.data_models.api_server import UserMessageContentRoleType
from nat.memory.models import MemoryItem
from nat.plugins.langchain.agent.auto_memory_wrapper.register import AutoMemoryAgentConfig

from datarobot_genai.dragent.frontends.converters import aggregate_dragent_event_responses
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse
from datarobot_genai.dragent.frontends.stream_converter import convert_chunks_to_agui_events

logger = logging.getLogger(__name__)


class StreamingMemoryAgentConfig(  # type: ignore[call-arg, misc]
    AutoMemoryAgentConfig,
    name="streaming_memory_agent",
):
    """Streaming variant of ``auto_memory_agent``.

    Inherits ``memory_name``, ``inner_agent_name``,
    ``save_user_messages_to_memory``, ``retrieve_memory_for_every_response``,
    ``save_ai_messages_to_memory``, ``search_params``, and ``add_params``
    from :class:`AutoMemoryAgentConfig`.  No additional fields — only the
    ``_type`` discriminator differs, so a workflow can switch between the two
    by renaming ``_type``.
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


def _last_user_text(messages: list[Message]) -> str:
    for msg in reversed(messages):
        if msg.role == UserMessageContentRoleType.USER and msg.content:
            return str(msg.content)
    return ""


def _with_memory_context(messages: list[Message], memory_text: str) -> list[Message]:
    """Return a new list with a system message inserted before the last user message."""
    payload = f"Relevant context from memory:\n{memory_text}"
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].role == UserMessageContentRoleType.USER and out[i].content:
            out.insert(i, Message(role=UserMessageContentRoleType.SYSTEM, content=payload))
            return out
    out.insert(0, Message(role=UserMessageContentRoleType.SYSTEM, content=payload))
    return out


@register_function(config_type=StreamingMemoryAgentConfig)  # type: ignore[untyped-decorator]
async def streaming_memory_agent(
    config: StreamingMemoryAgentConfig, builder: Builder
) -> AsyncGenerator[Any, None]:
    """Build the streaming memory agent workflow."""
    memory_editor = await builder.get_memory_client(config.memory_name)
    inner_agent_fn = await builder.get_function(config.inner_agent_name)

    async def _stream_fn(
        chat_request: ChatRequest,
    ) -> Annotated[
        AsyncGenerator[DRAgentEventResponse, None],
        Streaming(convert=aggregate_dragent_event_responses),
    ]:
        user_text = _last_user_text(chat_request.messages)
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
        messages = list(chat_request.messages)
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

        inner_request = chat_request.model_copy(update={"messages": messages, "stream": True})

        # 3. astream the inner agent.  Wrap the chunk iterator so we can both
        #    accumulate the response text for the post-stream memory save and
        #    hand the chunks to the shared AG-UI converter, which already
        #    handles parent_message_id, tool-call-id registry handoff to the
        #    step adaptor, and client-disconnect / error semantics.
        ai_buffer: list[str] = []

        async def _collecting_chunks() -> AsyncGenerator[ChatResponseChunk, None]:
            async for chunk in inner_agent_fn.astream(inner_request, to_type=ChatResponseChunk):
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    ai_buffer.append(chunk.choices[0].delta.content)
                yield chunk

        async for event_response in convert_chunks_to_agui_events(_collecting_chunks()):
            yield event_response

        # 4. Persist the assistant response.  Run regardless of partial errors
        #    so partial output still lands in memory; convert_chunks_to_agui_events
        #    has already surfaced any upstream error as a RunErrorEvent.
        ai_text = "".join(ai_buffer)
        if config.save_ai_messages_to_memory and ai_text:
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

    yield FunctionInfo.create(
        stream_fn=_stream_fn,
        description=config.description,
    )
