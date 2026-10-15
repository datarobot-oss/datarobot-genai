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

"""Per-user variant of the NAT tool_calling_agent workflow.

The built-in ``tool_calling_agent`` is registered as a *shared* workflow, which
means NAT's dependency validator forbids it from referencing per-user function
groups such as ``a2a_client``.  This module registers an identical workflow under
the name ``per_user_tool_calling_agent`` using ``register_per_user_function`` so
that per-user function groups can be used while still benefiting from OpenAI-style
structured tool calling (``bind_tools``).

NAT 1.6 added a ``stream_fn`` that yields ``ChatResponseChunk`` (and a ``single_fn`` that
returns ``str``).  This wrapper leaves that native output untouched; the
``datarobot_dragent_normalization`` middleware converts it into ``DRAgentEventResponse`` with
valid AG-UI event sequences.  Declare that middleware on this function (in the ``workflow`` block
when this agent is the workflow, or on the inner function when it is a memory wrapper's
``inner_agent_name``).
"""

import uuid
from collections.abc import AsyncGenerator
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.pregel._messages import StreamMessagesHandler
from langgraph.pregel._messages import _state_values
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_per_user_function
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import ChatResponseChunk
from nat.plugins.langchain.agent.tool_calling_agent.register import ToolCallAgentWorkflowConfig
from nat.plugins.langchain.agent.tool_calling_agent.register import tool_calling_agent_workflow

# Workaround: prior assistant messages from chat history were leaking back into
# the response stream as a single trailing mega-chunk after the new response.
#
# How that happens:
#   1. NAT's `Message` data model has no `id` field, so when `_stream_fn`
#      converts the request into BaseMessages via
#      `trim_messages([m.model_dump() for m in chat.messages], ...)`, every
#      BaseMessage handed to the graph has `id=None`.
#   2. Langgraph's `StreamMessagesHandler` uses a `seen` set, keyed by
#      `message.id`, to avoid emitting the same message twice. Its
#      `on_chain_start` only records ids when `id is not None`, so id-less
#      inputs are never added to `seen`.
#   3. When `agent_node` returns, `on_chain_end` walks the new state and emits
#      every BaseMessage not in `seen`. Prior history items (id=None) match
#      that condition and get streamed to the client as if the model had just
#      produced them, even though they were part of the conversation history.
#
# Fix: assign a uuid to every id-less input message before the original
# `on_chain_start` runs, mirroring what `StreamMessagesHandler._emit` already
# does for output messages. This makes the seen set track them correctly, so
# `on_chain_end` no longer treats them as new output.
#
_original_on_chain_start = StreamMessagesHandler.on_chain_start


def _patched_on_chain_start(
    self: StreamMessagesHandler,
    serialized: dict[str, Any],
    inputs: dict[str, Any],
    **kwargs: Any,
) -> Any:
    for value in _state_values(inputs):
        if isinstance(value, BaseMessage) and value.id is None:
            value.id = str(uuid.uuid4())
        elif isinstance(value, Sequence) and not isinstance(value, str):
            for item in value:
                if isinstance(item, BaseMessage) and item.id is None:
                    item.id = str(uuid.uuid4())
    return _original_on_chain_start(self, serialized, inputs, **kwargs)


StreamMessagesHandler.on_chain_start = _patched_on_chain_start  # type: ignore[method-assign]


class PerUserToolCallAgentWorkflowConfig(
    ToolCallAgentWorkflowConfig,
    name="per_user_tool_calling_agent",  # type: ignore[call-arg]
):
    """Per-user version of tool_calling_agent."""

    pass


async def _per_user_tool_calling_agent(
    config: PerUserToolCallAgentWorkflowConfig, builder: Any
) -> AsyncGenerator[Any, None]:
    """Re-register tool_calling_agent as a per-user function, unwrapped.

    The native ``FunctionInfo`` (``single_fn -> str``, ``stream_fn -> ChatResponseChunk``) is
    yielded as-is; the ``datarobot_dragent_normalization`` middleware converts that output into
    ``DRAgentEventResponse``.
    """
    original_gen = tool_calling_agent_workflow.__wrapped__(config, builder)
    try:
        fn_info = await original_gen.__anext__()
        yield fn_info
    finally:
        await original_gen.aclose()


register_per_user_function(
    config_type=PerUserToolCallAgentWorkflowConfig,
    input_type=ChatRequest,
    single_output_type=ChatResponse,
    streaming_output_type=ChatResponseChunk,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)(_per_user_tool_calling_agent)
