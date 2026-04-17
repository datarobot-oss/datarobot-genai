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

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from collections.abc import Iterator
from typing import TYPE_CHECKING
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import AIMessageChunk
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.outputs import ChatResult
from pydantic import ConfigDict

if TYPE_CHECKING:
    from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
    from langchain_core.callbacks import CallbackManagerForLLMRun


def _lc_messages_to_litellm(messages: list[BaseMessage]) -> list[dict]:
    """Convert LangChain messages to litellm/OpenAI message dicts."""
    from langchain_core.messages import FunctionMessage  # noqa: PLC0415
    from langchain_core.messages import HumanMessage  # noqa: PLC0415
    from langchain_core.messages import SystemMessage  # noqa: PLC0415
    from langchain_core.messages import ToolMessage  # noqa: PLC0415

    result: list[dict] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            d: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"]),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            result.append(d)
        elif isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            result.append(
                {
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                }
            )
        elif isinstance(msg, FunctionMessage):
            result.append({"role": "function", "content": msg.content, "name": msg.name})
        else:
            result.append({"role": "user", "content": str(msg.content)})
    return result


def _litellm_response_to_chat_result(response: Any) -> ChatResult:
    """Convert a litellm ModelResponse to a LangChain ChatResult."""
    choice = response.choices[0]
    message = choice.message
    content = message.content or ""

    tool_calls: list[dict] = []
    if getattr(message, "tool_calls", None):
        for tc in message.tool_calls:
            tool_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments),
                }
            )

    ai_message = (
        AIMessage(content=content, tool_calls=tool_calls)
        if tool_calls
        else AIMessage(content=content)
    )
    return ChatResult(generations=[ChatGeneration(message=ai_message)])


class RouterChatModel(BaseChatModel):
    """LangChain ``BaseChatModel`` that delegates to a ``litellm.Router``.

    Automatic failover is handled by the router: when the primary model
    raises an error the router cascades to each configured fallback in order.
    Supports both streaming and non-streaming generation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    router: Any  # litellm.Router

    @property
    def _llm_type(self) -> str:
        return "datarobot-router"

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: PLC0415

        formatted = [convert_to_openai_tool(t) for t in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self.bind(tools=formatted, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        litellm_messages = _lc_messages_to_litellm(messages)
        call_kwargs: dict[str, Any] = {}
        if stop:
            call_kwargs["stop"] = stop
        call_kwargs.update(kwargs)
        response = self.router.completion("primary", messages=litellm_messages, **call_kwargs)
        return _litellm_response_to_chat_result(response)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        litellm_messages = _lc_messages_to_litellm(messages)
        call_kwargs: dict[str, Any] = {}
        if stop:
            call_kwargs["stop"] = stop
        call_kwargs.update(kwargs)
        response = await self.router.acompletion(
            "primary", messages=litellm_messages, **call_kwargs
        )
        return _litellm_response_to_chat_result(response)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        litellm_messages = _lc_messages_to_litellm(messages)
        call_kwargs: dict[str, Any] = {"stream": True}
        if stop:
            call_kwargs["stop"] = stop
        call_kwargs.update(kwargs)
        for chunk in self.router.completion("primary", messages=litellm_messages, **call_kwargs):
            delta = chunk.choices[0].delta
            content = delta.content or ""
            tc_chunks = []
            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    idx = tc.index
                    tc_chunks.append(
                        {
                            "index": idx,
                            "id": tc.id or "",
                            "name": tc.function.name if tc.function else "",
                            "args": tc.function.arguments if tc.function else "",
                            "type": "tool_call_chunk",
                        }
                    )
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=content, tool_call_chunks=tc_chunks)
            )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        litellm_messages = _lc_messages_to_litellm(messages)
        call_kwargs: dict[str, Any] = {"stream": True}
        if stop:
            call_kwargs["stop"] = stop
        call_kwargs.update(kwargs)
        response = await self.router.acompletion(
            "primary", messages=litellm_messages, **call_kwargs
        )
        async for chunk in response:
            delta = chunk.choices[0].delta
            content = delta.content or ""
            tc_chunks = []
            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    idx = tc.index
                    tc_chunks.append(
                        {
                            "index": idx,
                            "id": tc.id or "",
                            "name": tc.function.name if tc.function else "",
                            "args": tc.function.arguments if tc.function else "",
                            "type": "tool_call_chunk",
                        }
                    )
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=content, tool_call_chunks=tc_chunks)
            )
