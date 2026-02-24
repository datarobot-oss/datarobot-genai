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

from ag_ui.core import TextMessageChunkEvent
from ag_ui.core import Tool
from ag_ui.core import UserMessage
from langchain_core.messages import ToolMessage as LangchainToolMessage
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import Message
from nat.data_models.api_server import Usage

from datarobot_genai.dragent.converters import convert_chat_request_to_run_agent_input
from datarobot_genai.dragent.converters import (
    convert_chat_response_chunk_to_dragent_chat_response_chunk,
)
from datarobot_genai.dragent.converters import convert_chat_response_to_dragent_chat_response
from datarobot_genai.dragent.converters import convert_dragent_event_response_to_dragent_chunk
from datarobot_genai.dragent.converters import convert_dragent_run_agent_input_to_chat_request
from datarobot_genai.dragent.converters import (
    convert_dragent_run_agent_input_to_chat_request_or_message,
)
from datarobot_genai.dragent.converters import convert_str_to_dragent_chat_response
from datarobot_genai.dragent.converters import convert_str_to_dragent_chat_response_chunk
from datarobot_genai.dragent.converters import convert_str_to_dragent_event_response
from datarobot_genai.dragent.converters import convert_tool_message_to_str
from datarobot_genai.dragent.request import DRAgentRunAgentInput
from datarobot_genai.dragent.response import DRAgentChatResponse
from datarobot_genai.dragent.response import DRAgentChatResponseChunk
from datarobot_genai.dragent.response import DRAgentEventResponse

# --- Input converters: AG-UI -> NAT ---


def _make_dragent_run_agent_input(
    *,
    messages=None,
    tools=None,
    forwarded_props=None,
):
    """Build DRAgentRunAgentInput with required RunAgentInput fields."""
    return DRAgentRunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        messages=messages or [UserMessage(id="1", content="Hello")],
        tools=tools or [],
        context=[],
        forwarded_props=forwarded_props or {},
        state={},
    )


def test_convert_dragent_run_agent_input_to_chat_request() -> None:
    # GIVEN a DRAgentRunAgentInput with messages, tools, and forwarded_props
    input_obj = _make_dragent_run_agent_input(
        messages=[UserMessage(id="1", content="Hello")],
        tools=[
            Tool(
                name="get_weather",
                description="Get weather",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            )
        ],
        forwarded_props={"model": "test-model", "stream": False},
    )

    # WHEN converting to ChatRequest
    result = convert_dragent_run_agent_input_to_chat_request(input_obj)

    # THEN result is a ChatRequest with messages, tools, and stream=True
    assert isinstance(result, ChatRequest)
    assert result.stream is True
    assert len(result.messages) == 1
    assert result.messages[0].role == "user"
    assert result.messages[0].content == "Hello"
    assert len(result.tools) == 1
    first_tool = result.tools[0]
    # ag_ui Tool.model_dump() may use "function" key or flat "name"
    tool_name = first_tool.get("function", {}).get("name") or first_tool.get("name")
    assert tool_name == "get_weather"


def test_convert_dragent_run_agent_input_to_chat_request_with_empty_forwarded_props() -> None:
    # GIVEN a DRAgentRunAgentInput with empty forwarded_props
    input_obj = _make_dragent_run_agent_input(
        messages=[UserMessage(id="1", content="Hi")],
        tools=[],
        forwarded_props={},
    )

    # WHEN converting to ChatRequest
    result = convert_dragent_run_agent_input_to_chat_request(input_obj)

    # THEN result has messages and stream=True
    assert result.stream is True
    assert len(result.messages) == 1
    assert result.messages[0].content == "Hi"


def test_convert_dragent_run_agent_input_to_chat_request_forwarded_props_error() -> None:
    # GIVEN a DRAgentRunAgentInput whose forwarded_props raises when converted to dict
    input_obj = _make_dragent_run_agent_input(
        messages=[UserMessage(id="1", content="Hi")],
        tools=[],
        forwarded_props={},
    )

    class BadForwardedProps:
        """Object that raises when dict() tries to iterate over it."""

        def __iter__(self):
            raise ValueError("bad")

    input_obj.forwarded_props = BadForwardedProps()

    # WHEN converting to ChatRequest (converter catches Exception and uses empty dict)
    result = convert_dragent_run_agent_input_to_chat_request(input_obj)

    # THEN conversion still succeeds with messages and stream=True
    assert isinstance(result, ChatRequest)
    assert result.stream is True
    assert len(result.messages) == 1


def test_convert_dragent_run_agent_input_to_chat_request_or_message() -> None:
    # GIVEN a DRAgentRunAgentInput
    input_obj = _make_dragent_run_agent_input(
        messages=[UserMessage(id="1", content="Hello")],
        tools=[],
        forwarded_props={},
    )

    # WHEN converting to ChatRequestOrMessage
    result = convert_dragent_run_agent_input_to_chat_request_or_message(input_obj)

    # THEN result is ChatRequestOrMessage wrapping the same data as ChatRequest
    assert isinstance(result, ChatRequestOrMessage)
    # ChatRequestOrMessage can be validated from chat request dump
    assert hasattr(result, "model_dump") or hasattr(result, "messages")


# --- Input converters: NAT -> AG-UI ---


def test_convert_chat_request_to_run_agent_input() -> None:
    # GIVEN a ChatRequest (NAT format)
    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        stream=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )

    # WHEN converting to RunAgentInput
    result = convert_chat_request_to_run_agent_input(request)

    # THEN result has messages and tools
    assert len(result.messages) == 1
    assert result.messages[0].content == "Hello"
    assert len(result.tools) == 1
    assert result.tools[0].name == "get_weather"


# --- Output converters: str -> DRAgentChatResponse ---


def test_convert_str_to_dragent_chat_response() -> None:
    # GIVEN a response string
    text = "Hello, world!"

    # WHEN converting to DRAgentChatResponse (NAT may return base ChatResponse)
    result = convert_str_to_dragent_chat_response(text)

    # THEN result has the expected content and zero usage
    assert isinstance(result, (DRAgentChatResponse, ChatResponse))
    assert result.usage is not None
    assert result.usage.prompt_tokens == 0
    assert result.usage.completion_tokens == 0
    assert result.usage.total_tokens == 0
    assert result.choices and result.choices[0].message.content == text


# --- Output converters: ChatResponse -> DRAgentChatResponse ---


def test_convert_chat_response_to_dragent_chat_response() -> None:
    # GIVEN a NAT ChatResponse (build via DRAgentChatResponse then dump to get valid shape)
    source = DRAgentChatResponse.from_string(
        "test content",
        usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    response = ChatResponse.model_validate(source.model_dump())

    # WHEN converting to DRAgentChatResponse
    result = convert_chat_response_to_dragent_chat_response(response)

    # THEN result matches structure and usage
    assert isinstance(result, DRAgentChatResponse)
    assert result.usage.prompt_tokens == 1
    assert result.usage.completion_tokens == 2
    assert result.usage.total_tokens == 3


# --- Output converters: str -> DRAgentChatResponseChunk ---


def test_convert_str_to_dragent_chat_response_chunk() -> None:
    # GIVEN a chunk string
    delta = "Hello"

    # WHEN converting to DRAgentChatResponseChunk
    result = convert_str_to_dragent_chat_response_chunk(delta)

    # THEN result is a chunk with delta, usage, and events
    assert isinstance(result, DRAgentChatResponseChunk)
    assert result.usage is not None
    assert result.usage.prompt_tokens == 0
    assert result.usage.completion_tokens == 0
    assert result.events is not None
    assert len(result.events) == 1
    assert result.events[0].delta == delta


# --- Output converters: ChatResponseChunk -> DRAgentChatResponseChunk ---


def test_convert_chat_response_chunk_to_dragent_chat_response_chunk() -> None:
    # GIVEN a NAT ChatResponseChunk (build from DRAgentChatResponseChunk for valid shape)
    source = DRAgentChatResponseChunk.create_streaming_chunk(
        "chunk content",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    source.events = []
    chunk = ChatResponseChunk.model_validate(source.model_dump())

    # WHEN converting to DRAgentChatResponseChunk
    result = convert_chat_response_chunk_to_dragent_chat_response_chunk(chunk)

    # THEN result matches
    assert isinstance(result, DRAgentChatResponseChunk)
    assert result.usage.total_tokens == 15


# --- Output converters: str -> DRAgentEventResponse ---


def test_convert_str_to_dragent_event_response() -> None:
    # GIVEN a response string
    delta = "streaming delta"

    # WHEN converting to DRAgentEventResponse
    result = convert_str_to_dragent_event_response(delta)

    # THEN result has delta, default usage_metrics, and events
    assert isinstance(result, DRAgentEventResponse)
    assert result.delta == delta
    assert result.usage_metrics is not None
    assert result.usage_metrics["prompt_tokens"] == 0
    assert result.usage_metrics["completion_tokens"] == 0
    assert result.usage_metrics["total_tokens"] == 0
    assert result.events is not None
    assert len(result.events) == 1
    assert result.events[0].delta == delta


# --- Output converters: DRAgentEventResponse -> DRAgentChatResponseChunk ---


def test_convert_dragent_event_response_to_dragent_chunk_with_usage() -> None:
    # GIVEN a DRAgentEventResponse with usage_metrics
    response = DRAgentEventResponse(
        delta="content",
        usage_metrics={
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
        pipeline_interactions=None,
        events=[],
    )

    # WHEN converting to DRAgentChatResponseChunk
    result = convert_dragent_event_response_to_dragent_chunk(response)

    # THEN chunk has correct usage and content
    assert isinstance(result, DRAgentChatResponseChunk)
    assert result.usage is not None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 20
    assert result.usage.total_tokens == 30


def test_convert_dragent_event_response_to_dragent_chunk_without_usage() -> None:
    # GIVEN a DRAgentEventResponse with no usage_metrics
    response = DRAgentEventResponse(
        delta="content",
        usage_metrics=None,
        pipeline_interactions=None,
        events=[],
    )

    # WHEN converting to DRAgentChatResponseChunk
    result = convert_dragent_event_response_to_dragent_chunk(response)

    # THEN chunk has zero usage
    assert isinstance(result, DRAgentChatResponseChunk)
    assert result.usage is not None
    assert result.usage.prompt_tokens == 0
    assert result.usage.completion_tokens == 0
    assert result.usage.total_tokens == 0


def test_convert_dragent_event_response_to_dragent_chunk_preserves_events() -> None:
    response = DRAgentEventResponse(
        delta="d",
        usage_metrics=None,
        pipeline_interactions=None,
        events=[TextMessageChunkEvent(delta="d")],
    )

    # WHEN converting to DRAgentChatResponseChunk
    result = convert_dragent_event_response_to_dragent_chunk(response)

    # THEN chunk has the same events
    assert result.events is not None
    assert len(result.events) == 1
    assert result.events[0].delta == "d"


# --- Various converters ---


def test_convert_tool_message_to_str() -> None:
    # GIVEN a langchain ToolMessage
    message = LangchainToolMessage(content="tool result", tool_call_id="tc_1")

    # WHEN converting to str
    result = convert_tool_message_to_str(message)

    # THEN result is the message content
    assert result == "tool result"
