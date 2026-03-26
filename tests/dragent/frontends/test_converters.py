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

from ag_ui.core import TextMessageContentEvent
from ag_ui.core import Tool
from ag_ui.core import UserMessage
from langchain_core.messages import ToolMessage as LangchainToolMessage
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.api_server import Message

from datarobot_genai.dragent.frontends.converters import aggregate_dragent_event_responses
from datarobot_genai.dragent.frontends.converters import convert_chat_request_to_run_agent_input
from datarobot_genai.dragent.frontends.converters import convert_dragent_event_response_to_str
from datarobot_genai.dragent.frontends.converters import (
    convert_dragent_run_agent_input_to_chat_request,
)
from datarobot_genai.dragent.frontends.converters import (
    convert_dragent_run_agent_input_to_chat_request_or_message,
)
from datarobot_genai.dragent.frontends.converters import convert_str_to_dragent_event_response
from datarobot_genai.dragent.frontends.converters import convert_tool_message_to_str
from datarobot_genai.dragent.frontends.request import DRAgentRunAgentInput
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse

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


# --- Output converters: str -> DRAgentEventResponse ---


def test_convert_str_to_dragent_event_response() -> None:
    # WHEN converting a chunk
    result = convert_str_to_dragent_event_response("hello")

    # THEN only TextMessageContentEvent is emitted (stateless, no Start/End)
    assert isinstance(result, DRAgentEventResponse)
    assert len(result.events) == 1
    assert result.events[0].type.value == "TEXT_MESSAGE_CONTENT"
    assert result.events[0].delta == "hello"


# --- Various converters ---


def test_convert_tool_message_to_str() -> None:
    # GIVEN a langchain ToolMessage
    message = LangchainToolMessage(content="tool result", tool_call_id="tc_1")

    # WHEN converting to str
    result = convert_tool_message_to_str(message)

    # THEN result is the message content
    assert result == "tool result"


# --- aggregate_dragent_event_responses ---


def test_aggregate_dragent_event_responses_combines_events() -> None:
    # GIVEN two responses with events
    r1 = DRAgentEventResponse(events=[TextMessageContentEvent(message_id="m1", delta="Hello ")])
    r2 = DRAgentEventResponse(events=[TextMessageContentEvent(message_id="m1", delta="world")])

    # WHEN aggregating
    result = aggregate_dragent_event_responses([r1, r2])

    # THEN all events are combined in order
    assert isinstance(result, DRAgentEventResponse)
    assert len(result.events) == 2
    assert result.events[0].delta == "Hello "
    assert result.events[1].delta == "world"


def test_aggregate_dragent_event_responses_empty() -> None:
    # GIVEN an empty list
    result = aggregate_dragent_event_responses([])

    # THEN result has no events
    assert result.events == []


# --- convert_dragent_event_response_to_str ---


def test_convert_dragent_event_response_to_str_joins_text_deltas() -> None:
    # GIVEN a response with multiple TextMessageContentEvents
    response = DRAgentEventResponse(
        events=[
            TextMessageContentEvent(message_id="m1", delta="Hello "),
            TextMessageContentEvent(message_id="m1", delta="world"),
        ]
    )

    # WHEN converting to str
    result = convert_dragent_event_response_to_str(response)

    # THEN result is the concatenated text
    assert result == "Hello world"


def test_convert_dragent_event_response_to_str_ignores_non_text_events() -> None:
    # GIVEN a response mixing text events with non-text-bearing events
    from ag_ui.core import StepStartedEvent
    from ag_ui.core import TextMessageChunkEvent

    response = DRAgentEventResponse(
        events=[
            TextMessageContentEvent(message_id="m1", delta="Hi"),
            TextMessageChunkEvent(message_id="m1", delta=" there"),
            StepStartedEvent(step_name="some_step"),
        ]
    )

    # WHEN converting to str
    result = convert_dragent_event_response_to_str(response)

    # THEN TextMessageContentEvent and TextMessageChunkEvent deltas are included;
    # non-text events (e.g. StepStartedEvent) are ignored.
    assert result == "Hi there"


def test_convert_dragent_event_response_to_str_empty_events() -> None:
    # GIVEN a response with no events
    response = DRAgentEventResponse(events=[])

    # WHEN converting to str
    result = convert_dragent_event_response_to_str(response)

    # THEN result is empty string
    assert result == ""
