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

import datetime

from ag_ui.core import StepStartedEvent
from ag_ui.core import TextMessageChunkEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import Tool
from ag_ui.core import ToolCallArgsEvent
from ag_ui.core import ToolCallChunkEvent
from ag_ui.core import ToolCallEndEvent
from ag_ui.core import ToolCallStartEvent
from ag_ui.core import UserMessage
from langchain_core.messages import ToolMessage as LangchainToolMessage
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import ChatResponseChunkChoice
from nat.data_models.api_server import ChoiceDelta
from nat.data_models.api_server import Message

from datarobot_genai.dragent.frontends.converters import aggregate_dragent_event_responses
from datarobot_genai.dragent.frontends.converters import convert_chat_request_to_run_agent_input
from datarobot_genai.dragent.frontends.converters import (
    convert_dragent_event_response_to_chat_response_chunk,
)
from datarobot_genai.dragent.frontends.converters import (
    convert_dragent_event_response_to_openai_chat_completion_chunk,
)
from datarobot_genai.dragent.frontends.converters import convert_dragent_event_response_to_str
from datarobot_genai.dragent.frontends.converters import (
    convert_dragent_run_agent_input_to_chat_request,
)
from datarobot_genai.dragent.frontends.converters import (
    convert_dragent_run_agent_input_to_chat_request_or_message,
)
from datarobot_genai.dragent.frontends.converters import (
    convert_nat_chat_response_chunk_to_openai_chat_completion_chunk,
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
    # GIVEN a response string
    delta = "streaming delta"

    # WHEN converting to DRAgentEventResponse
    result = convert_str_to_dragent_event_response(delta)

    # THEN result has default usage_metrics and events
    assert isinstance(result, DRAgentEventResponse)
    assert result.usage_metrics is not None
    assert result.usage_metrics["prompt_tokens"] == 0
    assert result.usage_metrics["completion_tokens"] == 0
    assert result.usage_metrics["total_tokens"] == 0
    assert result.events is not None
    assert len(result.events) == 1

    # THEN event is a CustomEvent with the correct name and value
    assert result.events[0].value["delta"] == delta
    assert result.events[0].name == "DEFAULT_NAT_RESPONSE"


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


def test_aggregate_dragent_event_responses_keeps_last_datarobot_moderations() -> None:
    # GIVEN pass-through chunks without moderation metadata and a text chunk with guards output
    r1 = DRAgentEventResponse(events=[TextMessageContentEvent(message_id="m1", delta="Hello ")])
    r2 = DRAgentEventResponse(
        events=[TextMessageContentEvent(message_id="m1", delta="world")],
        datarobot_moderations={"score": 0.1},
    )

    result = aggregate_dragent_event_responses([r1, r2])

    assert result.datarobot_moderations == {"score": 0.1}


def test_aggregate_dragent_event_responses_sums_usage_metrics() -> None:
    # GIVEN multiple batches each reporting token usage (e.g. multi-step invoke yields)
    r1 = DRAgentEventResponse(
        events=[TextMessageContentEvent(message_id="m1", delta="a")],
        usage_metrics={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    )
    r2 = DRAgentEventResponse(
        events=[TextMessageContentEvent(message_id="m1", delta="b")],
        usage_metrics={"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
    )

    result = aggregate_dragent_event_responses([r1, r2])

    assert result.usage_metrics == {
        "prompt_tokens": 5,
        "completion_tokens": 7,
        "total_tokens": 12,
    }


def test_aggregate_dragent_event_responses_last_model_and_original_chunk() -> None:
    # GIVEN streaming-style batches with model / NAT chunk metadata on later items
    chunk_first = _make_chat_response_chunk(content="a", chunk_id="chunk-a")
    chunk_last = _make_chat_response_chunk(content="b", chunk_id="chunk-b")
    r1 = DRAgentEventResponse(
        events=[TextMessageContentEvent(message_id="m1", delta="a")],
        model="first-model",
        original_chunk=chunk_first,
    )
    r2 = DRAgentEventResponse(
        events=[TextMessageContentEvent(message_id="m1", delta="b")],
        model="last-model",
        original_chunk=chunk_last,
    )

    result = aggregate_dragent_event_responses([r1, r2])

    assert result.model == "last-model"
    assert result.original_chunk is chunk_last


def test_aggregate_dragent_event_responses_empty() -> None:
    # GIVEN an empty list
    result = aggregate_dragent_event_responses([])

    # THEN result has no events
    assert result.events == []
    assert result.usage_metrics is None
    assert result.model is None
    assert result.original_chunk is None
    assert result.datarobot_moderations is None


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


# --- convert_dragent_event_response_to_chat_response_chunk ---


def _make_chat_response_chunk(
    *, content: str, chunk_id: str = "existing-chunk"
) -> ChatResponseChunk:
    return ChatResponseChunk(
        id=chunk_id,
        choices=[
            ChatResponseChunkChoice(index=0, delta=ChoiceDelta(content=content)),
        ],
        created=datetime.datetime.now(datetime.UTC),
    )


def test_convert_dragent_event_response_to_chat_response_chunk_from_text_events() -> None:
    # GIVEN a response with no original_chunk and multiple text content events
    response = DRAgentEventResponse(
        original_chunk=None,
        events=[
            TextMessageContentEvent(message_id="m1", delta="Hello "),
            TextMessageContentEvent(message_id="m1", delta="world"),
        ],
    )

    # WHEN converting to ChatResponseChunk
    result = convert_dragent_event_response_to_chat_response_chunk(response)

    # THEN a new chunk is built with concatenated content and OpenAI-like shape
    assert isinstance(result, ChatResponseChunk)
    assert len(result.choices) == 1
    assert result.choices[0].index == 0
    assert result.choices[0].delta.content == "Hello world"
    assert len(result.id) == 32
    assert result.created.tzinfo == datetime.UTC


def test_convert_dragent_event_response_to_chat_response_chunk_includes_chunk_events() -> None:
    # GIVEN content split across TextMessageContentEvent and TextMessageChunkEvent
    response = DRAgentEventResponse(
        original_chunk=None,
        events=[
            TextMessageContentEvent(message_id="m1", delta="a"),
            TextMessageChunkEvent(message_id="m1", delta="b"),
        ],
    )

    # WHEN converting
    result = convert_dragent_event_response_to_chat_response_chunk(response)

    # THEN both event types contribute to delta content
    assert result.choices[0].delta.content == "ab"


def test_convert_dragent_event_response_to_chat_response_chunk_ignores_non_text_events() -> None:
    # GIVEN text events interleaved with non-text AG-UI events
    response = DRAgentEventResponse(
        original_chunk=None,
        events=[
            TextMessageContentEvent(message_id="m1", delta="x"),
            StepStartedEvent(step_name="s"),
            TextMessageChunkEvent(message_id="m1", delta="y"),
        ],
    )

    # WHEN converting
    result = convert_dragent_event_response_to_chat_response_chunk(response)

    # THEN only text-bearing events are concatenated
    assert result.choices[0].delta.content == "xy"


def test_convert_dragent_event_response_to_chat_response_chunk_empty_events() -> None:
    # GIVEN no original_chunk and no text events
    response = DRAgentEventResponse(original_chunk=None, events=[])

    # WHEN converting
    result = convert_dragent_event_response_to_chat_response_chunk(response)

    # THEN content is empty but structure is still valid
    assert result.choices[0].delta.content == ""


def test_convert_dragent_event_response_to_chat_response_chunk_returns_original_chunk() -> None:
    # GIVEN a response that already carries a NAT ChatResponseChunk
    existing = _make_chat_response_chunk(content="from-stream")
    response = DRAgentEventResponse(
        original_chunk=existing,
        events=[TextMessageContentEvent(message_id="m1", delta="ignored when chunk set")],
    )

    # WHEN converting
    result = convert_dragent_event_response_to_chat_response_chunk(response)

    # THEN the existing chunk is returned unchanged (events are not merged)
    assert result is existing
    assert result.choices[0].delta.content == "from-stream"


# --- Tool-call event conversion ------------------------------------------


def test_convert_dragent_event_response_to_chat_response_chunk_tool_call_start() -> None:
    # GIVEN a ToolCallStartEvent
    response = DRAgentEventResponse(
        original_chunk=None,
        events=[ToolCallStartEvent(tool_call_id="tc_1", tool_call_name="lookup")],
    )

    # WHEN converting
    result = convert_dragent_event_response_to_chat_response_chunk(response)

    # THEN the chunk carries the OpenAI "first chunk" shape:
    # id/type set, function.name set, content null, index 0.
    assert result.choices[0].delta.content is None
    tool_calls = result.choices[0].delta.tool_calls
    assert tool_calls is not None and len(tool_calls) == 1
    assert tool_calls[0].index == 0
    assert tool_calls[0].id == "tc_1"
    assert tool_calls[0].type == "function"
    assert tool_calls[0].function is not None
    assert tool_calls[0].function.name == "lookup"
    assert tool_calls[0].function.arguments is None


def test_convert_dragent_event_response_to_chat_response_chunk_tool_call_args() -> None:
    # GIVEN a ToolCallArgsEvent (subsequent chunk)
    response = DRAgentEventResponse(
        original_chunk=None,
        events=[ToolCallArgsEvent(tool_call_id="tc_1", delta='{"q": "weat')],
    )

    # WHEN converting
    result = convert_dragent_event_response_to_chat_response_chunk(response)

    # THEN the chunk only carries function.arguments with id/type null and index 0
    tool_calls = result.choices[0].delta.tool_calls
    assert tool_calls is not None and len(tool_calls) == 1
    assert tool_calls[0].index == 0
    assert tool_calls[0].id is None
    assert tool_calls[0].type is None
    assert tool_calls[0].function is not None
    assert tool_calls[0].function.name is None
    assert tool_calls[0].function.arguments == '{"q": "weat'


def test_convert_dragent_event_response_to_chat_response_chunk_tool_call_chunk_with_id() -> None:
    # GIVEN a ToolCallChunkEvent that carries both id+name and arguments delta
    response = DRAgentEventResponse(
        original_chunk=None,
        events=[
            ToolCallChunkEvent(tool_call_id="tc_2", tool_call_name="search", delta='{"q":'),
        ],
    )

    # WHEN converting
    result = convert_dragent_event_response_to_chat_response_chunk(response)

    # THEN the chunk carries id/type and both name and arguments
    tool_calls = result.choices[0].delta.tool_calls
    assert tool_calls is not None and len(tool_calls) == 1
    assert tool_calls[0].id == "tc_2"
    assert tool_calls[0].type == "function"
    assert tool_calls[0].function is not None
    assert tool_calls[0].function.name == "search"
    assert tool_calls[0].function.arguments == '{"q":'


def test_convert_dragent_event_response_to_chat_response_chunk_tool_call_chunk_args_only() -> None:
    # GIVEN a subsequent ToolCallChunkEvent without an id
    response = DRAgentEventResponse(
        original_chunk=None,
        events=[ToolCallChunkEvent(delta='"weather"}')],
    )

    # WHEN converting
    result = convert_dragent_event_response_to_chat_response_chunk(response)

    # THEN the chunk has only function.arguments set; id/type are null and index is 0
    tool_calls = result.choices[0].delta.tool_calls
    assert tool_calls is not None and len(tool_calls) == 1
    assert tool_calls[0].index == 0
    assert tool_calls[0].id is None
    assert tool_calls[0].type is None
    assert tool_calls[0].function is not None
    assert tool_calls[0].function.arguments == '"weather"}'


def test_convert_dragent_event_response_to_chat_response_chunk_tool_call_end_is_skipped() -> None:
    # GIVEN only a ToolCallEndEvent (end is conveyed via finish_reason, not delta)
    response = DRAgentEventResponse(
        original_chunk=None,
        events=[ToolCallEndEvent(tool_call_id="tc_1")],
    )

    # WHEN converting
    result = convert_dragent_event_response_to_chat_response_chunk(response)

    # THEN no tool_calls are emitted and content reverts to the empty default
    assert result.choices[0].delta.tool_calls is None
    assert result.choices[0].delta.content == ""


# --- convert_dragent_event_response_to_openai_chat_completion_chunk ---


def test_convert_dragent_event_response_to_openai_chunk_from_text_events() -> None:
    response = DRAgentEventResponse(
        original_chunk=None,
        events=[
            TextMessageContentEvent(message_id="m1", delta="Hello "),
            TextMessageContentEvent(message_id="m1", delta="world"),
        ],
    )
    result = convert_dragent_event_response_to_openai_chat_completion_chunk(response)
    assert result.object == "chat.completion.chunk"
    assert result.choices[0].delta.content == "Hello world"
    assert isinstance(result.created, int)


def test_convert_dragent_event_response_to_openai_chunk_returns_mapped_original_chunk() -> None:
    existing = _make_chat_response_chunk(content="from-stream", chunk_id="nat-chunk-1")
    response = DRAgentEventResponse(
        original_chunk=existing,
        events=[TextMessageContentEvent(message_id="m1", delta="ignored")],
    )
    result = convert_dragent_event_response_to_openai_chat_completion_chunk(response)
    assert result.id == "nat-chunk-1"
    assert result.choices[0].delta.content == "from-stream"
    assert isinstance(result.created, int)


def test_openai_chunk_from_nat_matches_dragent_event_path_for_text() -> None:
    response = DRAgentEventResponse(
        original_chunk=None,
        events=[TextMessageContentEvent(message_id="m1", delta="same")],
    )
    nat = convert_dragent_event_response_to_chat_response_chunk(response)
    via_nat = convert_nat_chat_response_chunk_to_openai_chat_completion_chunk(nat)
    direct = convert_dragent_event_response_to_openai_chat_completion_chunk(response)
    assert direct.choices[0].delta.content == via_nat.choices[0].delta.content == "same"
