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

from collections.abc import AsyncGenerator
from typing import Any

import pytest
from ag_ui.core import AssistantMessage
from ag_ui.core import EventType
from ag_ui.core import RunAgentInput
from ag_ui.core import RunFinishedEvent
from ag_ui.core import RunStartedEvent
from ag_ui.core import SystemMessage
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from ag_ui.core import Tool
from ag_ui.core import ToolMessage
from ag_ui.core import UserMessage
from ragas import MultiTurnSample

from datarobot_genai.core.agents import BaseAgent
from datarobot_genai.core.agents import InvokeReturn
from datarobot_genai.core.agents import UsageMetrics
from datarobot_genai.core.chat.completions import agent_chat_completion_wrapper
from datarobot_genai.core.chat.completions import convert_chat_completion_params_to_run_agent_input
from datarobot_genai.core.chat.completions import is_streaming


def test_is_streaming_false_by_default() -> None:
    # GIVEN no explicit stream flag
    params: dict[str, Any] = {"messages": []}

    # WHEN checking streaming
    # THEN default is False
    assert is_streaming(params) is False


def test_is_streaming_true_when_bool_true() -> None:
    # GIVEN a boolean True stream flag
    params = {"stream": True}

    # WHEN checking streaming
    # THEN result is True
    assert is_streaming(params) is True


@pytest.mark.parametrize("value", ["true", "TRUE", "TrUe"])  # GIVEN truthy string stream flag
def test_is_streaming_true_when_string_true_case_insensitive(value: str) -> None:
    params = {"stream": value}

    # WHEN checking streaming
    # THEN result is True
    assert is_streaming(params) is True


@pytest.mark.parametrize("value", ["false", "FALSE", "FaLsE"])  # GIVEN falsy string stream flag
def test_is_streaming_false_when_string_false_case_insensitive(value: str) -> None:
    params = {"stream": value}

    # WHEN checking streaming
    # THEN result is False
    assert is_streaming(params) is False


def test_convert_chat_completion_params_to_run_agent_input() -> None:
    # GIVEN a chat completion parameters
    params = {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm good, thank you!"},
            {
                "role": "tool",
                "content": "The weather in Tokyo is sunny.",
                "tool_call_id": "tool_call_id_1",
            },
            {"role": "system", "content": "You are a helpful assistant."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a given city",
                    "parameters": {"city": "string"},
                },
            },
        ],
        "model": "test-model",
        "authorization_context": {"api_key": "test-api-key"},
        "forwarded_headers": {"X-Forwarded-For": "127.0.0.1"},
    }

    # WHEN converting to run agent input
    run_agent_input = convert_chat_completion_params_to_run_agent_input(params)

    # THEN the run agent input is correct
    assert run_agent_input.messages == [
        UserMessage(id="message_0", content="Hello, how are you?"),
        AssistantMessage(id="message_1", content="I'm good, thank you!"),
        ToolMessage(
            id="message_2", content="The weather in Tokyo is sunny.", tool_call_id="tool_call_id_1"
        ),
        SystemMessage(id="message_3", content="You are a helpful assistant."),
    ]
    assert run_agent_input.tools == [
        Tool(
            name="get_weather",
            description="Get the weather for a given city",
            parameters={"city": "string"},
        ),
    ]
    assert run_agent_input.forwarded_props == {
        "model": "test-model",
        "authorization_context": {"api_key": "test-api-key"},
        "forwarded_headers": {"X-Forwarded-For": "127.0.0.1"},
    }


class TestAGUIAgent(BaseAgent):
    async def invoke(self, run_agent_input: RunAgentInput) -> InvokeReturn:
        events = [
            (
                RunStartedEvent(thread_id=run_agent_input.thread_id, run_id=run_agent_input.run_id),
                None,
                UsageMetrics(),
            ),
            (TextMessageStartEvent(message_id="message_0"), None, UsageMetrics()),
            (
                TextMessageContentEvent(
                    message_id="message_0",
                    delta="Hey I am agent 1. I am going to help you with your question.",
                ),
                None,
                UsageMetrics(),
            ),
            (TextMessageEndEvent(message_id="message_id_0"), None, UsageMetrics()),
            (TextMessageStartEvent(message_id="message_id_1"), None, UsageMetrics()),
            (
                TextMessageContentEvent(
                    message_id="message_1",
                    delta="Hey I am agent 2. I am going to help you with your question.",
                ),
                None,
                UsageMetrics(),
            ),
            (TextMessageEndEvent(message_id="message_id_1"), None, UsageMetrics()),
            (
                RunFinishedEvent(
                    thread_id=run_agent_input.thread_id, run_id=run_agent_input.run_id
                ),
                MultiTurnSample(
                    user_input=[{"id": "message_0", "content": "Hello, how are you?"}],
                ),
                UsageMetrics(
                    completion_tokens=100,
                    prompt_tokens=100,
                    total_tokens=100,
                ),
            ),
        ]

        for event in events:
            yield event


async def test_agent_chat_completion_wrapper_streaming() -> None:
    # GIVEN a chat completion parameters
    params = {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
        ],
        "stream": True,
    }

    # WHEN calling the agent chat completion wrapper
    generator = await agent_chat_completion_wrapper(TestAGUIAgent(), params)

    # THEN the generator returns an async generator
    assert isinstance(generator, AsyncGenerator)

    # THEN the generator returns the expected events in the correct order
    all_events = [event async for event in generator]
    assert len(all_events) == 8
    assert all_events[0][0].type == EventType.RUN_STARTED
    assert all_events[1][0].type == EventType.TEXT_MESSAGE_START
    assert all_events[2][0].type == EventType.TEXT_MESSAGE_CONTENT
    assert all_events[3][0].type == EventType.TEXT_MESSAGE_END
    assert all_events[4][0].type == EventType.TEXT_MESSAGE_START
    assert all_events[5][0].type == EventType.TEXT_MESSAGE_CONTENT
    assert all_events[6][0].type == EventType.TEXT_MESSAGE_END
    assert all_events[7][0].type == EventType.RUN_FINISHED

    # THEN the final response contains additional metadata
    assert all_events[7][1] is not None
    assert all_events[7][2] is not None
    assert all_events[7][2]["total_tokens"] == 100
    assert all_events[7][2]["prompt_tokens"] == 100
    assert all_events[7][2]["completion_tokens"] == 100


async def test_agent_chat_completion_wrapper_non_streaming() -> None:
    # GIVEN a chat completion parameters
    params = {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
        ],
        "stream": False,
    }

    # WHEN calling the agent chat completion wrapper
    response, pipeline_interactions, usage_metrics = await agent_chat_completion_wrapper(
        TestAGUIAgent(), params
    )

    # THEN the response is the expected response
    assert response == "Hey I am agent 2. I am going to help you with your question."

    # THEN the pipeline interactions are not None
    assert pipeline_interactions is not None

    # THEN the usage metrics are not None
    assert usage_metrics is not None
    assert usage_metrics["total_tokens"] == 100
    assert usage_metrics["prompt_tokens"] == 100
    assert usage_metrics["completion_tokens"] == 100
