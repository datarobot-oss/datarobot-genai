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

# ruff: noqa: I001
import json
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import patch

from ag_ui.core import AssistantMessage
from ag_ui.core import SystemMessage as AgSystemMessage
from ag_ui.core.types import FunctionCall as AgFunctionCall
from ag_ui.core.types import ToolCall as AgToolCall
from ag_ui.core.types import ToolMessage as AgToolMessage

from crewai.types.streaming import CrewStreamingOutput
import pytest
from ag_ui.core import RunAgentInput
from ag_ui.core import RunFinishedEvent
from ag_ui.core import RunStartedEvent
from ag_ui.core import TextMessageChunkEvent
from ag_ui.core import UserMessage
from crewai import CrewOutput
from ragas import MultiTurnSample
from ragas.messages import AIMessage
from ragas.messages import HumanMessage

from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.crewai.agent import CrewAIAgent


# --- Test helpers ---


class CrewForTest:
    def __init__(self, output: CrewOutput | None = None, *args: Any, **kwargs: Any):
        self.args = args
        self.kwargs = kwargs
        self.output = output

    async def kickoff_async(self, *, inputs: dict[str, Any]) -> CrewOutput | CrewStreamingOutput:  # type: ignore[name-defined]
        return self.output or CrewOutput(raw="final-output")


class ListenerForTest:
    def __init__(self, messages: list[Any] | None = None):
        self.messages = messages or []
        self.called_setup = False

    def setup_listeners(self, crewai_event_bus: Any) -> None:
        self.called_setup = True


@pytest.fixture
def mock_ragas_event_listener() -> ListenerForTest:
    event_listener = ListenerForTest(
        messages=[HumanMessage(content="hi"), AIMessage(content="there")]
    )
    with patch(
        "datarobot_genai.crewai.agent.CrewAIRagasEventListener"
    ) as mock_ragas_event_listener:
        mock_ragas_event_listener.return_value = event_listener
        yield event_listener


class AgentForTest(CrewAIAgent):
    def __init__(self, crew_output: CrewOutput | None = None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.crew_output = crew_output

    @property
    def agents(self) -> list[Any]:
        return [object()]

    @property
    def tasks(self) -> list[Any]:
        return [object()]

    def make_kickoff_inputs(self, user_prompt_content: str) -> dict[str, Any]:
        return {"topic": user_prompt_content}

    def crew(self) -> Any:
        return CrewForTest(self.crew_output)


# --- Tests for create_pipeline_interactions_from_messages ---


def test_create_pipeline_interactions_from_messages_none() -> None:
    assert CrewAIAgent.create_pipeline_interactions_from_messages(None) is None


def test_create_pipeline_interactions_from_messages_returns_sample() -> None:
    msgs = [HumanMessage(content="hi")]
    sample = CrewAIAgent.create_pipeline_interactions_from_messages(msgs)
    assert sample is not None
    assert sample.user_input == msgs


# --- Tests for CrewAIAgent invoke ---


@pytest.fixture
def run_agent_input() -> RunAgentInput:
    return RunAgentInput(
        messages=[UserMessage(content="{}", id="message_id")],
        tools=[],
        forwarded_props=dict(model="m", authorization_context={}, forwarded_headers={}),
        thread_id="thread_id",
        run_id="run_id",
        state={},
        context=[],
    )


async def test_invoke(run_agent_input, mock_ragas_event_listener) -> None:
    # GIVEN agent with predefined crew output and forwarded headers
    out = CrewOutput(
        raw="agent result",
        token_usage=UsageMetrics(completion_tokens=1, prompt_tokens=2, total_tokens=3),
    )
    forwarded_headers = {"header-name": "header-value"}
    agent = AgentForTest(
        out,
        api_base="https://x/",
        api_key="k",
        verbose=True,
        forwarded_headers=forwarded_headers,
    )

    # WHEN invoke agent is called
    gen = agent.invoke(run_agent_input)

    # THEN response is an async generator
    assert isinstance(gen, AsyncGenerator)
    events = [event async for event in gen]

    # THEN ragas event listener was setup
    assert mock_ragas_event_listener.called_setup

    # THEN events contain AG-UI lifecycle and text message chunk events
    # RunStarted, TextMessageChunk, RunFinished
    assert len(events) == 3

    # THEN first event is RunStartedEvent
    assert isinstance(events[0][0], RunStartedEvent)

    # THEN text message chunk contains the agent result
    assert isinstance(events[1][0], TextMessageChunkEvent)
    assert events[1][0].delta == "agent result"

    # THEN last event is RunFinishedEvent with pipeline interactions
    run_finished_event, pipeline_interactions, usage = events[-1]
    assert isinstance(run_finished_event, RunFinishedEvent)

    # THEN pipeline interactions is not None and is a MultiTurnSample
    assert pipeline_interactions is not None
    assert isinstance(pipeline_interactions, MultiTurnSample)

    # THEN pipeline interactions has expected events
    assert pipeline_interactions.user_input == [
        HumanMessage(content="hi"),
        AIMessage(content="there", metadata=None, type="ai", tool_calls=None),
    ]

    # THEN usage is the expected usage
    assert usage == {"completion_tokens": 1, "prompt_tokens": 2, "total_tokens": 3}


async def test_invoke_does_not_include_chat_history_by_default(
    mock_ragas_event_listener, run_agent_input_with_history
) -> None:
    captured_inputs: dict[str, Any] = {}

    class CapturingCrew(CrewForTest):
        def kickoff_async(self, *, inputs: dict[str, Any]) -> CrewOutput | CrewStreamingOutput:  # type: ignore[override]
            captured_inputs.update(inputs)
            return super().kickoff_async(inputs=inputs)

    out = CrewOutput(raw="agent result")
    agent = AgentForTest(out, api_base="https://x/", api_key="k", verbose=False)
    agent.crew = lambda: CapturingCrew(out)  # type: ignore[assignment]

    _ = [event async for event in agent.invoke(run_agent_input_with_history)]

    assert captured_inputs["topic"] == "Follow-up"
    assert "chat_history" not in captured_inputs


async def test_invoke_injects_crew_chat_messages_for_multi_turn(
    mock_ragas_event_listener, run_agent_input_with_history
) -> None:
    """Multi-turn history is injected as crew_chat_messages JSON, not as a chat_history string."""
    captured_inputs: dict[str, Any] = {}

    class CapturingCrew(CrewForTest):
        async def kickoff_async(
            self, *, inputs: dict[str, Any]
        ) -> CrewOutput | CrewStreamingOutput:
            captured_inputs.update(inputs)
            return await super().kickoff_async(inputs=inputs)

    out = CrewOutput(raw="agent result")
    agent = AgentForTest(out, api_base="https://x/", api_key="k", verbose=False)
    agent.crew = lambda: CapturingCrew(out)  # type: ignore[assignment]

    _ = [event async for event in agent.invoke(run_agent_input_with_history)]

    assert "crew_chat_messages" in captured_inputs
    chat_msgs_raw = captured_inputs["crew_chat_messages"]
    assert isinstance(chat_msgs_raw, str)
    chat_msgs = json.loads(chat_msgs_raw)
    assert len(chat_msgs) > 0
    assert all(isinstance(m, dict) and "role" in m for m in chat_msgs)


@pytest.fixture
def run_agent_input_multi_turn() -> RunAgentInput:
    return RunAgentInput(
        messages=[
            AgSystemMessage(id="sys_1", content="You are helpful."),
            UserMessage(id="user_1", content="search cats"),
            AssistantMessage(
                id="asst_1",
                content=None,
                tool_calls=[
                    AgToolCall(
                        id="call_1",
                        function=AgFunctionCall(name="search", arguments='{"q": "cats"}'),
                    )
                ],
            ),
            AgToolMessage(id="tool_1", content="found cats", tool_call_id="call_1"),
            UserMessage(id="user_2", content="tell me more"),
        ],
        tools=[],
        forwarded_props=dict(model="m", authorization_context={}, forwarded_headers={}),
        thread_id="thread_id",
        run_id="run_id",
        state={},
        context=[],
    )


async def test_invoke_multi_turn_injects_crew_chat_messages(
    mock_ragas_event_listener, run_agent_input_multi_turn
) -> None:
    """Multi-turn conversations inject crew_chat_messages into kickoff inputs."""
    captured_inputs: dict[str, Any] = {}

    class CapturingCrew(CrewForTest):
        async def kickoff_async(
            self, *, inputs: dict[str, Any]
        ) -> CrewOutput | CrewStreamingOutput:
            captured_inputs.update(inputs)
            return await super().kickoff_async(inputs=inputs)

    out = CrewOutput(raw="agent result")
    agent = AgentForTest(out, api_base="https://x/", api_key="k", verbose=False)
    agent.crew = lambda: CapturingCrew(out)  # type: ignore[assignment]

    _ = [event async for event in agent.invoke(run_agent_input_multi_turn)]

    # crew_chat_messages should be injected as a JSON string
    assert "crew_chat_messages" in captured_inputs
    chat_msgs_raw = captured_inputs["crew_chat_messages"]
    assert isinstance(chat_msgs_raw, str)
    chat_msgs = json.loads(chat_msgs_raw)
    assert len(chat_msgs) > 0
    assert all(isinstance(m, dict) and "role" in m for m in chat_msgs)
