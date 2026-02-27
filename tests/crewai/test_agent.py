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
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import patch

import pytest
from ag_ui.core import RunAgentInput
from ag_ui.core import RunFinishedEvent
from ag_ui.core import RunStartedEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
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

    def kickoff(self, *, inputs: dict[str, Any]) -> CrewOutput:  # type: ignore[name-defined]
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
def patch_mcp_tools_context() -> None:
    with patch("datarobot_genai.crewai.agent.mcp_tools_context") as mock_mcp_tools_context:
        mock_mcp_tools_context.return_value.__enter__.return_value = ["tool1"]
        yield mock_mcp_tools_context


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


async def test_invoke(run_agent_input, patch_mcp_tools_context, mock_ragas_event_listener) -> None:
    # GIVEN agent with predefined crew output and forwarded headers
    out = CrewOutput(
        raw="agent result",
        token_usage=UsageMetrics(completion_tokens=1, prompt_tokens=2, total_tokens=3),
    )
    forwarded_headers = {"header-name": "header-value"}
    authorization_context = {"x-datarobot-api-key": "scoped-token-123"}
    agent = AgentForTest(
        out,
        api_base="https://x/",
        api_key="k",
        verbose=True,
        forwarded_headers=forwarded_headers,
        authorization_context=authorization_context,
    )

    # WHEN invoke agent is called
    gen = agent.invoke(run_agent_input)

    # THEN response is an async generator
    assert isinstance(gen, AsyncGenerator)
    events = [event async for event in gen]

    # THEN MCP context was called with forwarded headers and authorization context
    patch_mcp_tools_context.assert_called_once_with(
        authorization_context=authorization_context,
        forwarded_headers=forwarded_headers,
    )

    # THEN MCP tools are set
    assert agent.mcp_tools == ["tool1"]

    # THEN ragas event listener was setup
    assert mock_ragas_event_listener.called_setup

    # THEN events contain AG-UI lifecycle and text message events
    # RunStarted, TextMessageStart, TextMessageContent, TextMessageEnd, RunFinished
    assert len(events) == 5

    # THEN first event is RunStartedEvent
    assert isinstance(events[0][0], RunStartedEvent)

    # THEN text message events contain the agent result
    assert isinstance(events[1][0], TextMessageStartEvent)
    assert isinstance(events[2][0], TextMessageContentEvent)
    assert events[2][0].delta == "agent result"
    assert isinstance(events[3][0], TextMessageEndEvent)

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
    patch_mcp_tools_context, mock_ragas_event_listener, run_agent_input_with_history
) -> None:
    captured_inputs: dict[str, Any] = {}

    class CapturingCrew(CrewForTest):
        def kickoff(self, *, inputs: dict[str, Any]) -> CrewOutput:  # type: ignore[override]
            captured_inputs.update(inputs)
            return super().kickoff(inputs=inputs)

    out = CrewOutput(raw="agent result")
    agent = AgentForTest(out, api_base="https://x/", api_key="k", verbose=False)
    agent.crew = lambda: CapturingCrew(out)  # type: ignore[assignment]

    _ = [event async for event in agent.invoke(run_agent_input_with_history)]

    assert captured_inputs["topic"] == "Follow-up"
    assert "chat_history" not in captured_inputs


async def test_invoke_overwrites_blank_chat_history_placeholder(
    patch_mcp_tools_context, mock_ragas_event_listener, run_agent_input_with_history
) -> None:
    captured_inputs: dict[str, Any] = {}

    class CapturingCrew(CrewForTest):
        def kickoff(self, *, inputs: dict[str, Any]) -> CrewOutput:  # type: ignore[override]
            captured_inputs.update(inputs)
            return super().kickoff(inputs=inputs)

    class AgentWithPlaceholder(AgentForTest):
        def make_kickoff_inputs(self, user_prompt_content: str) -> dict[str, Any]:
            return {"topic": user_prompt_content, "chat_history": ""}

    out = CrewOutput(raw="agent result")
    agent = AgentWithPlaceholder(out, api_base="https://x/", api_key="k", verbose=False)
    agent.crew = lambda: CapturingCrew(out)  # type: ignore[assignment]

    _ = [event async for event in agent.invoke(run_agent_input_with_history)]

    history = captured_inputs["chat_history"]
    assert isinstance(history, str)
    assert history.strip() != ""
    assert "Prior conversation:" in history
    assert "system: You are a helper." in history
    assert "user: First question" in history
    assert "assistant: First answer" in history
    assert "user: Follow-up" not in history


async def test_invoke_does_not_overwrite_non_empty_chat_history_override(
    patch_mcp_tools_context, mock_ragas_event_listener, run_agent_input_with_history
) -> None:
    captured_inputs: dict[str, Any] = {}

    class CapturingCrew(CrewForTest):
        def kickoff(self, *, inputs: dict[str, Any]) -> CrewOutput:  # type: ignore[override]
            captured_inputs.update(inputs)
            return super().kickoff(inputs=inputs)

    class AgentWithOverride(AgentForTest):
        def make_kickoff_inputs(self, user_prompt_content: str) -> dict[str, Any]:
            return {"topic": user_prompt_content, "chat_history": "CUSTOM OVERRIDE"}

    out = CrewOutput(raw="agent result")
    agent = AgentWithOverride(out, api_base="https://x/", api_key="k", verbose=False)
    agent.crew = lambda: CapturingCrew(out)  # type: ignore[assignment]

    _ = [event async for event in agent.invoke(run_agent_input_with_history)]

    assert captured_inputs["chat_history"] == "CUSTOM OVERRIDE"
