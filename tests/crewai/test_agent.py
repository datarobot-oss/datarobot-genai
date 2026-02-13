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
from typing import Any
from collections.abc import AsyncGenerator
from unittest.mock import patch

from ag_ui.core import RunAgentInput, UserMessage
from crewai import CrewOutput
import pytest
from ragas import MultiTurnSample
from ragas.messages import AIMessage
from ragas.messages import HumanMessage

from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.crewai.agent import CrewAIAgent


# --- Test helpers ---


class TestCrew:
    def __init__(self, output: CrewOutput | None = None, *args: Any, **kwargs: Any):
        self.args = args
        self.kwargs = kwargs
        self.output = output

    def kickoff(self, *, inputs: dict[str, Any]) -> CrewOutput:  # type: ignore[name-defined]
        return self.output or CrewOutput(raw="final-output")


class TestListener:
    def __init__(self, messages: list[Any] | None = None):
        self.messages = messages or []
        self.called_setup = False

    def setup_listeners(self, crewai_event_bus: Any) -> None:
        self.called_setup = True


@pytest.fixture
def mock_ragas_event_listener() -> TestListener:
    event_listener = TestListener(messages=[HumanMessage(content="hi"), AIMessage(content="there")])
    with patch(
        "datarobot_genai.crewai.agent.CrewAIRagasEventListener"
    ) as mock_ragas_event_listener:
        mock_ragas_event_listener.return_value = event_listener
        yield event_listener


class TestAgent(CrewAIAgent):
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
        return TestCrew(self.crew_output)


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
    agent = TestAgent(
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

    # THEN there is just one event
    assert len(events) == 1

    # THEN the event is a tuple of (delta, pipeline interactions, UsageMetrics)
    delta, pipeline_interactions, usage = events[0]

    # THEN delta is the expected delta
    assert delta == "agent result"

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
