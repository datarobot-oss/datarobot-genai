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
from unittest.mock import patch

import pytest
from ag_ui.core import AssistantMessage
from ag_ui.core import RunAgentInput
from ag_ui.core import RunFinishedEvent
from ag_ui.core import RunStartedEvent
from ag_ui.core import StepFinishedEvent
from ag_ui.core import StepStartedEvent
from ag_ui.core import SystemMessage as AgSystemMessage
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from ag_ui.core import UserMessage
from ag_ui.core.types import FunctionCall as AgFunctionCall
from ag_ui.core.types import ToolCall as AgToolCall
from ag_ui.core.types import ToolMessage as AgToolMessage

import datarobot_genai.llama_index.agent as agent_mod
from llama_index.core.agent.workflow import AgentInput
from llama_index.core.agent.workflow import AgentOutput
from llama_index.core.agent.workflow import AgentStream
from llama_index.core.agent.workflow import ToolCall
from llama_index.core.agent.workflow import ToolCallResult
from llama_index.core.agent.workflow.workflow_events import AgentWorkflowStartEvent
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import MessageRole
from llama_index.core.tools import ToolOutput
from llama_index.core.tools import ToolSelection
from ragas import MultiTurnSample

from datarobot_genai.llama_index.agent import DataRobotLiteLLM
from datarobot_genai.llama_index.agent import LlamaIndexAgent


class Handler:
    def __init__(self, events: list[Any], state: Any) -> None:
        self._events = events
        self._state = state

    async def stream_events(self) -> AsyncGenerator[Any, None]:
        for e in self._events:
            yield e

    @property
    def ctx(self) -> Any:
        class Ctx:
            async def get(self, key: str) -> Any:  # noqa: A003 - method name from API
                assert key == "state"
                return state

        state = self._state
        return Ctx()


class Workflow:
    def __init__(self, events: list[Any], state: Any) -> None:
        self._events = events
        self._state = state

    def run(self, *, user_msg: str) -> Handler:  # noqa: ARG002
        return Handler(self._events, self._state)


class MyLlamaAgent(LlamaIndexAgent):
    def __init__(self, workflow: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._wf = workflow

    def build_workflow(self) -> Any:
        return self._wf

    def extract_response_text(self, result_state: Any, events: list[Any]) -> str:
        return f"{result_state}:{len(events)}"


# --- Tests for DataRobotLiteLLM ---


def test_datarobot_litellm_metadata_properties() -> None:
    llm = DataRobotLiteLLM(model="dr/model", max_tokens=256)
    meta = llm.metadata

    assert meta.context_window == 128000
    assert meta.num_output == 256
    assert meta.is_chat_model is True
    assert meta.is_function_calling_model is True
    assert meta.model_name == "dr/model"


# --- Tests for create_pipeline_interactions_from_events ---


def test_create_pipeline_interactions_from_events_none() -> None:
    assert LlamaIndexAgent.create_pipeline_interactions_from_events(None) is None


# --- Tests for LlamaIndexAgent ---


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


@pytest.fixture
def events() -> list[Any]:
    return [
        AgentWorkflowStartEvent(),
        AgentInput(
            input=[ChatMessage(content="{'input': 'say hello world'}", role=MessageRole.USER)],
            current_agent_name="Agent 1",
        ),
        AgentStream(delta="Hello ", response="", current_agent_name="Agent 1"),
        AgentStream(delta="World\n", response="", current_agent_name="Agent 1"),
        AgentOutput(
            response=ChatMessage(content="Hello World", role=MessageRole.ASSISTANT),
            current_agent_name="Agent 1",
        ),
        AgentInput(
            input=[
                ChatMessage(content="{'input': 'say hello world'}", role=MessageRole.USER),
                ChatMessage(content="Hello World", role=MessageRole.ASSISTANT),
            ],
            current_agent_name="Agent 2",
        ),
        AgentOutput(
            response=ChatMessage(content="", role=MessageRole.ASSISTANT),
            current_agent_name="Agent 2",
            tool_calls=[ToolSelection(tool_name="tool1", tool_kwargs={"a": 1}, tool_id="tool1")],
        ),
        ToolCall(tool_name="tool1", tool_kwargs={"a": 1}, tool_id="tool1"),
        ToolCallResult(
            tool_name="tool1",
            tool_kwargs={"a": 1},
            tool_id="tool1",
            tool_output=ToolOutput(
                tool_name="tool1",
                content="Hello World",
                raw_input={"a": 1},
                raw_output="Hello World",
                is_error=False,
            ),
            return_direct=False,
        ),
        AgentStream(delta="Hello ", response="", current_agent_name="Agent 2"),
        AgentStream(delta="World Again\n", response="", current_agent_name="Agent 2"),
        AgentOutput(
            response=ChatMessage(content="Hello World Again", role=MessageRole.ASSISTANT),
            current_agent_name="Agent 2",
        ),
    ]


@pytest.fixture
def workflow(events: list[Any]) -> Workflow:
    return Workflow(events=events, state="S")


@pytest.fixture
def agent(workflow: Workflow) -> MyLlamaAgent:
    forwarded_headers = {
        "x-datarobot-api-key": "scoped-token-123",
    }
    return MyLlamaAgent(workflow, forwarded_headers=forwarded_headers)


async def test_llama_index_agent_invoke(
    agent: MyLlamaAgent, run_agent_input: RunAgentInput
) -> None:
    # GIVEN: fake agent with fake workflow (two agent steps + tool call in fixture)
    # WHEN: invoke the agent with a run agent input
    resp = agent.invoke(run_agent_input)

    # THEN: the response is an async generator
    assert isinstance(resp, AsyncGenerator)
    events = [event async for event in resp]

    # THEN: the events follow AG-UI lifecycle pattern
    ag_events, pipeline_interactions, usage = zip(*events)

    # THEN: first event is RunStartedEvent
    assert isinstance(ag_events[0], RunStartedEvent)

    # THEN: text message events contain the expected deltas (after step / lifecycle events)
    text_starts = [e for e in ag_events if isinstance(e, TextMessageStartEvent)]
    assert len(text_starts) == 1
    content_events = [e for e in ag_events if isinstance(e, TextMessageContentEvent)]
    assert [e.delta for e in content_events] == ["Hello ", "World\n", "Hello ", "World Again\n"]

    # THEN: TextMessageEnd is present
    end_events = [e for e in ag_events if isinstance(e, TextMessageEndEvent)]
    assert len(end_events) == 1

    # THEN: last event is RunFinishedEvent with pipeline interactions
    assert isinstance(ag_events[-1], RunFinishedEvent)
    assert isinstance(pipeline_interactions[-1], MultiTurnSample)

    # THEN: the last usage is the expected usage
    assert usage[-1] == {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}


def test_make_input_message_includes_history_summary(run_agent_input_with_history) -> None:
    """By default, LlamaIndexAgent.make_input_message should NOT include history."""
    workflow = Workflow(events=[], state="S")
    agent = MyLlamaAgent(workflow)

    text = agent.make_input_message(run_agent_input_with_history)

    assert text == "Follow-up"


def test_make_input_message_zero_history_disables_summary(
    run_agent_input_with_history,
) -> None:
    """When max_history_messages is 0, history should be disabled."""
    workflow = Workflow(events=[], state="S")
    agent = MyLlamaAgent(workflow, max_history_messages=0)

    text = agent.make_input_message(run_agent_input_with_history)
    assert text == "Follow-up"


async def test_invoke_agent_output_with_dict_tool_calls(
    run_agent_input: RunAgentInput,
) -> None:
    """AgentOutput with tool_calls as list of dicts (tool_name from .get()) is handled."""

    class AgentOutputLike:
        response = None
        current_agent_name = "A"
        tool_calls = [{"tool_name": "dict_tool", "tool_kwargs": {"x": 1}, "tool_id": "id1"}]

    AgentOutputLike.__name__ = "AgentOutput"

    workflow = Workflow(
        events=[
            AgentWorkflowStartEvent(),
            AgentOutputLike(),
            AgentStream(delta="hey", response="", current_agent_name="A"),
        ],
        state="S",
    )
    agent = MyLlamaAgent(workflow)

    events_out = [e async for e in agent.invoke(run_agent_input)]
    ag_events, pipeline, _ = zip(*events_out)

    assert isinstance(ag_events[-1], RunFinishedEvent)
    content = [e for e in ag_events if isinstance(e, TextMessageContentEvent)]
    assert [e.delta for e in content] == ["hey"]
    assert pipeline[-1] is not None


async def test_invoke_agent_stream_multiple_agents(
    run_agent_input: RunAgentInput,
) -> None:
    """Multi-agent stream yields text, step events, and completes."""
    workflow = Workflow(
        events=[
            AgentWorkflowStartEvent(),
            AgentStream(delta="one", response="", current_agent_name="Agent1"),
            AgentStream(delta=" two", response="", current_agent_name="Agent2"),
        ],
        state="S",
    )
    agent = MyLlamaAgent(workflow)

    events_out = [e async for e in agent.invoke(run_agent_input)]
    ag_events, _, _ = zip(*events_out)

    content = [e for e in ag_events if isinstance(e, TextMessageContentEvent)]
    assert [e.delta for e in content] == ["one", " two"]
    assert any(isinstance(e, TextMessageStartEvent) for e in ag_events)
    assert any(isinstance(e, TextMessageEndEvent) for e in ag_events)

    step_started = [e for e in ag_events if isinstance(e, StepStartedEvent)]
    step_finished = [e for e in ag_events if isinstance(e, StepFinishedEvent)]
    if step_started:
        assert {e.step_name for e in step_started} <= {"Agent1", "Agent2"}
    if step_finished:
        assert {e.step_name for e in step_finished} <= {"Agent1", "Agent2"}

    assert isinstance(ag_events[-1], RunFinishedEvent)


async def test_invoke_passes_chat_history_for_multi_turn() -> None:
    """invoke() passes structured chat_history kwarg to workflow.run() for multi-turn."""
    captured_kwargs: dict[str, Any] = {}

    class CapturingWorkflow(Workflow):
        def run(self, **kwargs: Any) -> Handler:  # type: ignore[override]
            captured_kwargs.update(kwargs)
            return Handler([], "S")

    workflow = CapturingWorkflow(events=[], state="S")
    agent = MyLlamaAgent(workflow)

    run_agent_input = RunAgentInput(
        messages=[
            AgSystemMessage(id="sys_1", content="You are a helper."),
            UserMessage(id="user_1", content="First question"),
            AssistantMessage(id="asst_1", content="First answer"),
            UserMessage(id="user_2", content="Follow-up"),
        ],
        tools=[],
        forwarded_props=dict(model="m", authorization_context={}, forwarded_headers={}),
        thread_id="thread_id",
        run_id="run_id",
        state={},
        context=[],
    )

    _ = [event async for event in agent.invoke(run_agent_input)]

    assert "chat_history" in captured_kwargs
    chat_history = captured_kwargs["chat_history"]
    assert len(chat_history) > 0
    assert hasattr(chat_history[0], "role")


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
            AssistantMessage(id="asst_2", content="Here are cats."),
            UserMessage(id="user_2", content="tell me more"),
        ],
        tools=[],
        forwarded_props=dict(model="m", authorization_context={}, forwarded_headers={}),
        thread_id="thread_id",
        run_id="run_id",
        state={},
        context=[],
    )


async def test_invoke_multi_turn_passes_chat_history_param(run_agent_input_multi_turn) -> None:
    """Multi-turn conversations pass chat_history to workflow.run()."""
    workflow_run_kwargs: dict[str, Any] = {}

    class CapturingWorkflow:
        def run(self, **kwargs: Any) -> "Handler":
            workflow_run_kwargs.update(kwargs)
            return Handler([], None)

    class MultiTurnAgent(LlamaIndexAgent):
        async def build_workflow(self) -> Any:
            return CapturingWorkflow()

        def extract_response_text(self, result_state: Any, events: list[Any]) -> str:
            return ""

    agent = MultiTurnAgent()

    _ = [event async for event in agent.invoke(run_agent_input_multi_turn)]

    assert "chat_history" in workflow_run_kwargs
    chat_history = workflow_run_kwargs["chat_history"]
    assert len(chat_history) > 0
    assert hasattr(chat_history[0], "role")
