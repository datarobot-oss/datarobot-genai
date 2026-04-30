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
from unittest.mock import Mock

import pytest
from ag_ui.core import AssistantMessage
from ag_ui.core import EventType
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
from llama_index.core.agent.workflow import AgentInput
from llama_index.core.agent.workflow import AgentOutput
from llama_index.core.agent.workflow import AgentStream
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow import ToolCall
from llama_index.core.agent.workflow import ToolCallResult
from llama_index.core.agent.workflow.workflow_events import AgentWorkflowStartEvent
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import MessageRole
from llama_index.core.llms.mock import MockLLM
from llama_index.core.tools import ToolOutput
from llama_index.core.tools import ToolSelection
from ragas import MultiTurnSample

from datarobot_genai.core.memory.base import BaseMemoryClient
from datarobot_genai.llama_index.agent import DataRobotLiteLLM
from datarobot_genai.llama_index.agent import LlamaIndexAgent
from datarobot_genai.llama_index.agent import datarobot_agent_class_from_llamaindex


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


class CapturingWorkflow(Workflow):
    def __init__(self, events: list[Any], state: Any) -> None:
        super().__init__(events, state)
        self.captured_user_msgs: list[str] = []

    def run(self, *, user_msg: str) -> Handler:
        self.captured_user_msgs.append(user_msg)
        return super().run(user_msg=user_msg)


class FakeMemoryClient(BaseMemoryClient):
    def __init__(self, retrieved: str = "saved memory") -> None:
        self.retrieved = retrieved
        self.retrieve_calls: list[dict[str, Any]] = []
        self.store_calls: list[dict[str, Any]] = []

    async def retrieve(
        self,
        prompt: str,
        run_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        self.retrieve_calls.append(
            {
                "prompt": prompt,
                "run_id": run_id,
                "agent_id": agent_id,
                "app_id": app_id,
                "attributes": attributes,
            }
        )
        return self.retrieved

    async def store(
        self,
        user_message: str,
        run_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self.store_calls.append(
            {
                "user_message": user_message,
                "run_id": run_id,
                "agent_id": agent_id,
                "app_id": app_id,
                "attributes": attributes,
            }
        )


class FailingMemoryClient(BaseMemoryClient):
    async def retrieve(
        self,
        prompt: str,
        run_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        raise RuntimeError("mem0 retrieve unavailable")

    async def store(
        self,
        user_message: str,
        run_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        raise RuntimeError("mem0 store unavailable")


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


# --- Tests for datarobot_agent_class_from_llamaindex ---


@pytest.mark.asyncio
async def test_datarobot_agent_class_from_llamaindex_build_workflow_returns_bound_workflow() -> (
    None
):
    workflow = Mock(name="agent_workflow")
    cls = datarobot_agent_class_from_llamaindex(workflow, [], lambda _s, _e: "")
    agent = cls()
    assert await agent.build_workflow() is workflow


def test_datarobot_agent_class_from_llamaindex_set_llm_propagates_to_agents() -> None:
    a1 = Mock()
    a1.name = "planner"
    a1.tools = []
    a2 = Mock()
    a2.name = "writer"
    a2.tools = []
    cls = datarobot_agent_class_from_llamaindex(Mock(), [a1, a2], lambda _s, _e: "")
    llm = object()
    _ = cls(llm=llm)
    assert a1.llm is llm
    assert a2.llm is llm


def test_datarobot_agent_class_from_llamaindex_allow_parallel_tool_calls_on_function_agents() -> (
    None
):
    planner = FunctionAgent(name="planner", llm=MockLLM(), tools=[])
    writer = FunctionAgent(name="writer", llm=MockLLM(), tools=[])
    cls = datarobot_agent_class_from_llamaindex(Mock(), [planner, writer], lambda _s, _e: "")
    agent = cls(allow_parallel_tool_calls=False)
    assert agent.allow_parallel_tool_calls is False
    assert planner.allow_parallel_tool_calls is False
    assert writer.allow_parallel_tool_calls is False


def test_datarobot_agent_class_from_llamaindex_set_tools_merges_original_plus_mcp() -> None:
    orig1 = Mock(name="static_tool_1")
    orig2 = Mock(name="static_tool_2")
    a1 = Mock()
    a1.name = "planner"
    a1.tools = [orig1]
    a2 = Mock()
    a2.name = "writer"
    a2.tools = [orig2]
    mcp = Mock(name="mcp_tool")
    cls = datarobot_agent_class_from_llamaindex(Mock(), [a1, a2], lambda _s, _e: "")
    agent = cls(tools=[mcp])
    assert a1.tools == [orig1, mcp]
    assert a2.tools == [orig2, mcp]
    assert agent.tools == [mcp]


def test_datarobot_agent_class_from_llamaindex_delegates_extract_response_text() -> None:
    captured: list[tuple[Any, int]] = []

    def extract(state: Any, events: list[Any]) -> str:
        captured.append((state, len(events)))
        return "extracted"

    cls = datarobot_agent_class_from_llamaindex(Mock(), [], extract)
    agent = cls()
    assert agent.extract_response_text("final_state", [1, 2, 3]) == "extracted"
    assert captured == [("final_state", 3)]


@pytest.mark.asyncio
async def test_datarobot_agent_class_from_llamaindex_invoke_streams(
    workflow: Workflow, run_agent_input: RunAgentInput
) -> None:
    cls = datarobot_agent_class_from_llamaindex(workflow, [], lambda _s, _e: "")
    agent = cls(forwarded_headers={"x-datarobot-api-key": "scoped-token-123"})
    events_out = [e async for e in agent.invoke(run_agent_input)]
    assert events_out
    assert events_out[-1][0].type == EventType.RUN_FINISHED


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

    # THEN: each agent step has its own text message (start/end pair) with a unique id
    text_starts = [e for e in ag_events if isinstance(e, TextMessageStartEvent)]
    assert len(text_starts) == 2
    content_events = [e for e in ag_events if isinstance(e, TextMessageContentEvent)]
    assert [e.delta for e in content_events] == ["Hello ", "World\n", "Hello ", "World Again\n"]

    # THEN: each TextMessageStart has a matching TextMessageEnd with the same id
    end_events = [e for e in ag_events if isinstance(e, TextMessageEndEvent)]
    assert len(end_events) == 2
    assert {e.message_id for e in text_starts} == {e.message_id for e in end_events}
    assert len({e.message_id for e in text_starts}) == 2

    # THEN: last event is RunFinishedEvent with pipeline interactions
    assert isinstance(ag_events[-1], RunFinishedEvent)
    assert isinstance(pipeline_interactions[-1], MultiTurnSample)

    # THEN: the last usage is the expected usage
    assert usage[-1] == {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}


async def test_invoke_uses_raw_user_prompt(run_agent_input_with_history) -> None:
    # GIVEN a llamaindex agent without prompt placeholders
    workflow = CapturingWorkflow(events=[], state="S")
    agent = MyLlamaAgent(workflow)

    # WHEN invoking the agent
    _ = [event async for event in agent.invoke(run_agent_input_with_history)]

    # THEN the workflow receives the raw final user prompt
    assert workflow.captured_user_msgs == ["Follow-up"]


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
    """Multi-agent stream starts a fresh text message when agent ownership changes."""
    workflow = Workflow(
        events=[
            AgentWorkflowStartEvent(),
            AgentStream(delta="one", response="", current_agent_name="Agent1"),
            AgentStream(delta=" two", response="", current_agent_name="Agent2"),
        ],
        state="S",
    )
    agent = MyLlamaAgent(workflow)

    # GIVEN a stream that switches agents between incremental text chunks
    # WHEN invoking the agent
    events_out = [e async for e in agent.invoke(run_agent_input)]
    ag_events, _, _ = zip(*events_out)

    # THEN the streamed text deltas are preserved
    content = [e for e in ag_events if isinstance(e, TextMessageContentEvent)]
    assert [e.delta for e in content] == ["one", " two"]

    # THEN each agent step gets its own text message lifecycle
    text_starts = [e for e in ag_events if isinstance(e, TextMessageStartEvent)]
    text_ends = [e for e in ag_events if isinstance(e, TextMessageEndEvent)]
    assert len(text_starts) == 2
    assert len(text_ends) == 2
    assert {e.message_id for e in text_starts} == {e.message_id for e in text_ends}
    assert len({e.message_id for e in text_starts}) == 2

    # THEN the second agent's step starts before its text message opens
    second_step_started_idx = next(
        i
        for i, event in enumerate(ag_events)
        if isinstance(event, StepStartedEvent) and event.step_name == "Agent2"
    )
    second_text_start_idx = next(i for i, event in enumerate(ag_events) if event == text_starts[1])
    second_text_content_idx = next(
        i
        for i, event in enumerate(ag_events)
        if isinstance(event, TextMessageContentEvent) and event.delta == " two"
    )
    assert second_step_started_idx < second_text_start_idx < second_text_content_idx
    assert text_starts[1].message_id == content[1].message_id

    # THEN the run still completes cleanly
    assert isinstance(ag_events[-1], RunFinishedEvent)


async def test_invoke_fallback_text_stays_within_active_step(
    run_agent_input: RunAgentInput,
) -> None:
    """Fallback text is emitted before the active step is finished."""
    workflow = Workflow(
        events=[
            AgentWorkflowStartEvent(),
            AgentInput(
                input=[ChatMessage(content="{'input': 'say hello'}", role=MessageRole.USER)],
                current_agent_name="Agent1",
            ),
        ],
        state="FINAL",
    )
    agent = MyLlamaAgent(workflow)

    # GIVEN a workflow that assigns agent ownership but streams no text
    # WHEN invoking the agent
    events_out = [e async for e in agent.invoke(run_agent_input)]
    ag_events, _, _ = zip(*events_out)

    # THEN the fallback text lifecycle stays inside the active step
    step_started_idx = next(
        i for i, event in enumerate(ag_events) if isinstance(event, StepStartedEvent)
    )
    text_start_idx = next(
        i for i, event in enumerate(ag_events) if isinstance(event, TextMessageStartEvent)
    )
    text_content_idx = next(
        i for i, event in enumerate(ag_events) if isinstance(event, TextMessageContentEvent)
    )
    text_end_idx = next(
        i for i, event in enumerate(ag_events) if isinstance(event, TextMessageEndEvent)
    )
    step_finished_idx = next(
        i for i, event in enumerate(ag_events) if isinstance(event, StepFinishedEvent)
    )

    assert step_started_idx < text_start_idx < text_content_idx < text_end_idx < step_finished_idx

    content = next(event for event in ag_events if isinstance(event, TextMessageContentEvent))
    assert content.delta == "FINAL:2"
    assert isinstance(ag_events[-1], RunFinishedEvent)


async def test_invoke_fallback_text_not_suppressed_by_prior_step_text(
    run_agent_input: RunAgentInput,
) -> None:
    """Fallback text still emits for the active step after earlier streamed text."""
    workflow = Workflow(
        events=[
            AgentWorkflowStartEvent(),
            AgentStream(delta="outline", response="", current_agent_name="Planner"),
            AgentInput(
                input=[ChatMessage(content="{'input': 'write answer'}", role=MessageRole.USER)],
                current_agent_name="Writer",
            ),
        ],
        state="FINAL",
    )
    agent = MyLlamaAgent(workflow)

    # GIVEN a prior step already streamed text
    # WHEN the active step relies on fallback extraction for its final answer
    events_out = [e async for e in agent.invoke(run_agent_input)]
    ag_events, _, _ = zip(*events_out)

    # THEN the fallback text still appears within the active step
    content = [event for event in ag_events if isinstance(event, TextMessageContentEvent)]
    assert [event.delta for event in content] == ["outline", "FINAL:3"]

    writer_step_started_idx = next(
        i
        for i, event in enumerate(ag_events)
        if isinstance(event, StepStartedEvent) and event.step_name == "Writer"
    )
    fallback_text_start_idx = next(
        i
        for i, event in enumerate(ag_events)
        if isinstance(event, TextMessageStartEvent) and event.message_id == content[1].message_id
    )
    fallback_text_end_idx = next(
        i
        for i, event in enumerate(ag_events)
        if isinstance(event, TextMessageEndEvent) and event.message_id == content[1].message_id
    )
    writer_step_finished_idx = next(
        i
        for i, event in enumerate(ag_events)
        if isinstance(event, StepFinishedEvent) and event.step_name == "Writer"
    )

    assert (
        writer_step_started_idx
        < fallback_text_start_idx
        < fallback_text_end_idx
        < writer_step_finished_idx
    )
    assert isinstance(ag_events[-1], RunFinishedEvent)


async def test_invoke_does_not_emit_fallback_after_tool_when_step_already_streamed_text(
    run_agent_input: RunAgentInput,
) -> None:
    """Fallback text does not duplicate content after a tool result in the same step."""
    workflow = Workflow(
        events=[
            AgentWorkflowStartEvent(),
            AgentInput(
                input=[ChatMessage(content="{'input': 'use tool'}", role=MessageRole.USER)],
                current_agent_name="Agent1",
            ),
            AgentStream(delta="prefix", response="", current_agent_name="Agent1"),
            AgentOutput(
                response=ChatMessage(content="prefix", role=MessageRole.ASSISTANT),
                current_agent_name="Agent1",
                tool_calls=[
                    ToolSelection(tool_name="tool1", tool_kwargs={"a": 1}, tool_id="tool1")
                ],
            ),
            ToolCall(tool_name="tool1", tool_kwargs={"a": 1}, tool_id="tool1"),
            ToolCallResult(
                tool_name="tool1",
                tool_kwargs={"a": 1},
                tool_id="tool1",
                tool_output=ToolOutput(
                    tool_name="tool1",
                    content="tool result",
                    raw_input={"a": 1},
                    raw_output="tool result",
                    is_error=False,
                ),
                return_direct=False,
            ),
        ],
        state="FINAL",
    )
    agent = MyLlamaAgent(workflow)

    # GIVEN a step that already streamed assistant text before a tool call
    # WHEN the step ends without any additional text after the tool result
    events_out = [e async for e in agent.invoke(run_agent_input)]
    ag_events, _, _ = zip(*events_out)

    # THEN fallback text is not emitted again for the same step
    content = [event for event in ag_events if isinstance(event, TextMessageContentEvent)]
    assert [event.delta for event in content] == ["prefix"]

    text_starts = [event for event in ag_events if isinstance(event, TextMessageStartEvent)]
    text_ends = [event for event in ag_events if isinstance(event, TextMessageEndEvent)]
    assert len(text_starts) == 1
    assert len(text_ends) == 1
    assert text_starts[0].message_id == text_ends[0].message_id == content[0].message_id
    assert isinstance(ag_events[-1], RunFinishedEvent)


async def test_invoke_replaces_chat_history_placeholder() -> None:
    """invoke() replaces {chat_history} inside the raw user prompt."""
    workflow = CapturingWorkflow(events=[], state="S")
    agent = MyLlamaAgent(workflow)

    run_agent_input = RunAgentInput(
        messages=[
            AgSystemMessage(id="sys_1", content="You are a helper."),
            UserMessage(id="user_1", content="First question"),
            AssistantMessage(id="asst_1", content="First answer"),
            UserMessage(id="user_2", content="History:\n{chat_history}\n\nLatest: Follow-up"),
        ],
        tools=[],
        forwarded_props=dict(model="m", authorization_context={}, forwarded_headers={}),
        thread_id="thread_id",
        run_id="run_id",
        state={},
        context=[],
    )

    _ = [event async for event in agent.invoke(run_agent_input)]

    assert workflow.captured_user_msgs
    text = workflow.captured_user_msgs[0]
    assert "system: You are a helper." in text
    assert "user: First question" in text
    assert "assistant: First answer" in text
    assert "{chat_history}" not in text


async def test_invoke_retrieves_and_stores_memory(
    run_agent_input: RunAgentInput,
) -> None:
    # GIVEN a llamaindex agent whose raw user prompt opts into memory
    workflow = CapturingWorkflow(events=[], state="S")
    memory_client = FakeMemoryClient(retrieved="Use concise answers.")
    agent = MyLlamaAgent(workflow, memory_client=memory_client)
    run_agent_input.messages = [
        UserMessage(id="message_id", content="Memory:\n{memory}\n\nLatest: Follow-up")
    ]

    # WHEN invoking the agent
    events = [event async for event in agent.invoke(run_agent_input)]

    # THEN the run completes, the prompt includes retrieved memory, and the turn is stored
    expected_user_msg = "Memory:\nUse concise answers.\n\nLatest: Follow-up"
    assert isinstance(events[-1][0], RunFinishedEvent)
    assert workflow.captured_user_msgs == [expected_user_msg]
    assert memory_client.retrieve_calls == [
        {
            "prompt": "Memory:\n{memory}\n\nLatest: Follow-up",
            "run_id": None,
            "agent_id": "MyLlamaAgent",
            "app_id": "tests.llamaindex.test_agent",
            "attributes": {"thread_id": "thread_id"},
        }
    ]
    assert memory_client.store_calls == [
        {
            "user_message": "Memory:\n{memory}\n\nLatest: Follow-up",
            "run_id": "run_id",
            "agent_id": "MyLlamaAgent",
            "app_id": "tests.llamaindex.test_agent",
            "attributes": {"thread_id": "thread_id"},
        }
    ]


async def test_invoke_skips_memory_when_placeholder_is_absent(
    run_agent_input: RunAgentInput,
) -> None:
    # GIVEN a llamaindex agent whose raw user prompt does not opt into memory
    workflow = Workflow(events=[], state="S")
    memory_client = FakeMemoryClient(retrieved="Use concise answers.")
    agent = MyLlamaAgent(workflow, memory_client=memory_client)

    # WHEN invoking the agent
    _ = [event async for event in agent.invoke(run_agent_input)]

    # THEN memory retrieval and storage are both skipped
    assert memory_client.retrieve_calls == []
    assert memory_client.store_calls == []


async def test_invoke_gracefully_degrades_when_memory_fails(
    run_agent_input: RunAgentInput,
) -> None:
    # GIVEN a llamaindex agent whose raw user prompt opts into memory
    workflow = CapturingWorkflow(events=[], state="S")
    agent = MyLlamaAgent(workflow, memory_client=FailingMemoryClient())
    run_agent_input.messages = [
        UserMessage(id="message_id", content="Memory:\n{memory}\n\nLatest: Follow-up")
    ]

    # WHEN invoking the agent
    events = [event async for event in agent.invoke(run_agent_input)]

    # THEN the run still completes and the unresolved placeholder is removed
    assert isinstance(events[-1][0], RunFinishedEvent)
    assert workflow.captured_user_msgs == ["Memory:\n\n\nLatest: Follow-up"]
    assert "{memory}" not in workflow.captured_user_msgs[0]


async def test_invoke_does_not_store_memory_when_workflow_fails(
    run_agent_input: RunAgentInput,
) -> None:
    class FailingWorkflow(CapturingWorkflow):
        def run(self, *, user_msg: str) -> Handler:
            self.captured_user_msgs.append(user_msg)

            class FailingHandler:
                async def stream_events(self) -> AsyncGenerator[Any, None]:
                    raise RuntimeError("workflow failed")
                    yield  # pragma: no cover

            return FailingHandler()

    # GIVEN a llamaindex agent whose workflow fails after memory retrieval
    workflow = FailingWorkflow(events=[], state="S")
    memory_client = FakeMemoryClient(retrieved="Use concise answers.")
    agent = MyLlamaAgent(workflow, memory_client=memory_client)
    run_agent_input.messages = [
        UserMessage(id="message_id", content="Memory:\n{memory}\n\nLatest: Follow-up")
    ]

    # WHEN invoking the agent and the workflow fails
    with pytest.raises(RuntimeError, match="workflow failed"):
        _ = [event async for event in agent.invoke(run_agent_input)]

    # THEN retrieval happens, but storage is skipped because the run never finishes
    expected_user_msg = "Memory:\nUse concise answers.\n\nLatest: Follow-up"
    assert workflow.captured_user_msgs == [expected_user_msg]
    assert memory_client.retrieve_calls == [
        {
            "prompt": "Memory:\n{memory}\n\nLatest: Follow-up",
            "run_id": None,
            "agent_id": "MyLlamaAgent",
            "app_id": "tests.llamaindex.test_agent",
            "attributes": {"thread_id": "thread_id"},
        }
    ]
    assert memory_client.store_calls == []
