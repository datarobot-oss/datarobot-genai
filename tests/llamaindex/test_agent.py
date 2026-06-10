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

import uuid
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from ag_ui.core import AssistantMessage
from ag_ui.core import EventType
from ag_ui.core import ReasoningMessageChunkEvent
from ag_ui.core import RunAgentInput
from ag_ui.core import RunFinishedEvent
from ag_ui.core import RunStartedEvent
from ag_ui.core import StepFinishedEvent
from ag_ui.core import StepStartedEvent
from ag_ui.core import SystemMessage as AgSystemMessage
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from ag_ui.core import ToolCallEndEvent
from ag_ui.core import ToolCallResultEvent
from ag_ui.core import ToolCallStartEvent
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

from datarobot_genai.core.agents.verify import validate_sequence
from datarobot_genai.core.memory.base import BaseMemoryClient
from datarobot_genai.llama_index.agent import DataRobotLiteLLM
from datarobot_genai.llama_index.agent import LlamaIndexAgent
from datarobot_genai.llama_index.agent import _thinking_delta_from_raw
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


# --- Tests for reasoning_content extraction ---


def test_thinking_delta_from_raw_object_form() -> None:
    """Reasoning is read from a LiteLLM streaming chunk object (delta.reasoning_content)."""
    raw = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(reasoning_content="step "))]
    )
    assert _thinking_delta_from_raw(raw) == "step "


def test_thinking_delta_from_raw_dict_form() -> None:
    """Reasoning is read from a model_dump'd chunk dict as well."""
    raw = {"choices": [{"delta": {"reasoning_content": "step "}}]}
    assert _thinking_delta_from_raw(raw) == "step "


def test_thinking_delta_from_raw_absent_or_empty_is_none() -> None:
    assert _thinking_delta_from_raw(None) is None
    assert _thinking_delta_from_raw(SimpleNamespace(choices=[])) is None
    assert _thinking_delta_from_raw({"choices": [{"delta": {"reasoning_content": ""}}]}) is None
    # A content-only delta (no reasoning_content attribute) yields None, not an error.
    raw = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"))])
    assert _thinking_delta_from_raw(raw) is None


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

    # THEN: the sequence is valid per the AG-UI verifier
    validate_sequence(ag_events)

    # THEN: first event is RunStartedEvent
    assert isinstance(ag_events[0], RunStartedEvent)

    # THEN: each agent step gets its own text message (Agent 1 streamed text,
    # then Agent 2 streamed text after the tool call) with distinct message ids.
    text_starts = [e for e in ag_events if isinstance(e, TextMessageStartEvent)]
    text_ends = [e for e in ag_events if isinstance(e, TextMessageEndEvent)]
    assert len(text_starts) == 2
    assert len(text_ends) == 2
    assert text_starts[0].message_id != text_starts[1].message_id
    assert text_starts[0].message_id == text_ends[0].message_id
    assert text_starts[1].message_id == text_ends[1].message_id

    content_events = [e for e in ag_events if isinstance(e, TextMessageContentEvent)]
    assert [e.delta for e in content_events] == ["Hello ", "World\n", "Hello ", "World Again\n"]

    # THEN: each STEP_STARTED has a matching STEP_FINISHED for the same agent
    step_started = [e for e in ag_events if isinstance(e, StepStartedEvent)]
    step_finished = [e for e in ag_events if isinstance(e, StepFinishedEvent)]
    assert [e.step_name for e in step_started] == ["Agent 1", "Agent 2"]
    assert [e.step_name for e in step_finished] == ["Agent 1", "Agent 2"]

    # THEN: text messages live inside their step boundaries (END before STEP_FINISHED)
    indexed = list(enumerate(ag_events))
    end_idx = [i for i, e in indexed if isinstance(e, TextMessageEndEvent)]
    finish_idx = [i for i, e in indexed if isinstance(e, StepFinishedEvent)]
    assert end_idx[0] < finish_idx[0]
    assert end_idx[1] < finish_idx[1]

    # THEN: the tool call sits between the two agents' text messages
    tool_start_idx = next(i for i, e in indexed if isinstance(e, ToolCallStartEvent))
    tool_end_idx = next(i for i, e in indexed if isinstance(e, ToolCallEndEvent))
    text_start_idx_1 = next(
        i
        for i, e in indexed
        if isinstance(e, TextMessageStartEvent) and e.message_id == text_starts[1].message_id
    )
    assert end_idx[0] < tool_start_idx < tool_end_idx < text_start_idx_1

    # THEN: the tool call result has its own unique message_id (the tool_call_id),
    # independent of any text bubble.
    tool_results = [e for e in ag_events if isinstance(e, ToolCallResultEvent)]
    assert len(tool_results) == 1
    assert tool_results[0].message_id == tool_results[0].tool_call_id
    assert tool_results[0].message_id != text_starts[0].message_id
    assert tool_results[0].message_id != text_starts[1].message_id

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

    validate_sequence(ag_events)
    assert isinstance(ag_events[-1], RunFinishedEvent)
    content = [e for e in ag_events if isinstance(e, TextMessageContentEvent)]
    assert [e.delta for e in content] == ["hey"]
    assert pipeline[-1] is not None


async def test_invoke_tool_result_message_id_is_distinct_from_text_bubbles(
    run_agent_input: RunAgentInput,
) -> None:
    """Text -> tool -> text within one step: tool_result.message_id is the
    tool_call_id and does not collide with either surrounding text bubble.
    """
    workflow = Workflow(
        events=[
            AgentWorkflowStartEvent(),
            AgentInput(
                input=[ChatMessage(content="go", role=MessageRole.USER)],
                current_agent_name="A",
            ),
            AgentStream(delta="thinking ", response="", current_agent_name="A"),
            AgentStream(delta="aloud", response="", current_agent_name="A"),
            AgentOutput(
                response=ChatMessage(content="thinking aloud", role=MessageRole.ASSISTANT),
                current_agent_name="A",
                tool_calls=[ToolSelection(tool_name="t", tool_kwargs={}, tool_id="t1")],
            ),
            ToolCall(tool_name="t", tool_kwargs={}, tool_id="t1"),
            ToolCallResult(
                tool_name="t",
                tool_kwargs={},
                tool_id="t1",
                tool_output=ToolOutput(
                    tool_name="t",
                    content="ok",
                    raw_input={},
                    raw_output="ok",
                    is_error=False,
                ),
                return_direct=False,
            ),
            AgentStream(delta="after tool", response="", current_agent_name="A"),
        ],
        state="S",
    )
    agent = MyLlamaAgent(workflow)
    events_out = [e async for e in agent.invoke(run_agent_input)]
    ag_events, _, _ = zip(*events_out)

    validate_sequence(ag_events)
    text_starts = [e for e in ag_events if isinstance(e, TextMessageStartEvent)]
    tool_results = [e for e in ag_events if isinstance(e, ToolCallResultEvent)]
    assert len(text_starts) == 2
    assert len(tool_results) == 1
    assert tool_results[0].message_id == tool_results[0].tool_call_id == "t1"
    assert tool_results[0].message_id != text_starts[0].message_id
    assert tool_results[0].message_id != text_starts[1].message_id
    # The two text bubbles within the step still have distinct ids.
    assert text_starts[0].message_id != text_starts[1].message_id


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

    validate_sequence(ag_events)
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


# --- Tests for reasoning models ---


async def test_invoke_emits_reasoning_chunks_from_thinking_delta(
    run_agent_input: RunAgentInput,
) -> None:
    """AgentStream.thinking_delta surfaces as self-contained AG-UI reasoning chunks.

    Reasoning models (Qwen extended thinking, Claude, etc.) stream incremental
    reasoning on ``AgentStream.thinking_delta``. Each delta becomes one
    ``REASONING_MESSAGE_CHUNK`` event with no lifecycle wrapping, mirroring the
    langgraph adapter.
    """
    workflow = Workflow(
        events=[
            AgentWorkflowStartEvent(),
            AgentStream(delta="", thinking_delta="step 1 ", response="", current_agent_name="A"),
            AgentStream(delta="", thinking_delta="step 2", response="", current_agent_name="A"),
        ],
        state="S",
    )
    agent = MyLlamaAgent(workflow)

    events_out = [e async for e in agent.invoke(run_agent_input)]
    ag_events, _, _ = zip(*events_out)

    validate_sequence(ag_events)
    reasoning = [e for e in ag_events if isinstance(e, ReasoningMessageChunkEvent)]
    assert [e.delta for e in reasoning] == ["step 1 ", "step 2"]
    # Reasoning is routed to REASONING events, never leaked into text content.
    text_deltas = [e.delta for e in ag_events if isinstance(e, TextMessageContentEvent)]
    assert "step 1 " not in text_deltas
    assert "step 2" not in text_deltas


async def test_invoke_emits_reasoning_before_text(
    run_agent_input: RunAgentInput,
) -> None:
    """Reasoning is emitted before the text lifecycle opens for the same step."""
    workflow = Workflow(
        events=[
            AgentWorkflowStartEvent(),
            AgentStream(
                delta="", thinking_delta="thinking...", response="", current_agent_name="A"
            ),
            AgentStream(delta="answer", thinking_delta=None, response="", current_agent_name="A"),
        ],
        state="S",
    )
    agent = MyLlamaAgent(workflow)

    events_out = [e async for e in agent.invoke(run_agent_input)]
    ag_events, _, _ = zip(*events_out)

    validate_sequence(ag_events)
    types = [e.type for e in ag_events]
    assert types.index(EventType.REASONING_MESSAGE_CHUNK) < types.index(
        EventType.TEXT_MESSAGE_START
    )
    text = [e for e in ag_events if isinstance(e, TextMessageContentEvent)]
    assert [e.delta for e in text] == ["answer"]

    # Reasoning must not share the assistant text's message_id, else a frontend
    # grouping by id folds the reasoning into the text bubble.
    reasoning_id = next(
        e.message_id for e in ag_events if isinstance(e, ReasoningMessageChunkEvent)
    )
    text_start_id = next(e.message_id for e in ag_events if isinstance(e, TextMessageStartEvent))
    assert reasoning_id != text_start_id
    uuid.UUID(reasoning_id)  # reasoning id is its own valid UUID


async def test_invoke_emits_reasoning_from_raw_chunk(
    run_agent_input: RunAgentInput,
) -> None:
    """When the LLM doesn't populate thinking_delta, reasoning is recovered from the
    raw LiteLLM chunk on ``AgentStream.raw`` (the stock litellm wrapper drops it from
    ``thinking_delta`` but still exposes it on the raw chunk).
    """
    raw = {"choices": [{"delta": {"reasoning_content": "hmm"}}]}
    workflow = Workflow(
        events=[
            AgentWorkflowStartEvent(),
            AgentStream(
                delta="", thinking_delta=None, response="", current_agent_name="A", raw=raw
            ),
            AgentStream(delta="answer", thinking_delta=None, response="", current_agent_name="A"),
        ],
        state="S",
    )
    agent = MyLlamaAgent(workflow)

    events_out = [e async for e in agent.invoke(run_agent_input)]
    ag_events, _, _ = zip(*events_out)

    validate_sequence(ag_events)
    reasoning = [e for e in ag_events if isinstance(e, ReasoningMessageChunkEvent)]
    assert [e.delta for e in reasoning] == ["hmm"]
    text = [e for e in ag_events if isinstance(e, TextMessageContentEvent)]
    assert [e.delta for e in text] == ["answer"]


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
