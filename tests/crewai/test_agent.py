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
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

from crewai.types.streaming import CrewStreamingOutput
from crewai.types.streaming import StreamChunk
from crewai.types.streaming import StreamChunkType
import pytest
from ag_ui.core import ReasoningEndEvent
from ag_ui.core import ReasoningMessageEndEvent
from ag_ui.core import ReasoningMessageStartEvent
from ag_ui.core import RunAgentInput
from ag_ui.core import RunFinishedEvent
from ag_ui.core import RunStartedEvent
from ag_ui.core import StepFinishedEvent
from ag_ui.core import StepStartedEvent
from ag_ui.core import TextMessageChunkEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from ag_ui.core import UserMessage
from crewai import CrewOutput
from ragas import MultiTurnSample
from ragas.messages import AIMessage
from ragas.messages import HumanMessage

from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.core.agents.verify import validate_sequence
from datarobot_genai.core.memory.base import BaseMemoryClient
from datarobot_genai.crewai.agent import CrewAIAgent
from datarobot_genai.crewai.agent import datarobot_agent_class_from_crew


# --- Test helpers ---


def _mock_crewai_agent() -> MagicMock:
    agent = MagicMock()
    agent.llm = None
    agent.function_calling_llm = None
    agent.tools = []
    agent.verbose = True
    return agent


class CrewForTest:
    def __init__(self, output: CrewOutput | None = None, *args: Any, **kwargs: Any):
        self.args = args
        self.kwargs = kwargs
        self.output = output
        self.verbose = True

    async def kickoff_async(self, *, inputs: dict[str, Any]) -> CrewOutput | CrewStreamingOutput:  # type: ignore[name-defined]
        return self.output or CrewOutput(raw="final-output")


class ListenerForTest:
    def __init__(self, messages: list[Any] | None = None):
        self.messages = messages or []
        self.called_setup = False

    def setup_listeners(self, crewai_event_bus: Any) -> None:
        self.called_setup = True


class FakeStreamingOutput(CrewStreamingOutput):
    def __init__(
        self,
        *,
        async_iterator: AsyncGenerator[StreamChunk, None],
        token_usage: Any | None = None,
    ) -> None:
        super().__init__(async_iterator=async_iterator)
        self._forced_result = SimpleNamespace(
            token_usage=token_usage
            or SimpleNamespace(completion_tokens=1, prompt_tokens=2, total_tokens=3)
        )

    @property
    def result(self) -> Any:
        return self._forced_result


class StreamingCrewForTest:
    def __init__(self, output: CrewStreamingOutput) -> None:
        self.output = output
        self.verbose = True

    async def kickoff_async(self, *, inputs: dict[str, Any]) -> CrewStreamingOutput:  # type: ignore[override]
        return self.output


class StreamingListenerForTest:
    def __init__(self) -> None:
        self.reasoning_event = False

    def setup_listeners(self, crewai_event_bus: Any) -> None:
        return None


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
    def __init__(self, crew_output: CrewOutput | None = None, *args: Any, **kwargs: Any) -> None:
        crew_kw = kwargs.pop("crew", None)
        self.crew_output = crew_output
        self._crew_for_test: Any = crew_kw if crew_kw is not None else CrewForTest(crew_output)
        self._agents_for_test: list[MagicMock] = [_mock_crewai_agent(), _mock_crewai_agent()]
        self._tasks_for_test: list[MagicMock] = [MagicMock(), MagicMock()]
        super().__init__(*args, **kwargs)

    @property
    def agents(self) -> list[Any]:
        return self._agents_for_test

    @property
    def tasks(self) -> list[Any]:
        return self._tasks_for_test

    def make_kickoff_inputs(self, user_prompt_content: str) -> dict[str, Any]:
        return {"topic": user_prompt_content}

    @property
    def crew(self) -> Any:
        return self._crew_for_test


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


@pytest.fixture
def run_agent_input_with_structured_prompt() -> RunAgentInput:
    return RunAgentInput(
        messages=[UserMessage(content='{"topic": "AI"}', id="message_id")],
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


async def test_invoke_streaming_starts_fresh_text_message_per_step(
    run_agent_input, mock_ragas_event_listener
) -> None:
    async def stream_chunks() -> AsyncGenerator[StreamChunk, None]:
        yield StreamChunk(
            content="plan",
            chunk_type=StreamChunkType.TEXT,
            task_name="plan",
            agent_role="Planner",
        )
        yield StreamChunk(
            content="write",
            chunk_type=StreamChunkType.TEXT,
            task_name="write",
            agent_role="Writer",
        )

    # GIVEN a streamed CrewAI run that switches agent roles between text chunks
    streaming_output = FakeStreamingOutput(async_iterator=stream_chunks())
    agent = AgentForTest(
        api_base="https://x/",
        api_key="k",
        verbose=False,
        crew=StreamingCrewForTest(streaming_output),
    )
    streaming_listener = StreamingListenerForTest()

    with patch(
        "datarobot_genai.crewai.agent.CrewAIStreamingEventListener",
        return_value=streaming_listener,
    ):
        # WHEN invoking the agent
        events_out = [event async for event in agent.invoke(run_agent_input)]

    ag_events, _, _ = zip(*events_out)

    # THEN the AG-UI event sequence remains valid
    validate_sequence(list(ag_events))

    # THEN each step gets its own text message lifecycle with a distinct id
    text_starts = [event for event in ag_events if isinstance(event, TextMessageStartEvent)]
    text_contents = [event for event in ag_events if isinstance(event, TextMessageContentEvent)]
    text_ends = [event for event in ag_events if isinstance(event, TextMessageEndEvent)]
    assert [event.delta for event in text_contents] == ["plan", "write"]
    assert len(text_starts) == 2
    assert len(text_ends) == 2
    assert len({event.message_id for event in text_starts}) == 2
    assert {event.message_id for event in text_starts} == {event.message_id for event in text_ends}

    # THEN the first text message closes before Planner finishes
    planner_text_end_idx = next(
        i for i, event in enumerate(ag_events) if isinstance(event, TextMessageEndEvent)
    )
    planner_step_finished_idx = next(
        i
        for i, event in enumerate(ag_events)
        if isinstance(event, StepFinishedEvent) and event.step_name == "Planner"
    )
    writer_step_started_idx = next(
        i
        for i, event in enumerate(ag_events)
        if isinstance(event, StepStartedEvent) and event.step_name == "Writer"
    )
    writer_text_start_idx = next(
        i
        for i, event in enumerate(ag_events)
        if isinstance(event, TextMessageStartEvent)
        and event.message_id == text_contents[1].message_id
    )
    assert planner_text_end_idx < planner_step_finished_idx < writer_step_started_idx
    assert writer_step_started_idx < writer_text_start_idx


async def test_invoke_streaming_closes_reasoning_before_finishing_step(
    run_agent_input, mock_ragas_event_listener
) -> None:
    streaming_listener = StreamingListenerForTest()

    async def stream_chunks() -> AsyncGenerator[StreamChunk, None]:
        streaming_listener.reasoning_event = True
        yield StreamChunk(
            content="think",
            chunk_type=StreamChunkType.TEXT,
            task_name="plan",
            agent_role="Planner",
        )
        streaming_listener.reasoning_event = False
        yield StreamChunk(
            content="answer",
            chunk_type=StreamChunkType.TEXT,
            task_name="write",
            agent_role="Writer",
        )

    # GIVEN a streamed CrewAI run that exits reasoning mode while switching steps
    streaming_output = FakeStreamingOutput(async_iterator=stream_chunks())
    agent = AgentForTest(
        api_base="https://x/",
        api_key="k",
        verbose=False,
        crew=StreamingCrewForTest(streaming_output),
    )

    with patch(
        "datarobot_genai.crewai.agent.CrewAIStreamingEventListener",
        return_value=streaming_listener,
    ):
        # WHEN invoking the agent
        events_out = [event async for event in agent.invoke(run_agent_input)]

    ag_events, _, _ = zip(*events_out)

    # THEN the AG-UI event sequence remains valid
    validate_sequence(list(ag_events))

    # THEN the planner reasoning lifecycle closes before Planner finishes
    reasoning_start = next(
        event for event in ag_events if isinstance(event, ReasoningMessageStartEvent)
    )
    writer_text_start = next(
        event for event in ag_events if isinstance(event, TextMessageStartEvent)
    )
    assert reasoning_start.message_id != writer_text_start.message_id

    reasoning_message_end_idx = next(
        i for i, event in enumerate(ag_events) if isinstance(event, ReasoningMessageEndEvent)
    )
    reasoning_end_idx = next(
        i for i, event in enumerate(ag_events) if isinstance(event, ReasoningEndEvent)
    )
    planner_step_finished_idx = next(
        i
        for i, event in enumerate(ag_events)
        if isinstance(event, StepFinishedEvent) and event.step_name == "Planner"
    )
    writer_step_started_idx = next(
        i
        for i, event in enumerate(ag_events)
        if isinstance(event, StepStartedEvent) and event.step_name == "Writer"
    )
    assert reasoning_message_end_idx < reasoning_end_idx < planner_step_finished_idx
    assert planner_step_finished_idx < writer_step_started_idx


async def test_invoke_does_not_include_chat_history_by_default(
    mock_ragas_event_listener, run_agent_input_with_history
) -> None:
    captured_inputs: dict[str, Any] = {}

    class CapturingCrew(CrewForTest):
        def kickoff_async(self, *, inputs: dict[str, Any]) -> CrewOutput | CrewStreamingOutput:  # type: ignore[override]
            captured_inputs.update(inputs)
            return super().kickoff_async(inputs=inputs)

    out = CrewOutput(raw="agent result")
    agent = AgentForTest(
        out, api_base="https://x/", api_key="k", verbose=False, crew=CapturingCrew(out)
    )

    _ = [event async for event in agent.invoke(run_agent_input_with_history)]

    assert captured_inputs["topic"] == "Follow-up"
    assert "chat_history" not in captured_inputs


async def test_invoke_overwrites_blank_chat_history_placeholder(
    mock_ragas_event_listener, run_agent_input_with_history
) -> None:
    captured_inputs: dict[str, Any] = {}

    class CapturingCrew(CrewForTest):
        def kickoff_async(self, *, inputs: dict[str, Any]) -> CrewOutput | CrewStreamingOutput:  # type: ignore[override]
            captured_inputs.update(inputs)
            return super().kickoff_async(inputs=inputs)

    class AgentWithPlaceholder(AgentForTest):
        def make_kickoff_inputs(self, user_prompt_content: str) -> dict[str, Any]:
            return {"topic": user_prompt_content, "chat_history": ""}

    out = CrewOutput(raw="agent result")
    agent = AgentWithPlaceholder(
        out, api_base="https://x/", api_key="k", verbose=False, crew=CapturingCrew(out)
    )

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
    mock_ragas_event_listener, run_agent_input_with_history
) -> None:
    captured_inputs: dict[str, Any] = {}

    class CapturingCrew(CrewForTest):
        def kickoff_async(self, *, inputs: dict[str, Any]) -> CrewOutput | CrewStreamingOutput:  # type: ignore[override]
            captured_inputs.update(inputs)
            return super().kickoff_async(inputs=inputs)

    class AgentWithOverride(AgentForTest):
        def make_kickoff_inputs(self, user_prompt_content: str) -> dict[str, Any]:
            return {"topic": user_prompt_content, "chat_history": "CUSTOM OVERRIDE"}

    out = CrewOutput(raw="agent result")
    agent = AgentWithOverride(
        out, api_base="https://x/", api_key="k", verbose=False, crew=CapturingCrew(out)
    )

    _ = [event async for event in agent.invoke(run_agent_input_with_history)]

    assert captured_inputs["chat_history"] == "CUSTOM OVERRIDE"


# --- datarobot_agent_class_from_crew ---


def test_datarobot_agent_class_from_crew_subclass_and_kickoff_inputs() -> None:
    crew = MagicMock()
    crew.verbose = True
    ca = _mock_crewai_agent()
    cb = _mock_crewai_agent()
    ta, tb = MagicMock(), MagicMock()

    def kickoff(u: str) -> dict[str, Any]:
        return {"topic": u, "extra": 1}

    agent_cls = datarobot_agent_class_from_crew(crew, [ca, cb], [ta, tb], kickoff)
    assert issubclass(agent_cls, CrewAIAgent)

    instance = agent_cls(api_base="https://x/", api_key="k", verbose=False)
    assert instance.crew is crew
    assert instance.agents == [ca, cb]
    assert instance.tasks == [ta, tb]
    assert instance.make_kickoff_inputs("hello") == {"topic": "hello", "extra": 1}


def test_datarobot_agent_class_from_crew_set_tools_merges_with_original() -> None:
    crew = MagicMock()
    crew.verbose = True
    orig_a = MagicMock()
    orig_b = MagicMock()
    mcp_tool = MagicMock()

    ca = _mock_crewai_agent()
    ca.tools = [orig_a]
    cb = _mock_crewai_agent()
    cb.tools = [orig_b]

    agent_cls = datarobot_agent_class_from_crew(
        crew, [ca, cb], [MagicMock(), MagicMock()], lambda u: {"topic": u}
    )
    instance = agent_cls()
    instance.set_tools([mcp_tool])

    assert ca.tools == [orig_a, mcp_tool]
    assert cb.tools == [orig_b, mcp_tool]
    assert instance.tools == [mcp_tool]


def test_crewai_agent_set_llm_skips_propagation_when_none() -> None:
    """Pre-built CrewAI agents keep their LLM when BaseAgent is constructed without llm."""

    class _Agent(CrewAIAgent):
        def __init__(self) -> None:
            self._preserved = object()
            self._inner = _mock_crewai_agent()
            self._inner.llm = self._preserved
            self._inner.function_calling_llm = self._preserved
            self._test_crew = CrewForTest()
            super().__init__(api_key="k", api_base="https://x/", verbose=False)

        @property
        def agents(self) -> list[Any]:
            return [self._inner]

        @property
        def tasks(self) -> list[Any]:
            return [MagicMock()]

        def make_kickoff_inputs(self, user_prompt_content: str) -> dict[str, Any]:
            return {"topic": user_prompt_content}

        @property
        def crew(self) -> Any:
            return self._test_crew

    agent = _Agent()
    assert agent._inner.llm is agent._preserved
    assert agent._inner.function_calling_llm is agent._preserved


def test_crewai_agent_init_accepts_roles_goals_backstories() -> None:
    agent = AgentForTest(
        verbose=False,
        roles=["R0", "R1"],
        goals=["G0", "G1"],
        backstories=["B0", "B1"],
    )
    a0, a1 = agent._agents_for_test
    assert a0.role == "R0" and a1.role == "R1"
    assert a0.goal == "G0" and a1.goal == "G1"
    assert a0.backstory == "B0" and a1.backstory == "B1"


def test_crewai_agent_init_accepts_execution_settings() -> None:
    agent = AgentForTest(
        verbose=False,
        max_iter=12,
        max_rpm=15,
        max_execution_time=120,
        allow_delegation=False,
        max_retry_limit=3,
        reasoning=True,
        max_reasoning_attempts=2,
    )
    a0, a1 = agent._agents_for_test
    assert a0.max_iter == 12 and a1.max_iter == 12
    assert a0.max_rpm == 15 and a1.max_rpm == 15
    assert a0.max_execution_time == 120 and a1.max_execution_time == 120
    assert a0.allow_delegation is False and a1.allow_delegation is False
    assert a0.max_retry_limit == 3 and a1.max_retry_limit == 3
    assert a0.reasoning is True and a1.reasoning is True
    assert a0.max_reasoning_attempts == 2 and a1.max_reasoning_attempts == 2


def test_crewai_agent_set_roles_goals_backstories_per_agent() -> None:
    agent = AgentForTest(verbose=False)
    a0, a1 = agent._agents_for_test
    a0.role = a0.goal = a0.backstory = "orig0"
    a1.role = a1.goal = a1.backstory = "orig1"

    agent.set_roles(["Planner", "Writer"])
    agent.set_goals(["Plan {topic}", "Write about {topic}"])
    agent.set_backstories(["bs0", "bs1"])

    assert a0.role == "Planner"
    assert a1.role == "Writer"
    assert a0.goal == "Plan {topic}"
    assert a1.goal == "Write about {topic}"
    assert a0.backstory == "bs0"
    assert a1.backstory == "bs1"


def test_crewai_agent_set_roles_length_mismatch() -> None:
    agent = AgentForTest(verbose=False)
    with pytest.raises(ValueError, match="roles length"):
        agent.set_roles(["only-one"])


def test_crewai_agent_singular_setters_require_index_when_multiple_agents() -> None:
    agent = AgentForTest(verbose=False)
    with pytest.raises(ValueError, match="set_role"):
        agent.set_role("X")


def test_crewai_agent_singular_setters_with_agent_index() -> None:
    agent = AgentForTest(verbose=False)
    a0, a1 = agent._agents_for_test
    agent.set_role("R1", agent_index=1)
    agent.set_goal("G0", agent_index=0)
    agent.set_backstory("B1", agent_index=1)
    assert a0.role != "R1"
    assert a1.role == "R1"
    assert a0.goal == "G0"
    assert a1.backstory == "B1"


def test_crewai_agent_singular_setters_single_agent_without_index() -> None:
    class SingleAgentForTest(CrewAIAgent):
        def __init__(self) -> None:
            self._only = _mock_crewai_agent()
            super().__init__(verbose=False)

        @property
        def agents(self) -> list[Any]:
            return [self._only]

        @property
        def tasks(self) -> list[Any]:
            return [MagicMock()]

        def make_kickoff_inputs(self, user_prompt_content: str) -> dict[str, Any]:
            return {"topic": user_prompt_content}

        @property
        def crew(self) -> Any:
            return CrewForTest()

    agent = SingleAgentForTest()
    agent.set_role("Solo")
    agent.set_goal("Do {topic}")
    agent.set_backstory("solo-bs")
    assert agent._only.role == "Solo"
    assert agent._only.goal == "Do {topic}"
    assert agent._only.backstory == "solo-bs"


async def test_invoke_retrieves_and_stores_memory(
    mock_ragas_event_listener, run_agent_input_with_structured_prompt
) -> None:
    captured_inputs: dict[str, Any] = {}

    class CapturingCrew(CrewForTest):
        async def kickoff_async(
            self, *, inputs: dict[str, Any]
        ) -> CrewOutput | CrewStreamingOutput:  # type: ignore[override]
            captured_inputs.update(inputs)
            return await super().kickoff_async(inputs=inputs)

    class AgentWithMemory(AgentForTest):
        def make_kickoff_inputs(self, user_prompt_content: str) -> dict[str, Any]:
            return {"topic": user_prompt_content, "memory": ""}

    # GIVEN a CrewAI agent whose kickoff inputs opt into memory
    memory_client = FakeMemoryClient(retrieved="Use concise answers.")
    out = CrewOutput(raw="agent result")
    agent = AgentWithMemory(
        out,
        api_base="https://x/",
        api_key="k",
        verbose=False,
        memory_client=memory_client,
        crew=CapturingCrew(out),
    )

    # WHEN invoke is called
    _ = [event async for event in agent.invoke(run_agent_input_with_structured_prompt)]

    # THEN the retrieved memory is injected and the user prompt is persisted after success
    assert captured_inputs["memory"] == "Use concise answers."
    assert memory_client.retrieve_calls == [
        {
            "prompt": '{"topic": "AI"}',
            "run_id": None,
            "agent_id": "AgentWithMemory",
            "app_id": "tests.crewai.test_agent",
            "attributes": {"thread_id": "thread_id"},
        }
    ]
    assert memory_client.store_calls == [
        {
            "user_message": '{"topic": "AI"}',
            "run_id": "run_id",
            "agent_id": "AgentWithMemory",
            "app_id": "tests.crewai.test_agent",
            "attributes": {"thread_id": "thread_id"},
        }
    ]


async def test_invoke_skips_memory_when_prompt_does_not_use_it(
    mock_ragas_event_listener, run_agent_input_with_structured_prompt
) -> None:
    # GIVEN a CrewAI agent whose kickoff inputs do not opt into memory
    memory_client = FakeMemoryClient(retrieved="Use concise answers.")
    out = CrewOutput(raw="agent result")
    agent = AgentForTest(
        out,
        api_base="https://x/",
        api_key="k",
        verbose=False,
        memory_client=memory_client,
    )

    # WHEN invoke is called
    _ = [event async for event in agent.invoke(run_agent_input_with_structured_prompt)]

    # THEN memory retrieval and storage are both skipped
    assert memory_client.retrieve_calls == []
    assert memory_client.store_calls == []


async def test_invoke_does_not_overwrite_non_empty_memory_override(
    mock_ragas_event_listener, run_agent_input_with_structured_prompt
) -> None:
    captured_inputs: dict[str, Any] = {}

    class CapturingCrew(CrewForTest):
        async def kickoff_async(
            self, *, inputs: dict[str, Any]
        ) -> CrewOutput | CrewStreamingOutput:  # type: ignore[override]
            captured_inputs.update(inputs)
            return await super().kickoff_async(inputs=inputs)

    class AgentWithMemoryOverride(AgentForTest):
        def make_kickoff_inputs(self, user_prompt_content: str) -> dict[str, Any]:
            return {"topic": user_prompt_content, "memory": "CUSTOM MEMORY"}

    # GIVEN a CrewAI agent that provides its own memory override
    memory_client = FakeMemoryClient(retrieved="Use concise answers.")
    out = CrewOutput(raw="agent result")
    agent = AgentWithMemoryOverride(
        out,
        api_base="https://x/",
        api_key="k",
        verbose=False,
        memory_client=memory_client,
        crew=CapturingCrew(out),
    )

    # WHEN invoke is called
    _ = [event async for event in agent.invoke(run_agent_input_with_structured_prompt)]

    # THEN the explicit override is preserved and the turn is still stored
    assert captured_inputs["memory"] == "CUSTOM MEMORY"
    assert memory_client.retrieve_calls == []
    assert memory_client.store_calls == [
        {
            "user_message": '{"topic": "AI"}',
            "run_id": "run_id",
            "agent_id": "AgentWithMemoryOverride",
            "app_id": "tests.crewai.test_agent",
            "attributes": {"thread_id": "thread_id"},
        }
    ]


async def test_invoke_gracefully_degrades_when_memory_fails(
    mock_ragas_event_listener, run_agent_input_with_structured_prompt
) -> None:
    captured_inputs: dict[str, Any] = {}

    class CapturingCrew(CrewForTest):
        async def kickoff_async(
            self, *, inputs: dict[str, Any]
        ) -> CrewOutput | CrewStreamingOutput:  # type: ignore[override]
            captured_inputs.update(inputs)
            return await super().kickoff_async(inputs=inputs)

    class AgentWithMemory(AgentForTest):
        def make_kickoff_inputs(self, user_prompt_content: str) -> dict[str, Any]:
            return {"topic": user_prompt_content, "memory": ""}

    # GIVEN a CrewAI agent whose memory provider errors at runtime
    out = CrewOutput(raw="agent result")
    agent = AgentWithMemory(
        out,
        api_base="https://x/",
        api_key="k",
        verbose=False,
        memory_client=FailingMemoryClient(),
        crew=CapturingCrew(out),
    )

    # WHEN invoke is called
    events = [event async for event in agent.invoke(run_agent_input_with_structured_prompt)]

    # THEN the run still completes and the memory placeholder remains empty
    assert events
    assert isinstance(events[-1][0], RunFinishedEvent)
    assert captured_inputs["memory"] == ""


async def test_invoke_does_not_store_memory_when_run_fails(
    mock_ragas_event_listener, run_agent_input_with_structured_prompt
) -> None:
    class FailingCrew(CrewForTest):
        async def kickoff_async(
            self, *, inputs: dict[str, Any]
        ) -> CrewOutput | CrewStreamingOutput:  # type: ignore[override]
            raise RuntimeError("crew failed")

    class AgentWithMemory(AgentForTest):
        def make_kickoff_inputs(self, user_prompt_content: str) -> dict[str, Any]:
            return {"topic": user_prompt_content, "memory": ""}

    # GIVEN a CrewAI agent whose run fails after retrieving memory
    memory_client = FakeMemoryClient(retrieved="Use concise answers.")
    out = CrewOutput(raw="agent result")
    agent = AgentWithMemory(
        out,
        api_base="https://x/",
        api_key="k",
        verbose=False,
        memory_client=memory_client,
        crew=FailingCrew(out),
    )

    # WHEN invoke is called and the crew fails
    with pytest.raises(RuntimeError, match="crew failed"):
        _ = [event async for event in agent.invoke(run_agent_input_with_structured_prompt)]

    # THEN retrieval may happen, but storage is skipped because the run never finishes
    assert memory_client.retrieve_calls == [
        {
            "prompt": '{"topic": "AI"}',
            "run_id": None,
            "agent_id": "AgentWithMemory",
            "app_id": "tests.crewai.test_agent",
            "attributes": {"thread_id": "thread_id"},
        }
    ]
    assert memory_client.store_calls == []
