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
import queue
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

from crewai.types.streaming import CrewStreamingOutput
from crewai.types.streaming import StreamChunk
from crewai.types.streaming import StreamChunkType
import pytest
from ag_ui.core import ReasoningEndEvent
from ag_ui.core import ReasoningMessageContentEvent
from ag_ui.core import ReasoningMessageEndEvent
from ag_ui.core import ReasoningMessageStartEvent
from ag_ui.core import ReasoningStartEvent
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

    async def akickoff(self, *, inputs: dict[str, Any]) -> CrewOutput | CrewStreamingOutput:  # type: ignore[name-defined]
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
    def __init__(self, crew_output: CrewOutput | None = None, **kwargs: Any) -> None:
        crew_kw = kwargs.pop("crew", None)
        self.crew_output = crew_output
        self._crew_for_test: Any = crew_kw if crew_kw is not None else CrewForTest(crew_output)
        self._agents_for_test: list[MagicMock] = [_mock_crewai_agent(), _mock_crewai_agent()]
        self._tasks_for_test: list[MagicMock] = [MagicMock(), MagicMock()]
        super().__init__(**kwargs)

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


async def test_invoke_resets_agent_executors_per_request(
    run_agent_input, mock_ragas_event_listener
) -> None:
    """Each request must start with fresh agent executors. CrewAI caches each agent's executor
    and never clears its accumulated ``messages``/``iterations`` on reuse, so reusing the crew
    across requests would leak prior-request state (stale tool_use history -> bedrock errors;
    bloated context -> the model leaking tool calls as text). ``invoke`` resets them per run.
    """
    agent = AgentForTest(CrewOutput(raw="ok"), api_base="https://x/", api_key="k")
    # Simulate executors left over from a previous request.
    for a in agent.agents:
        a.agent_executor = object()

    _ = [event async for event in agent.invoke(run_agent_input)]

    assert all(a.agent_executor is None for a in agent.agents)


async def test_invoke_does_not_include_chat_history_by_default(
    mock_ragas_event_listener, run_agent_input_with_history
) -> None:
    captured_inputs: dict[str, Any] = {}

    class CapturingCrew(CrewForTest):
        def akickoff(self, *, inputs: dict[str, Any]) -> CrewOutput | CrewStreamingOutput:  # type: ignore[override]
            captured_inputs.update(inputs)
            return super().akickoff(inputs=inputs)

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
        def akickoff(self, *, inputs: dict[str, Any]) -> CrewOutput | CrewStreamingOutput:  # type: ignore[override]
            captured_inputs.update(inputs)
            return super().akickoff(inputs=inputs)

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
        def akickoff(self, *, inputs: dict[str, Any]) -> CrewOutput | CrewStreamingOutput:  # type: ignore[override]
            captured_inputs.update(inputs)
            return super().akickoff(inputs=inputs)

    class AgentWithOverride(AgentForTest):
        def make_kickoff_inputs(self, user_prompt_content: str) -> dict[str, Any]:
            return {"topic": user_prompt_content, "chat_history": "CUSTOM OVERRIDE"}

    out = CrewOutput(raw="agent result")
    agent = AgentWithOverride(
        out, api_base="https://x/", api_key="k", verbose=False, crew=CapturingCrew(out)
    )

    _ = [event async for event in agent.invoke(run_agent_input_with_history)]

    assert captured_inputs["chat_history"] == "CUSTOM OVERRIDE"


async def test_invoke_includes_tool_calls_in_history(
    mock_ragas_event_listener, run_agent_input_with_tool_history
) -> None:
    """Tool calls appear in the injected chat_history text in both content cases."""
    captured_inputs: dict[str, Any] = {}

    class CapturingCrew(CrewForTest):
        def akickoff(self, *, inputs: dict[str, Any]) -> CrewOutput | CrewStreamingOutput:  # type: ignore[override]
            captured_inputs.update(inputs)
            return super().akickoff(inputs=inputs)

    class AgentWithPlaceholder(AgentForTest):
        def make_kickoff_inputs(self, user_prompt_content: str) -> dict[str, Any]:
            return {"topic": user_prompt_content, "chat_history": ""}

    out = CrewOutput(raw="agent result")
    agent = AgentWithPlaceholder(
        out, api_base="https://x/", api_key="k", verbose=False, crew=CapturingCrew(out)
    )

    _ = [event async for event in agent.invoke(run_agent_input_with_tool_history)]

    history = captured_inputs["chat_history"]
    # Assistant turn with content AND a tool call: both are shown.
    assert "Let me check the weather. [tool_calls] get_weather" in history
    # Tool-call-only assistant turn (empty content): the call is still shown.
    assert "[tool_calls] log_event" in history


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


def test_datarobot_agent_class_from_crew_neutralizes_kickoff_storage() -> None:
    # The supplied crew's leaking sqlite kickoff-outputs handler must be replaced
    # with the no-op so a long-lived serve process can't exhaust its fd table.
    from datarobot_genai.crewai.kickoff_storage import _NoOpTaskOutputHandler

    crew = MagicMock()
    crew.verbose = True
    crew._task_output_handler = MagicMock()  # stand-in for the real sqlite handler

    datarobot_agent_class_from_crew(crew, [_mock_crewai_agent()], [MagicMock()], lambda u: {})

    assert isinstance(crew._task_output_handler, _NoOpTaskOutputHandler)


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


def test_set_tools_does_not_accumulate_across_requests() -> None:
    # The crew/agents are reused across requests; a new agent instance per request re-captures
    # the (already-injected) tools as "original". Without dedupe, injected tools accumulate
    # (-> CrewAI mangles duplicates into name_2/name_3 and bedrock rejects the calls).
    crew = MagicMock()
    crew.verbose = True
    orig = MagicMock()
    orig.name = "base_tool"
    mcp = MagicMock()
    mcp.name = "mcp_tool"

    ca = _mock_crewai_agent()
    ca.tools = [orig]
    task = MagicMock()
    task.agent = ca
    task.tools = [orig]

    agent_cls = datarobot_agent_class_from_crew(crew, [ca], [task], lambda u: {})
    for _ in range(3):  # three sequential requests reusing the same shared agent
        agent_cls().set_tools([mcp])

    assert [t.name for t in ca.tools] == ["base_tool", "mcp_tool"]  # not 3x mcp_tool
    assert [t.name for t in task.tools] == ["base_tool", "mcp_tool"]


def test_set_tools_uses_fresh_tool_when_name_reused_across_requests() -> None:
    # A later request's fresh same-named tool MUST replace the earlier one: each request's tool is
    # bound to a per-request event loop that closes when the request ends.
    crew = MagicMock()
    crew.verbose = True
    orig = MagicMock()
    orig.name = "base_tool"

    ca = _mock_crewai_agent()
    ca.tools = [orig]
    task = MagicMock()
    task.agent = ca
    task.tools = [orig]

    agent_cls = datarobot_agent_class_from_crew(crew, [ca], [task], lambda u: {})

    mcp_req1 = MagicMock()
    mcp_req1.name = "search"
    agent_cls().set_tools([mcp_req1])  # request 1 (its loop closes afterward)

    mcp_req2 = MagicMock()
    mcp_req2.name = "search"
    agent_cls().set_tools([mcp_req2])  # request 2: same name, fresh object/loop

    assert [t for t in ca.tools if t.name == "search"] == [mcp_req2]  # req2 survives, not req1
    assert [t for t in task.tools if t.name == "search"] == [mcp_req2]
    assert [t.name for t in ca.tools] == ["base_tool", "search"]  # still deduped, no accumulation


def test_set_tools_propagates_injected_tools_to_tasks() -> None:
    """Injected tools must reach each task, not just the agent.

    CrewAI snapshots ``agent.tools`` into ``task.tools`` at Crew build and runs off the
    snapshot; without re-syncing, injected MCP tools never reach the model.
    """
    crew = MagicMock()
    crew.verbose = True
    orig = MagicMock()
    mcp_tool = MagicMock()

    ca = _mock_crewai_agent()
    ca.tools = [orig]
    task = MagicMock()
    task.agent = ca
    task.tools = [orig]  # crewai's stale construction-time snapshot

    agent_cls = datarobot_agent_class_from_crew(crew, [ca], [task], lambda u: {"topic": u})
    agent_cls().set_tools([mcp_tool])

    assert ca.tools == [orig, mcp_tool]
    assert task.tools == [orig, mcp_tool]  # task re-synced to the agent's full toolset


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


# --- Streaming AG-UI message-lifecycle tests ---


def _text_chunk(content: str, agent_role: str) -> StreamChunk:
    return StreamChunk(
        content=content,
        chunk_type=StreamChunkType.TEXT,
        agent_role=agent_role,
        task_name=f"{agent_role}-task",
    )


class _FakeStreamingOutput(CrewStreamingOutput):
    """CrewStreamingOutput double that yields a fixed chunk list and a result."""

    def __init__(self, chunks: list[StreamChunk], result: CrewOutput) -> None:
        super().__init__(async_iterator=self._iter(chunks))
        self._fixed_result = result

    @staticmethod
    async def _iter(chunks: list[StreamChunk]):  # type: ignore[no-untyped-def]
        for c in chunks:
            yield c

    @property
    def result(self) -> CrewOutput:  # type: ignore[override]
        return self._fixed_result


async def test_invoke_streaming_emits_separate_messages_per_agent_role(
    mock_ragas_event_listener, run_agent_input
) -> None:
    # GIVEN a streaming crew that emits chunks under two different agent roles
    chunks = [
        _text_chunk("plan-a", "Planner"),
        _text_chunk("plan-b", "Planner"),
        _text_chunk("write-a", "Writer"),
        _text_chunk("write-b", "Writer"),
    ]
    result = CrewOutput(
        raw="ignored",
        token_usage=UsageMetrics(completion_tokens=1, prompt_tokens=2, total_tokens=3),
    )
    streaming = _FakeStreamingOutput(chunks, result)
    agent = AgentForTest(api_base="https://x/", api_key="k", verbose=False)
    agent._crew_for_test = CrewForTest(streaming)

    # WHEN we collect the AG-UI event stream
    events = [e async for (e, _, _) in agent.invoke(run_agent_input)]

    # THEN the sequence is valid per the AG-UI verifier
    validate_sequence(events)

    # THEN each step gets its own TEXT_MESSAGE_START / _END pair
    starts = [e for e in events if isinstance(e, TextMessageStartEvent)]
    ends = [e for e in events if isinstance(e, TextMessageEndEvent)]
    contents = [e for e in events if isinstance(e, TextMessageContentEvent)]
    assert len(starts) == 2, "expected one TEXT_MESSAGE_START per agent role"
    assert len(ends) == 2, "expected one TEXT_MESSAGE_END per agent role"

    # THEN the two messages have distinct ids
    assert starts[0].message_id != starts[1].message_id

    # THEN content events are partitioned by message_id (no cross-agent leakage)
    by_id: dict[str, list[str]] = {}
    for c in contents:
        by_id.setdefault(c.message_id, []).append(c.delta)
    assert by_id[starts[0].message_id] == ["plan-a", "plan-b"]
    assert by_id[starts[1].message_id] == ["write-a", "write-b"]

    # THEN each step boundary is closed before the next one opens
    step_started = [e for e in events if isinstance(e, StepStartedEvent)]
    step_finished = [e for e in events if isinstance(e, StepFinishedEvent)]
    assert [s.step_name for s in step_started] == ["Planner", "Writer"]
    assert [s.step_name for s in step_finished] == ["Planner", "Writer"]

    # THEN the Planner's TEXT_MESSAGE_END is emitted before STEP_FINISHED Planner,
    # which is emitted before STEP_STARTED Writer.
    planner_end_idx = events.index(ends[0])
    planner_step_finish_idx = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, StepFinishedEvent) and e.step_name == "Planner"
    )
    writer_step_start_idx = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, StepStartedEvent) and e.step_name == "Writer"
    )
    assert planner_end_idx < planner_step_finish_idx < writer_step_start_idx


async def test_invoke_streaming_skips_empty_text_chunks(
    mock_ragas_event_listener, run_agent_input
) -> None:
    # GIVEN a streaming crew that emits empty TEXT chunks alongside real content
    chunks = [
        _text_chunk("", "Solo"),
        _text_chunk("hello", "Solo"),
        _text_chunk("", "Solo"),
        _text_chunk(" world", "Solo"),
    ]
    result = CrewOutput(raw="ignored")
    streaming = _FakeStreamingOutput(chunks, result)
    agent = AgentForTest(api_base="https://x/", api_key="k", verbose=False)
    agent._crew_for_test = CrewForTest(streaming)

    # WHEN we collect the AG-UI event stream
    events = [e async for (e, _, _) in agent.invoke(run_agent_input)]

    # THEN the sequence is valid and empty chunks do not open a text message
    validate_sequence(events)
    starts = [e for e in events if isinstance(e, TextMessageStartEvent)]
    ends = [e for e in events if isinstance(e, TextMessageEndEvent)]
    contents = [e for e in events if isinstance(e, TextMessageContentEvent)]
    assert len(starts) == 1
    assert len(ends) == 1
    assert [c.delta for c in contents] == ["hello", " world"]
    assert all(c.delta for c in contents)


async def test_invoke_streaming_single_agent_role_uses_single_message(
    mock_ragas_event_listener, run_agent_input
) -> None:
    # GIVEN a streaming crew that only emits chunks for one agent role
    chunks = [
        _text_chunk("hello-", "Solo"),
        _text_chunk("world", "Solo"),
    ]
    result = CrewOutput(raw="ignored")
    streaming = _FakeStreamingOutput(chunks, result)
    agent = AgentForTest(api_base="https://x/", api_key="k", verbose=False)
    agent._crew_for_test = CrewForTest(streaming)

    # WHEN we collect the AG-UI event stream
    events = [e async for (e, _, _) in agent.invoke(run_agent_input)]

    # THEN the sequence is valid and there is exactly one text message
    validate_sequence(events)
    starts = [e for e in events if isinstance(e, TextMessageStartEvent)]
    ends = [e for e in events if isinstance(e, TextMessageEndEvent)]
    assert len(starts) == 1
    assert len(ends) == 1
    assert starts[0].message_id == ends[0].message_id


async def test_invoke_streaming_closes_reasoning_message_at_step_boundary(
    mock_ragas_event_listener, run_agent_input
) -> None:
    # GIVEN a streaming crew where the first agent emits a reasoning chunk and
    # the second agent emits a normal text chunk. The agent_role transition
    # must close REASONING_MESSAGE_END and REASONING_END for the outgoing step
    # before STEP_FINISHED is emitted.
    planner_chunk = _text_chunk("plan-thoughts", "Planner")
    writer_chunk = _text_chunk("write-text", "Writer")
    result = CrewOutput(raw="ignored")

    captured: list[Any] = []

    class _CapturingStreamingListener:
        def __init__(self) -> None:
            self.reasoning_event = False
            self.active_agent_role = ""
            self.tool_call_events: queue.Queue[Any] = queue.Queue()
            captured.append(self)

        def setup_listeners(self, _bus: Any) -> None:
            pass

    class _ReasoningTransitionStreamingOutput(CrewStreamingOutput):
        def __init__(self) -> None:
            super().__init__(async_iterator=self._iter())

        @staticmethod
        async def _iter():  # type: ignore[no-untyped-def]
            captured[0].reasoning_event = True
            yield planner_chunk
            captured[0].reasoning_event = False
            yield writer_chunk

        @property
        def result(self) -> CrewOutput:  # type: ignore[override]
            return result

    streaming = _ReasoningTransitionStreamingOutput()
    agent = AgentForTest(api_base="https://x/", api_key="k", verbose=False)
    agent._crew_for_test = CrewForTest(streaming)

    # WHEN we collect the AG-UI event stream
    with patch(
        "datarobot_genai.crewai.agent.CrewAIStreamingEventListener",
        _CapturingStreamingListener,
    ):
        events = [e async for (e, _, _) in agent.invoke(run_agent_input)]

    # THEN the sequence is valid per the AG-UI verifier
    validate_sequence(events)

    # THEN the Planner emits exactly one reasoning lifecycle and the Writer
    # emits exactly one text lifecycle
    reasoning_starts = [e for e in events if isinstance(e, ReasoningStartEvent)]
    reasoning_msg_starts = [e for e in events if isinstance(e, ReasoningMessageStartEvent)]
    reasoning_contents = [e for e in events if isinstance(e, ReasoningMessageContentEvent)]
    reasoning_msg_ends = [e for e in events if isinstance(e, ReasoningMessageEndEvent)]
    reasoning_ends = [e for e in events if isinstance(e, ReasoningEndEvent)]
    text_starts = [e for e in events if isinstance(e, TextMessageStartEvent)]
    text_ends = [e for e in events if isinstance(e, TextMessageEndEvent)]
    text_contents = [e for e in events if isinstance(e, TextMessageContentEvent)]
    assert len(reasoning_starts) == 1
    assert len(reasoning_msg_starts) == 1
    assert len(reasoning_msg_ends) == 1
    assert len(reasoning_ends) == 1
    assert len(text_starts) == 1
    assert len(text_ends) == 1

    # THEN reasoning content is on the Planner's message_id and text content
    # is on a fresh message_id (no cross-agent leakage)
    planner_id = reasoning_starts[0].message_id
    writer_id = text_starts[0].message_id
    assert planner_id != writer_id
    assert reasoning_msg_starts[0].message_id == planner_id
    assert reasoning_msg_ends[0].message_id == planner_id
    assert reasoning_ends[0].message_id == planner_id
    assert [c.delta for c in reasoning_contents if c.message_id == planner_id] == ["plan-thoughts"]
    assert text_ends[0].message_id == writer_id
    assert [c.delta for c in text_contents if c.message_id == writer_id] == ["write-text"]

    # THEN REASONING_MESSAGE_END and REASONING_END are emitted before
    # STEP_FINISHED Planner, which is emitted before STEP_STARTED Writer
    reasoning_msg_end_idx = events.index(reasoning_msg_ends[0])
    reasoning_end_idx = events.index(reasoning_ends[0])
    planner_step_finish_idx = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, StepFinishedEvent) and e.step_name == "Planner"
    )
    writer_step_start_idx = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, StepStartedEvent) and e.step_name == "Writer"
    )
    assert (
        reasoning_msg_end_idx < reasoning_end_idx < planner_step_finish_idx < writer_step_start_idx
    )


async def test_invoke_streaming_empty_agent_role_does_not_orphan_a_step(
    mock_ragas_event_listener, run_agent_input
) -> None:
    # GIVEN a streaming crew that emits a chunk with an empty agent_role between
    # two real roles. CrewAI does this at task boundaries (logged as
    # "[] Working on task:"); an empty role must not open a step that can never
    # be closed (regression: RUN_FINISHED while steps are still active).
    chunks = [
        _text_chunk("plan", "Planner"),
        _text_chunk("interlude", ""),
        _text_chunk("write", "Writer"),
    ]
    result = CrewOutput(raw="ignored")
    streaming = _FakeStreamingOutput(chunks, result)
    agent = AgentForTest(api_base="https://x/", api_key="k", verbose=False)
    agent._crew_for_test = CrewForTest(streaming)

    # WHEN we collect the AG-UI event stream
    events = [e async for (e, _, _) in agent.invoke(run_agent_input)]

    # THEN the sequence is valid: an empty agent_role leaves no step active at
    # RUN_FINISHED
    validate_sequence(events)

    # THEN no step is opened under an empty step_name, and only the real roles
    # produce steps, each opened and closed exactly once
    step_started = [e for e in events if isinstance(e, StepStartedEvent)]
    step_finished = [e for e in events if isinstance(e, StepFinishedEvent)]
    assert [s.step_name for s in step_started] == ["Planner", "Writer"]
    assert [s.step_name for s in step_finished] == ["Planner", "Writer"]


async def test_invoke_streaming_sources_step_role_from_bus_when_chunks_lack_role(
    mock_ragas_event_listener, run_agent_input
) -> None:
    # GIVEN a real multi-agent crew: CrewAI hardcodes chunk.agent_role="" for Crew streams,
    # so the active agent is known only from the bus (listener.active_agent_role).
    captured: list[Any] = []

    class _CapturingStreamingListener:
        def __init__(self) -> None:
            self.reasoning_event = False
            self.active_agent_role = ""
            self.tool_call_events: queue.Queue[Any] = queue.Queue()
            captured.append(self)

        def setup_listeners(self, _bus: Any) -> None:
            pass

    class _BusRoleStream(CrewStreamingOutput):
        def __init__(self) -> None:
            super().__init__(async_iterator=self._iter())

        @staticmethod
        async def _iter():  # type: ignore[no-untyped-def]
            captured[0].active_agent_role = "Planner"
            yield _text_chunk("plan-a", "")
            yield _text_chunk("plan-b", "")
            captured[0].active_agent_role = "Writer"
            yield _text_chunk("write-a", "")
            yield _text_chunk("write-b", "")

        @property
        def result(self) -> CrewOutput:  # type: ignore[override]
            return CrewOutput(raw="ignored")

    agent = AgentForTest(api_base="https://x/", api_key="k", verbose=False)
    agent._crew_for_test = CrewForTest(_BusRoleStream())

    # WHEN we collect the AG-UI event stream
    with patch(
        "datarobot_genai.crewai.agent.CrewAIStreamingEventListener", _CapturingStreamingListener
    ):
        events = [e async for (e, _, _) in agent.invoke(run_agent_input)]

    # THEN the sequence is valid and steps come from the bus role despite empty chunk roles
    validate_sequence(events)
    step_started = [e for e in events if isinstance(e, StepStartedEvent)]
    step_finished = [e for e in events if isinstance(e, StepFinishedEvent)]
    assert [s.step_name for s in step_started] == ["Planner", "Writer"]
    assert [s.step_name for s in step_finished] == ["Planner", "Writer"]

    # THEN each agent's text is partitioned into its own message (no cross-agent leakage)
    starts = [e for e in events if isinstance(e, TextMessageStartEvent)]
    contents = [e for e in events if isinstance(e, TextMessageContentEvent)]
    assert len(starts) == 2
    assert starts[0].message_id != starts[1].message_id
    by_id: dict[str, list[str]] = {}
    for c in contents:
        by_id.setdefault(c.message_id, []).append(c.delta)
    assert by_id[starts[0].message_id] == ["plan-a", "plan-b"]
    assert by_id[starts[1].message_id] == ["write-a", "write-b"]


async def test_invoke_streaming_emits_tool_call_events(
    mock_ragas_event_listener, run_agent_input
) -> None:
    # GIVEN a streaming crew where a tool fires between two text chunks: the streaming
    # listener queues the AG-UI ToolCall* sequence (as it does for CrewAI's bus events).
    from ag_ui.core import ToolCallArgsEvent
    from ag_ui.core import ToolCallEndEvent
    from ag_ui.core import ToolCallResultEvent
    from ag_ui.core import ToolCallStartEvent

    from datarobot_genai.crewai.streaming_events import ToolCallRecord

    chunk1 = _text_chunk("counting", "Writer")
    chunk2 = _text_chunk("done", "Writer")
    result = CrewOutput(raw="ignored")
    tcid = "tc-abc"
    captured: list[Any] = []

    class _CapturingStreamingListener:
        def __init__(self) -> None:
            self.reasoning_event = False
            self.active_agent_role = ""
            self.tool_call_events: queue.Queue[Any] = queue.Queue()
            captured.append(self)

        def setup_listeners(self, _bus: Any) -> None:
            pass

    class _StreamWithToolCall(CrewStreamingOutput):
        def __init__(self) -> None:
            super().__init__(async_iterator=self._iter())

        @staticmethod
        async def _iter():  # type: ignore[no-untyped-def]
            yield chunk1
            q = captured[0].tool_call_events
            q.put(
                ToolCallRecord(
                    kind="call", tool_call_id=tcid, name="word_counter", args='{"text": "a b c"}'
                )
            )
            q.put(ToolCallRecord(kind="result", tool_call_id=tcid, content="Word count: 3"))
            yield chunk2

        @property
        def result(self) -> CrewOutput:  # type: ignore[override]
            return result

    agent = AgentForTest(api_base="https://x/", api_key="k", verbose=False)
    agent._crew_for_test = CrewForTest(_StreamWithToolCall())

    with patch(
        "datarobot_genai.crewai.agent.CrewAIStreamingEventListener", _CapturingStreamingListener
    ):
        events = [e async for (e, _, _) in agent.invoke(run_agent_input)]

    # THEN the sequence (with the tool call interleaved) is valid per the AG-UI verifier
    validate_sequence(events)

    # THEN the full ToolCall* set is emitted, all sharing the one tool_call_id
    starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
    args = [e for e in events if isinstance(e, ToolCallArgsEvent)]
    ends = [e for e in events if isinstance(e, ToolCallEndEvent)]
    results = [e for e in events if isinstance(e, ToolCallResultEvent)]
    assert len(starts) == len(args) == len(ends) == len(results) == 1
    assert starts[0].tool_call_name == "word_counter"
    assert (
        starts[0].tool_call_id
        == args[0].tool_call_id
        == ends[0].tool_call_id
        == results[0].tool_call_id
        == tcid
    )

    # THEN the args, result content/role, and reused id are correct. validate_sequence does NOT
    # check TOOL_CALL_RESULT, so these must be asserted explicitly.
    assert args[0].delta == '{"text": "a b c"}'
    assert results[0].content == "Word count: 3"
    assert results[0].role == "tool"
    assert results[0].message_id == tcid  # result reuses the tool_call_id

    # THEN the call attaches to the just-closed text bubble; a fresh bubble opens after it,
    # and content does not leak across the two messages.
    type_names = [type(e).__name__ for e in events]
    text_starts = [e for e in events if isinstance(e, TextMessageStartEvent)]
    text_contents = [e for e in events if isinstance(e, TextMessageContentEvent)]
    assert len(text_starts) == 2
    assert starts[0].parent_message_id == text_starts[0].message_id
    assert text_starts[0].message_id != text_starts[1].message_id
    by_id: dict[str, list[str]] = {}
    for c in text_contents:
        by_id.setdefault(c.message_id, []).append(c.delta)
    assert by_id[text_starts[0].message_id] == ["counting"]
    assert by_id[text_starts[1].message_id] == ["done"]

    # THEN ordering: text closes, then END before RESULT, then the next text message opens
    text_start_idxs = [i for i, e in enumerate(events) if isinstance(e, TextMessageStartEvent)]
    second_text_start_idx = text_start_idxs[1]
    assert type_names.index("TextMessageEndEvent") < type_names.index("ToolCallStartEvent")
    assert (
        type_names.index("ToolCallEndEvent")
        < type_names.index("ToolCallResultEvent")
        < second_text_start_idx
    )


async def test_invoke_streaming_emits_tool_error_as_result(
    mock_ragas_event_listener, run_agent_input
) -> None:
    # GIVEN a tool that errors: the listener queues a result record carrying the error text
    from ag_ui.core import ToolCallResultEvent

    from datarobot_genai.crewai.streaming_events import ToolCallRecord

    captured: list[Any] = []

    class _CapturingStreamingListener:
        def __init__(self) -> None:
            self.reasoning_event = False
            self.active_agent_role = ""
            self.tool_call_events: queue.Queue[Any] = queue.Queue()
            captured.append(self)

        def setup_listeners(self, _bus: Any) -> None:
            pass

    class _StreamWithToolError(CrewStreamingOutput):
        def __init__(self) -> None:
            super().__init__(async_iterator=self._iter())

        @staticmethod
        async def _iter():  # type: ignore[no-untyped-def]
            yield _text_chunk("trying", "Writer")
            q = captured[0].tool_call_events
            q.put(ToolCallRecord(kind="call", tool_call_id="tc-err", name="flaky", args="{}"))
            q.put(ToolCallRecord(kind="result", tool_call_id="tc-err", content="Error: boom"))
            yield _text_chunk("recovered", "Writer")

        @property
        def result(self) -> CrewOutput:  # type: ignore[override]
            return CrewOutput(raw="ignored")

    agent = AgentForTest(api_base="https://x/", api_key="k", verbose=False)
    agent._crew_for_test = CrewForTest(_StreamWithToolError())

    with patch(
        "datarobot_genai.crewai.agent.CrewAIStreamingEventListener", _CapturingStreamingListener
    ):
        events = [e async for (e, _, _) in agent.invoke(run_agent_input)]

    # THEN the error surfaces as a TOOL_CALL_RESULT and the stream still ends cleanly
    validate_sequence(events)
    results = [e for e in events if isinstance(e, ToolCallResultEvent)]
    assert len(results) == 1
    assert results[0].content == "Error: boom"
    assert results[0].role == "tool"
    assert "RunFinishedEvent" in [type(e).__name__ for e in events]


async def test_invoke_streaming_tool_call_across_agent_step_boundary(
    mock_ragas_event_listener, run_agent_input
) -> None:
    # GIVEN a tool call queued during the Planner's turn, with a Writer chunk following: the
    # top-of-loop tool drain and the role-change step close must not double-close or orphan a
    # step. (Regression guard for the two close paths interacting.)
    from ag_ui.core import ToolCallStartEvent

    from datarobot_genai.crewai.streaming_events import ToolCallRecord

    captured: list[Any] = []

    class _CapturingStreamingListener:
        def __init__(self) -> None:
            self.reasoning_event = False
            self.active_agent_role = ""
            self.tool_call_events: queue.Queue[Any] = queue.Queue()
            captured.append(self)

        def setup_listeners(self, _bus: Any) -> None:
            pass

    class _Stream(CrewStreamingOutput):
        def __init__(self) -> None:
            super().__init__(async_iterator=self._iter())

        @staticmethod
        async def _iter():  # type: ignore[no-untyped-def]
            yield _text_chunk("plan", "Planner")
            q = captured[0].tool_call_events
            q.put(ToolCallRecord(kind="call", tool_call_id="tc1", name="word_counter", args="{}"))
            q.put(ToolCallRecord(kind="result", tool_call_id="tc1", content="1"))
            yield _text_chunk("write", "Writer")

        @property
        def result(self) -> CrewOutput:  # type: ignore[override]
            return CrewOutput(raw="ignored")

    agent = AgentForTest(api_base="https://x/", api_key="k", verbose=False)
    agent._crew_for_test = CrewForTest(_Stream())

    with patch(
        "datarobot_genai.crewai.agent.CrewAIStreamingEventListener", _CapturingStreamingListener
    ):
        events = [e async for (e, _, _) in agent.invoke(run_agent_input)]

    # THEN nesting is valid, both steps open/close exactly once, and the tool call is emitted
    # (as the Planner's, before the Planner step closes) without orphaning a step.
    validate_sequence(events)
    step_started = [e for e in events if isinstance(e, StepStartedEvent)]
    step_finished = [e for e in events if isinstance(e, StepFinishedEvent)]
    assert [s.step_name for s in step_started] == ["Planner", "Writer"]
    assert [s.step_name for s in step_finished] == ["Planner", "Writer"]
    assert len([e for e in events if isinstance(e, ToolCallStartEvent)]) == 1
    type_names = [type(e).__name__ for e in events]
    # the tool call (and its result) belong to the Planner: it lands inside the Planner step,
    # and its parent is the Planner's text bubble.
    starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
    text_starts = [e for e in events if isinstance(e, TextMessageStartEvent)]
    planner_finish_idx = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, StepFinishedEvent) and e.step_name == "Planner"
    )
    assert type_names.index("ToolCallStartEvent") < planner_finish_idx
    assert type_names.index("ToolCallResultEvent") < planner_finish_idx
    assert starts[0].parent_message_id == text_starts[0].message_id


async def test_invoke_streaming_tool_call_before_any_text_has_empty_parent(
    mock_ragas_event_listener, run_agent_input
) -> None:
    # GIVEN a tool that fires before any text/reasoning bubble is open: parent_message_id must be
    # "" (not a message_id that never got a TextMessageStart).
    from ag_ui.core import ToolCallStartEvent

    from datarobot_genai.crewai.streaming_events import ToolCallRecord

    captured: list[Any] = []

    class _CapturingStreamingListener:
        def __init__(self) -> None:
            self.reasoning_event = False
            self.active_agent_role = ""
            self.tool_call_events: queue.Queue[Any] = queue.Queue()
            captured.append(self)

        def setup_listeners(self, _bus: Any) -> None:
            pass

    class _Stream(CrewStreamingOutput):
        def __init__(self) -> None:
            super().__init__(async_iterator=self._iter())

        @staticmethod
        async def _iter():  # type: ignore[no-untyped-def]
            q = captured[0].tool_call_events
            q.put(ToolCallRecord(kind="call", tool_call_id="tc1", name="word_counter", args="{}"))
            q.put(ToolCallRecord(kind="result", tool_call_id="tc1", content="1"))
            yield _text_chunk("after", "Writer")

        @property
        def result(self) -> CrewOutput:  # type: ignore[override]
            return CrewOutput(raw="ignored")

    agent = AgentForTest(api_base="https://x/", api_key="k", verbose=False)
    agent._crew_for_test = CrewForTest(_Stream())

    with patch(
        "datarobot_genai.crewai.agent.CrewAIStreamingEventListener", _CapturingStreamingListener
    ):
        events = [e async for (e, _, _) in agent.invoke(run_agent_input)]

    validate_sequence(events)
    starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
    assert len(starts) == 1
    assert starts[0].parent_message_id == ""  # no bubble was open to attach to


async def test_invoke_streaming_tool_call_nests_under_its_own_agent_step(
    mock_ragas_event_listener, run_agent_input
) -> None:
    # GIVEN the native flow: each agent's tool is queued BEFORE that agent emits any text
    # chunk (native tool calls produce no content chunk). The record carries its owning agent's
    # role, so the drain must open that agent's step before emitting the call.
    from ag_ui.core import ToolCallStartEvent

    from datarobot_genai.crewai.streaming_events import ToolCallRecord

    captured: list[Any] = []

    class _CapturingStreamingListener:
        def __init__(self) -> None:
            self.reasoning_event = False
            self.active_agent_role = ""
            self.tool_call_events: queue.Queue[Any] = queue.Queue()
            captured.append(self)

        def setup_listeners(self, _bus: Any) -> None:
            pass

    class _Stream(CrewStreamingOutput):
        def __init__(self) -> None:
            super().__init__(async_iterator=self._iter())

        @staticmethod
        async def _iter():  # type: ignore[no-untyped-def]
            q = captured[0].tool_call_events
            # Planner's tool fires before Planner's text chunk.
            q.put(
                ToolCallRecord(
                    kind="call", tool_call_id="p", name="search", args="{}", agent_role="Planner"
                )
            )
            q.put(ToolCallRecord(kind="result", tool_call_id="p", content="ok"))
            yield _text_chunk("plan answer", "Planner")
            # Writer's tool fires before Writer's text chunk (while Planner's step is open).
            q.put(
                ToolCallRecord(
                    kind="call", tool_call_id="w", name="search", args="{}", agent_role="Writer"
                )
            )
            q.put(ToolCallRecord(kind="result", tool_call_id="w", content="ok"))
            yield _text_chunk("write answer", "Writer")

        @property
        def result(self) -> CrewOutput:  # type: ignore[override]
            return CrewOutput(raw="ignored")

    agent = AgentForTest(api_base="https://x/", api_key="k", verbose=False)
    agent._crew_for_test = CrewForTest(_Stream())

    with patch(
        "datarobot_genai.crewai.agent.CrewAIStreamingEventListener", _CapturingStreamingListener
    ):
        events = [e async for (e, _, _) in agent.invoke(run_agent_input)]

    # THEN each tool call nests under its OWN agent's step (not the previous agent's).
    validate_sequence(events)

    def _between(tc_idx: int, role: str) -> bool:
        start = next(
            i
            for i, e in enumerate(events)
            if isinstance(e, StepStartedEvent) and e.step_name == role
        )
        finish = next(
            i
            for i, e in enumerate(events)
            if isinstance(e, StepFinishedEvent) and e.step_name == role
        )
        return start < tc_idx < finish

    tool_idxs = [i for i, e in enumerate(events) if isinstance(e, ToolCallStartEvent)]
    assert len(tool_idxs) == 2
    assert _between(tool_idxs[0], "Planner")  # Planner's tool inside the Planner step
    assert _between(tool_idxs[1], "Writer")  # Writer's tool inside the Writer step, not Planner's
    assert [e.step_name for e in events if isinstance(e, StepStartedEvent)] == ["Planner", "Writer"]


async def test_invoke_streaming_closes_reasoning_before_tool_call(
    mock_ragas_event_listener, run_agent_input
) -> None:
    # GIVEN reasoning is open when a tool fires: REASONING_END must precede TOOL_CALL_START.

    from datarobot_genai.crewai.streaming_events import ToolCallRecord

    captured: list[Any] = []

    class _CapturingStreamingListener:
        def __init__(self) -> None:
            self.reasoning_event = False
            self.active_agent_role = ""
            self.tool_call_events: queue.Queue[Any] = queue.Queue()
            captured.append(self)

        def setup_listeners(self, _bus: Any) -> None:
            pass

    class _Stream(CrewStreamingOutput):
        def __init__(self) -> None:
            super().__init__(async_iterator=self._iter())

        @staticmethod
        async def _iter():  # type: ignore[no-untyped-def]
            captured[0].reasoning_event = True
            yield _text_chunk("thinking", "Writer")  # opens a reasoning message
            q = captured[0].tool_call_events
            q.put(ToolCallRecord(kind="call", tool_call_id="tc1", name="word_counter", args="{}"))
            q.put(ToolCallRecord(kind="result", tool_call_id="tc1", content="1"))
            captured[0].reasoning_event = False
            yield _text_chunk("answer", "Writer")

        @property
        def result(self) -> CrewOutput:  # type: ignore[override]
            return CrewOutput(raw="ignored")

    agent = AgentForTest(api_base="https://x/", api_key="k", verbose=False)
    agent._crew_for_test = CrewForTest(_Stream())

    with patch(
        "datarobot_genai.crewai.agent.CrewAIStreamingEventListener", _CapturingStreamingListener
    ):
        events = [e async for (e, _, _) in agent.invoke(run_agent_input)]

    validate_sequence(events)
    type_names = [type(e).__name__ for e in events]
    assert "ReasoningEndEvent" in type_names
    assert type_names.index("ReasoningEndEvent") < type_names.index("ToolCallStartEvent")


async def test_invoke_drains_the_real_listeners_queue(
    mock_ragas_event_listener, run_agent_input
) -> None:
    # GIVEN the REAL CrewAIStreamingEventListener instance (not the hand-mirrored stub): invoke
    # must drain ITS `tool_call_events` queue into AG-UI ToolCall* events. This closes the seam
    # the stub tests leave open -- a rename of the listener's attributes (tool_call_events /
    # active_agent_role / reasoning_event) would break invoke and fail here. Records are fed into
    # the real queue directly (the bus->queue path is covered in test_streaming_events.py); we do
    # NOT emit on the global bus, which would pollute its scope-stack contextvar across tests.
    from ag_ui.core import ToolCallResultEvent
    from ag_ui.core import ToolCallStartEvent

    from datarobot_genai.crewai.streaming_events import CrewAIStreamingEventListener
    from datarobot_genai.crewai.streaming_events import ToolCallRecord

    real_listener = CrewAIStreamingEventListener()

    class _Stream(CrewStreamingOutput):
        def __init__(self) -> None:
            super().__init__(async_iterator=self._iter())

        @staticmethod
        async def _iter():  # type: ignore[no-untyped-def]
            # Empty chunk role -> the step role is read from the real listener's active_agent_role,
            # so a rename of that attribute would break invoke and fail this test.
            real_listener.active_agent_role = "Writer"
            yield _text_chunk("working", "")
            real_listener.tool_call_events.put(
                ToolCallRecord(
                    kind="call", tool_call_id="tc1", name="word_counter", args='{"text": "a b"}'
                )
            )
            real_listener.tool_call_events.put(
                ToolCallRecord(kind="result", tool_call_id="tc1", content="2")
            )
            yield _text_chunk("done", "")

        @property
        def result(self) -> CrewOutput:  # type: ignore[override]
            return CrewOutput(raw="ignored")

    agent = AgentForTest(api_base="https://x/", api_key="k", verbose=False)
    agent._crew_for_test = CrewForTest(_Stream())

    with patch(
        "datarobot_genai.crewai.agent.CrewAIStreamingEventListener",
        lambda *a, **k: real_listener,
    ):
        events = [e async for (e, _, _) in agent.invoke(run_agent_input)]

    validate_sequence(events)
    starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
    results = [e for e in events if isinstance(e, ToolCallResultEvent)]
    assert len(starts) == 1
    assert len(results) == 1
    assert starts[0].tool_call_name == "word_counter"
    assert results[0].content == "2"
    assert starts[0].tool_call_id == results[0].tool_call_id


async def test_invoke_streaming_post_loop_tool_drain_has_empty_parent(
    mock_ragas_event_listener, run_agent_input
) -> None:
    # GIVEN a tool whose events arrive after the final chunk with no bubble open: the after-loop
    # drain (site 3) must emit it with parent_message_id="" -- guards the post-loop occurrence of
    # the parent fix, which the other tests (queueing before a trailing chunk) don't reach.
    from ag_ui.core import ToolCallStartEvent

    from datarobot_genai.crewai.streaming_events import ToolCallRecord

    captured: list[Any] = []

    class _CapturingStreamingListener:
        def __init__(self) -> None:
            self.reasoning_event = False
            self.active_agent_role = ""
            self.tool_call_events: queue.Queue[Any] = queue.Queue()
            captured.append(self)

        def setup_listeners(self, _bus: Any) -> None:
            pass

    class _Stream(CrewStreamingOutput):
        def __init__(self) -> None:
            super().__init__(async_iterator=self._iter())

        @staticmethod
        async def _iter():  # type: ignore[no-untyped-def]
            q = captured[0].tool_call_events
            q.put(ToolCallRecord(kind="call", tool_call_id="tc1", name="word_counter", args="{}"))
            q.put(ToolCallRecord(kind="result", tool_call_id="tc1", content="1"))
            for _ in ():  # async generator that yields no chunks -> drain happens post-loop
                yield

        @property
        def result(self) -> CrewOutput:  # type: ignore[override]
            return CrewOutput(raw="ignored")

    agent = AgentForTest(api_base="https://x/", api_key="k", verbose=False)
    agent._crew_for_test = CrewForTest(_Stream())

    with patch(
        "datarobot_genai.crewai.agent.CrewAIStreamingEventListener", _CapturingStreamingListener
    ):
        events = [e async for (e, _, _) in agent.invoke(run_agent_input)]

    validate_sequence(events)
    starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
    assert len(starts) == 1
    assert starts[0].parent_message_id == ""  # drained post-loop, no bubble was open
