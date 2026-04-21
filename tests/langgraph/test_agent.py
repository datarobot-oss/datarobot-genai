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
from functools import cached_property
from typing import Any
from unittest.mock import Mock

import pytest
from ag_ui.core import AssistantMessage
from ag_ui.core import BaseEvent
from ag_ui.core import EventType
from ag_ui.core import RunAgentInput
from ag_ui.core import SystemMessage as AgSystemMessage
from ag_ui.core import UserMessage
from langchain_core.messages import AIMessage
from langchain_core.messages import AIMessageChunk
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import MessagesState
from langgraph.graph.state import Command
from langgraph.graph.state import StateGraph
from langgraph.types import Interrupt

from datarobot_genai.core.memory.base import BaseMemoryClient
from datarobot_genai.langgraph.agent import LANGGRAPH_RESUME_STATE_KEY
from datarobot_genai.langgraph.agent import LangGraphAgent
from datarobot_genai.langgraph.agent import datarobot_agent_class_from_langgraph


@pytest.fixture
def authorization_context() -> dict[str, Any]:
    return {"user": {"id": "123", "name": "bar"}}


@pytest.fixture
def run_agent_input() -> RunAgentInput:
    return RunAgentInput(
        messages=[UserMessage(content='{"topic": "AI"}', id="message_id")],
        tools=[],
        forwarded_props=dict(model="m", authorization_context={}, forwarded_headers={}),
        thread_id="thread_id",
        run_id="run_id",
        state={},
        context=[],
    )


class SimpleLangGraphAgent(LangGraphAgent):
    @cached_property
    def workflow(self) -> StateGraph[MessagesState]:
        async def mock_stream_generator():
            # stream the first agent
            # tool call and respose
            yield (
                "first_agent",
                "messages",
                (
                    AIMessageChunk(
                        content="",
                        id="000",
                        tool_call_chunks=[
                            {"name": "get_info_about_city", "id": "tool_call_111", "args": None}
                        ],
                    ),
                    {},
                ),
            )
            yield (
                "first_agent",
                "messages",
                (
                    AIMessageChunk(
                        content="",
                        id="000",
                        tool_call_chunks=[{"name": "", "id": "", "args": "{'name': 'Paris'}"}],
                    ),
                    {},
                ),
            )
            yield (
                "first_agent",
                "messages",
                (
                    ToolMessage(
                        tool_call_id="tool_call_111",
                        id="000",
                        content="Paris is the capital city of France.",
                    ),
                    {},
                ),
            )
            # tool call end
            yield (
                "first_agent",
                "messages",
                (AIMessageChunk(content="Here is the information", id="111"), {}),
            )
            yield (
                "first_agent",
                "messages",
                (AIMessageChunk(content=" you requested about Paris.....", id="111"), {}),
            )

            yield (
                "first_agent",
                "updates",
                {
                    "first_agent": {
                        "usage": {
                            "total_tokens": 100,
                            "prompt_tokens": 50,
                            "completion_tokens": 50,
                        },
                        "messages": [
                            HumanMessage(content="Hi, tell me about Paris."),
                            AIMessage(
                                content="Here is the information you requested about Paris.....",
                                id="111",
                            ),
                        ],
                    }
                },
            )
            yield (
                "final_agent",
                "messages",
                (AIMessageChunk(content="Paris is the capital", id="222"), {}),
            )
            yield (
                "final_agent",
                "messages",
                (AIMessageChunk(content=" city of France.", id="222"), {}),
            )
            yield (
                "final_agent",
                "updates",
                {
                    "final_agent": {
                        "usage": {
                            "total_tokens": 100,
                            "prompt_tokens": 50,
                            "completion_tokens": 50,
                        },
                        "messages": [
                            HumanMessage(content="Hi, tell me about Paris."),
                            AIMessage(content="Paris is the capital city of France.", id="222"),
                        ],
                    }
                },
            )

        mock_graph_stream = Mock(astream=Mock(return_value=mock_stream_generator()))

        mock_state_graph = Mock(compile=Mock(return_value=mock_graph_stream))
        return mock_state_graph

    @property
    def prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Tell user about {topic}.",
                },
                {"role": "user", "content": "Hi, tell me about {topic}."},
            ]
        )

    @property
    def langgraph_config(self) -> dict[str, Any]:
        return {}


def test_datarobot_agent_class_from_langgraph_factory_receives_llm_tools_verbose() -> None:
    """graph_factory(llm, tools, verbose) is called when building workflow (each access)."""
    inner = SimpleLangGraphAgent()
    mock_graph = inner.workflow
    calls: list[tuple[Any, list[Any], bool]] = []

    def graph_factory(llm: Any, tools: list[Any], verbose: bool) -> Any:
        calls.append((llm, list(tools), verbose))
        return mock_graph

    pt = inner.prompt_template
    cls = datarobot_agent_class_from_langgraph(graph_factory, pt)
    mock_llm = Mock()
    extra_tools = [Mock()]
    agent = cls(
        llm=mock_llm,
        tools=extra_tools,
        verbose=True,
        api_key="k",
        api_base="https://x/",
    )
    _ = agent.workflow
    assert calls == [(mock_llm, extra_tools, True)]
    assert agent.prompt_template is pt

    _ = agent.workflow
    assert len(calls) == 2
    assert calls[1] == (mock_llm, extra_tools, True)


@pytest.mark.asyncio
async def test_datarobot_agent_class_from_langgraph_invoke_streams(
    run_agent_input: RunAgentInput,
) -> None:
    """Factory-built agent runs the same invoke/astream path as a hand-written subclass."""
    inner = SimpleLangGraphAgent()
    mock_graph = inner.workflow

    def graph_factory(llm: Any, tools: list[Any], verbose: bool) -> Any:
        return mock_graph

    cls = datarobot_agent_class_from_langgraph(graph_factory, inner.prompt_template)
    agent = cls(llm=Mock(), tools=[], verbose=True, api_key="k", api_base="https://x/")

    events = [e async for e in agent.invoke(run_agent_input)]
    assert events
    assert events[-1][0].type == EventType.RUN_FINISHED
    mock_graph.compile.assert_called()
    mock_graph.compile.return_value.astream.assert_called_once()


class HistoryAwareLangGraphAgent(LangGraphAgent):
    """LangGraph agent whose prompt template exposes {chat_history}."""

    @cached_property
    def workflow(self) -> StateGraph[MessagesState]:
        # Reuse the same mock workflow behaviour as SimpleLangGraphAgent; tests
        # that rely on streaming/invoke semantics use the original class.
        return SimpleLangGraphAgent().workflow

    @property
    def prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                {
                    "role": "system",
                    "content": "History transcript:\n{chat_history}",
                },
                {
                    "role": "user",
                    "content": "Latest request: {topic}",
                },
            ]
        )

    @property
    def langgraph_config(self) -> dict[str, Any]:
        return {}


class LangGraphAgentWithMemory(SimpleLangGraphAgent):
    @property
    def prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                {
                    "role": "system",
                    "content": "Relevant memory:\n{memory}",
                },
                {"role": "user", "content": "Hi, tell me about {topic}."},
            ]
        )


class LegacyOverrideLangGraphAgent(SimpleLangGraphAgent):
    async def convert_input_message(self, run_agent_input: RunAgentInput) -> Command:
        return await super().convert_input_message(run_agent_input)


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


async def test_convert_input_message_includes_history() -> None:
    """convert_input_message should include history when template uses {chat_history}."""
    agent = HistoryAwareLangGraphAgent()
    run_agent_input = RunAgentInput(
        messages=[
            AgSystemMessage(id="sys_1", content="You are a helper."),
            UserMessage(id="user_1", content='{"topic": "First question"}'),
            AssistantMessage(id="asst_1", content="First answer"),
            UserMessage(id="user_2", content='{"topic": "Follow-up"}'),
        ],
        tools=[],
        forwarded_props=dict(model="m", authorization_context={}, forwarded_headers={}),
        thread_id="thread_id",
        run_id="run_id",
        state={},
        context=[],
    )

    command = await agent.convert_input_message(run_agent_input)
    all_messages = command.update["messages"]

    # History is embedded as string in the template, so we expect 2 messages:
    # - System message with history transcript + user message
    assert len(all_messages) == 2
    assert isinstance(all_messages[0], SystemMessage)
    assert isinstance(all_messages[1], HumanMessage)

    # Verify history is embedded in the system message
    system_content = str(all_messages[0].content)
    assert "History transcript:" in system_content
    assert "system: You are a helper." in system_content
    assert "First question" in system_content
    assert "First answer" in system_content


async def test_convert_input_message_truncates_history() -> None:
    """History is truncated to max_history_messages prior turns."""
    agent = HistoryAwareLangGraphAgent()
    max_history = agent.max_history_messages

    # Create more messages than max_history_messages, all user role so that
    # everything before the final one is treated as history.
    messages = [
        UserMessage(id=f"user_{i}", content=f'{{"topic": "Topic {i}"}}')
        for i in range(max_history + 10)
    ]
    run_agent_input = RunAgentInput(
        messages=messages,
        tools=[],
        forwarded_props=dict(model="m", authorization_context={}, forwarded_headers={}),
        thread_id="thread_id",
        run_id="run_id",
        state={},
        context=[],
    )

    command = await agent.convert_input_message(run_agent_input)
    all_messages = command.update["messages"]

    # We expect 2 templated messages (system + user)
    assert len(all_messages) == 2

    # Verify history in system message is truncated
    system_content = str(all_messages[0].content)
    # The earliest messages should not appear (truncated)
    # Use exact match with closing brace to avoid substring matches (e.g., "Topic 1" in "Topic 10")
    assert '"topic": "Topic 0"' not in system_content
    # The last message (Topic 29) should be excluded from history
    assert '"topic": "Topic 29"' not in system_content
    # Messages within the history window should appear
    assert '"topic": "Topic 9"' in system_content


async def test_convert_input_message_injects_chat_history_variable() -> None:
    """convert_input_message should provide {chat_history} to the prompt template."""
    agent = HistoryAwareLangGraphAgent()
    run_agent_input = RunAgentInput(
        messages=[
            AgSystemMessage(id="sys_1", content="You are a helper."),
            UserMessage(id="user_1", content='{"topic": "First question"}'),
            AssistantMessage(id="asst_1", content="First answer"),
            UserMessage(id="user_2", content='{"topic": "Follow-up"}'),
        ],
        tools=[],
        forwarded_props=dict(model="m", authorization_context={}, forwarded_headers={}),
        thread_id="thread_id",
        run_id="run_id",
        state={},
        context=[],
    )

    command = await agent.convert_input_message(run_agent_input)
    all_messages = command.update["messages"]

    # Find the system message generated from the prompt template and assert that
    # it contains a rendered history transcript (prior turns only).
    history_msgs = [
        m
        for m in all_messages
        if isinstance(m, SystemMessage) and "History transcript:" in str(m.content)
    ]
    assert len(history_msgs) == 1
    history_text = str(history_msgs[0].content)

    assert "system: You are a helper." in history_text
    assert 'user: {"topic": "First question"}' in history_text
    assert "assistant: First answer" in history_text
    # The latest user turn should not appear inside the history transcript.
    assert "Follow-up" not in history_text


async def test_convert_input_message_zero_history_disables_history() -> None:
    """When max_history_messages is 0, no prior turns are included."""
    agent = SimpleLangGraphAgent(max_history_messages=0)

    run_agent_input = RunAgentInput(
        messages=[
            AgSystemMessage(id="sys_1", content="You are a helper."),
            UserMessage(id="user_1", content='{"topic": "First question"}'),
            AssistantMessage(id="asst_1", content="First answer"),
            UserMessage(id="user_2", content='{"topic": "Follow-up"}'),
        ],
        tools=[],
        forwarded_props=dict(model="m", authorization_context={}, forwarded_headers={}),
        thread_id="thread_id",
        run_id="run_id",
        state={},
        context=[],
    )

    command = await agent.convert_input_message(run_agent_input)
    all_messages = command.update["messages"]

    # Only the templated messages for the final user turn should remain.
    assert len(all_messages) == 2
    assert isinstance(all_messages[0], SystemMessage)
    assert isinstance(all_messages[1], HumanMessage)


def test_langgraph_default_checkpointer_is_in_memory_saver() -> None:
    agent = SimpleLangGraphAgent()
    assert isinstance(agent.langgraph_checkpointer, InMemorySaver)


async def test_langgraph_non_streaming(run_agent_input):
    # GIVEN a simple langgraph agent implementation
    agent = SimpleLangGraphAgent()

    # WHEN invoking the agent
    streaming_response_iterator = agent.invoke(run_agent_input)

    # THEN the streaming response iterator returns the expected responses
    # Iterate directly over the async generator to avoid event loop conflicts
    # Note: With the new async with implementation, _invoke is called when we start consuming
    first_item_consumed = False
    events = []
    async for (
        response_event,
        pipeline_interactions,
        usage_metrics,
    ) in streaming_response_iterator:
        # Check that agent.workflow is called with expected arguments after first consumption
        if not first_item_consumed:
            expected_command = Command(
                update={
                    "messages": [
                        SystemMessage(content="You are a helpful assistant. Tell user about AI."),
                        HumanMessage(content="Hi, tell me about AI."),
                    ]
                }
            )
            ckpt = agent.langgraph_checkpointer
            assert agent.workflow.compile.call_args_list[0].kwargs["checkpointer"] is ckpt
            agent.workflow.compile().astream.assert_called_once_with(
                input=expected_command,
                config={"configurable": {"thread_id": "thread_id"}},
                debug=True,
                stream_mode=["updates", "messages"],
                subgraphs=True,
            )
            first_item_consumed = True

        assert not isinstance(response_event, str) or pipeline_interactions is not None

        if isinstance(response_event, BaseEvent):
            events.append(response_event)

    assert len(events) == 14
    assert events[0].type == EventType.RUN_STARTED
    assert events[1].type == EventType.TOOL_CALL_START
    assert events[1].tool_call_id == "tool_call_111"
    assert events[1].tool_call_name == "get_info_about_city"
    assert events[1].parent_message_id == "000"
    assert events[2].type == EventType.TOOL_CALL_ARGS
    assert events[2].tool_call_id == "tool_call_111"
    assert events[2].delta == "{'name': 'Paris'}"
    assert events[3].type == EventType.TOOL_CALL_END
    assert events[3].tool_call_id == "tool_call_111"
    assert events[4].type == EventType.TOOL_CALL_RESULT
    assert events[4].tool_call_id == "tool_call_111"
    assert events[4].content == "Paris is the capital city of France."
    assert events[4].role == "tool"
    assert events[5].type == EventType.TEXT_MESSAGE_START
    assert events[5].message_id == "111"
    assert events[6].type == EventType.TEXT_MESSAGE_CONTENT
    assert events[6].delta == "Here is the information"
    assert events[6].message_id == "111"
    assert events[7].type == EventType.TEXT_MESSAGE_CONTENT
    assert events[7].delta == " you requested about Paris....."
    assert events[7].message_id == "111"
    assert events[8].type == EventType.TEXT_MESSAGE_END
    assert events[8].message_id == "111"
    assert events[9].type == EventType.TEXT_MESSAGE_START
    assert events[9].message_id == "222"
    assert events[10].type == EventType.TEXT_MESSAGE_CONTENT
    assert events[10].delta == "Paris is the capital"
    assert events[10].message_id == "222"
    assert events[11].type == EventType.TEXT_MESSAGE_CONTENT
    assert events[11].delta == " city of France."
    assert events[11].message_id == "222"
    assert events[12].type == EventType.TEXT_MESSAGE_END
    assert events[12].message_id == "222"
    assert events[13].type == EventType.RUN_FINISHED

    assert pipeline_interactions is not None
    assert usage_metrics is not None
    assert usage_metrics["total_tokens"] == 200
    assert usage_metrics["prompt_tokens"] == 100
    assert usage_metrics["completion_tokens"] == 100


async def test_langgraph_invoke_retrieves_and_stores_memory(run_agent_input) -> None:
    # GIVEN a langgraph agent whose prompt opts into memory
    memory_client = FakeMemoryClient(retrieved="Use concise answers.")
    agent = LangGraphAgentWithMemory(memory_client=memory_client)

    # WHEN invoking the agent
    _ = [event async for event in agent.invoke(run_agent_input)]

    # THEN memory retrieval is scoped to the thread and storage is scoped to the run
    assert memory_client.retrieve_calls == [
        {
            "prompt": '{"topic": "AI"}',
            "run_id": None,
            "agent_id": "LangGraphAgentWithMemory",
            "app_id": "tests.langgraph.test_agent",
            "attributes": {"thread_id": "thread_id"},
        }
    ]
    assert memory_client.store_calls == [
        {
            "user_message": '{"topic": "AI"}',
            "run_id": "run_id",
            "agent_id": "LangGraphAgentWithMemory",
            "app_id": "tests.langgraph.test_agent",
            "attributes": {"thread_id": "thread_id"},
        }
    ]


async def test_langgraph_invoke_skips_memory_retrieval_when_prompt_does_not_use_it(
    run_agent_input,
) -> None:
    # GIVEN a langgraph agent whose prompt does not declare {memory}
    memory_client = FakeMemoryClient(retrieved="Use concise answers.")
    agent = SimpleLangGraphAgent(memory_client=memory_client)

    # WHEN invoking the agent
    _ = [event async for event in agent.invoke(run_agent_input)]

    # THEN retrieval is skipped but the completed run is still stored
    assert memory_client.retrieve_calls == []
    assert memory_client.store_calls == []


async def test_langgraph_invoke_gracefully_degrades_when_memory_fails(run_agent_input) -> None:
    # GIVEN a langgraph agent whose memory provider errors at runtime
    agent = LangGraphAgentWithMemory(memory_client=FailingMemoryClient())

    # WHEN invoking the agent
    events = [event async for event in agent.invoke(run_agent_input)]

    # THEN the agent still completes successfully without failing the request
    assert events
    assert events[-1][0].type == EventType.RUN_FINISHED


async def test_langgraph_invoke_supports_legacy_convert_input_override(run_agent_input) -> None:
    # GIVEN a subclass overriding convert_input_message with the original signature
    agent = LegacyOverrideLangGraphAgent(
        memory_client=FakeMemoryClient(retrieved="Use concise answers.")
    )

    # WHEN invoking the agent
    events = [event async for event in agent.invoke(run_agent_input)]

    # THEN the override still works and the run completes
    assert events
    assert events[-1][0].type == EventType.RUN_FINISHED


async def test_build_input_command_explicit_resume_returns_command_resume() -> None:
    """HITL continuation: state contains langgraph_resume -> Command(resume=...)."""
    agent = HistoryAwareLangGraphAgent()
    run_agent_input = RunAgentInput(
        messages=[UserMessage(content='{"topic": "x"}', id="m")],
        tools=[],
        forwarded_props={},
        thread_id="thread_id",
        run_id="run_id",
        state={LANGGRAPH_RESUME_STATE_KEY: "human approved"},
        context=[],
    )
    command = await agent._build_input_command(run_agent_input, Mock())
    assert command.resume == "human approved"


class InterruptStreamAgent(SimpleLangGraphAgent):
    """Mock graph that only emits a LangGraph __interrupt__ update."""

    @cached_property
    def workflow(self) -> StateGraph[MessagesState]:
        async def mock_stream():
            yield (
                (),
                "updates",
                {"__interrupt__": (Interrupt(value={"question": "ok?"}, id="intr-1"),)},
            )

        mock_graph_stream = Mock(astream=Mock(return_value=mock_stream()))
        mock_state_graph = Mock(compile=Mock(return_value=mock_graph_stream))
        return mock_state_graph


@pytest.mark.asyncio
async def test_invoke_emits_interrupt_custom_event(run_agent_input: RunAgentInput) -> None:
    agent = InterruptStreamAgent()
    events = [e async for e in agent.invoke(run_agent_input)]
    custom_events = [e[0] for e in events if e[0].type == EventType.CUSTOM]
    assert len(custom_events) == 1
    assert custom_events[0].name == "langgraph.interrupt"
    assert custom_events[0].value["interrupts"][0]["id"] == "intr-1"
    finished = events[-1][0]
    assert finished.type == EventType.RUN_FINISHED
    assert finished.result is not None
    assert finished.result["langgraph"]["interrupted"] is True


async def test_langgraph_does_not_store_memory_when_run_fails(run_agent_input) -> None:
    # GIVEN an agent whose stream fails before completion
    class FailingWorkflowAgent(SimpleLangGraphAgent):
        @cached_property
        def workflow(self) -> StateGraph[MessagesState]:
            async def failing_stream():
                raise RuntimeError("graph failed")
                yield  # pragma: no cover

            return Mock(
                compile=Mock(return_value=Mock(astream=Mock(return_value=failing_stream())))
            )

    memory_client = FakeMemoryClient(retrieved="Use concise answers.")
    agent = FailingWorkflowAgent(memory_client=memory_client)

    # WHEN invoking the agent and the graph fails
    with pytest.raises(RuntimeError, match="graph failed"):
        _ = [event async for event in agent.invoke(run_agent_input)]

    # THEN the user message is not persisted because the run did not finish
    assert memory_client.store_calls == []


def test_create_pipeline_interactions_from_events_filters_tool_messages() -> None:
    # None returns None
    assert LangGraphAgent.create_pipeline_interactions_from_events(None) is None

    # Prepare events structure expected by the function using real message classes
    t1 = ToolMessage(content="tool", tool_call_id="tc_1")
    human = HumanMessage(content="hi")
    ai = AIMessage(content="ok")
    events: list[dict[str, Any]] = [
        {
            "node1": {
                "messages": [t1, human],
            }
        },
        {"node2": {"messages": [ai]}},
    ]

    sample = LangGraphAgent.create_pipeline_interactions_from_events(events)
    assert sample is not None
    # ToolMessage filtered out; order preserved
    msgs = sample.user_input
    assert len(msgs) == 2
