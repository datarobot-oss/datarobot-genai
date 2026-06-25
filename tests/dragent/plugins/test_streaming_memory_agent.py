# Copyright 2026 DataRobot, Inc. and its affiliates.
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

import logging
import os
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from ag_ui.core import AssistantMessage
from ag_ui.core import RunAgentInput
from ag_ui.core import SystemMessage as AgUiSystemMessage
from ag_ui.core import UserMessage as AgUiUserMessage
from ag_ui.core.events import EventType
from ag_ui.core.events import TextMessageContentEvent
from nat.data_models.agent import AgentBaseConfig
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.component_ref import MemoryRef
from nat.memory.models import MemoryItem
from nat.plugins.langchain.agent.auto_memory_wrapper.register import AutoMemoryAgentConfig

from datarobot_genai.dragent.frontends.response import DRAgentEventResponse
from datarobot_genai.dragent.plugins.datarobot_mem0_memory import UnconfiguredMemoryEditor
from datarobot_genai.dragent.plugins.streaming_memory_agent import StreamingMemoryAgentConfig
from datarobot_genai.dragent.plugins.streaming_memory_agent import _last_user_text
from datarobot_genai.dragent.plugins.streaming_memory_agent import _user_id_from_context
from datarobot_genai.dragent.plugins.streaming_memory_agent import _with_memory_context
from datarobot_genai.dragent.plugins.streaming_memory_agent import streaming_memory_agent

_MODULE = "datarobot_genai.dragent.plugins.streaming_memory_agent"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user(content: str, msg_id: str | None = None) -> AgUiUserMessage:
    return AgUiUserMessage(id=msg_id or f"u-{content}", content=content)


def _assistant(content: str, msg_id: str | None = None) -> AssistantMessage:
    return AssistantMessage(id=msg_id or f"a-{content}", content=content)


def _system(content: str, msg_id: str | None = None) -> AgUiSystemMessage:
    return AgUiSystemMessage(id=msg_id or f"s-{content}", content=content)


def _input(*messages) -> RunAgentInput:
    """Build a minimal RunAgentInput around a message list."""
    return RunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        state={},
        messages=list(messages),
        tools=[],
        context=[],
        forwarded_props={},
    )


def _chunk(content: str, msg_id: str = "m-1") -> DRAgentEventResponse:
    """Build a DRAgentEventResponse with a single text-content event."""
    return DRAgentEventResponse(
        events=[
            TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT,
                message_id=msg_id,
                delta=content,
            )
        ],
    )


def _make_config(memory_name: str = "mem0", **overrides) -> StreamingMemoryAgentConfig:
    """Build a config with sensible defaults for tests."""
    defaults: dict = {
        "llm_name": LLMRef("test-llm"),
        "description": "test streaming memory agent",
        "inner_agent_name": FunctionRef("inner-agent"),
        "memory_name": MemoryRef(memory_name),
    }
    defaults.update(overrides)
    return StreamingMemoryAgentConfig(**defaults)


def _make_builder(
    memory_editor: AsyncMock | None = None,
    inner_agent_chunks: list[DRAgentEventResponse] | None = None,
) -> MagicMock:
    """Build a NAT Builder mock with the methods the plugin calls."""
    builder = MagicMock()
    builder.get_memory_client = AsyncMock(return_value=memory_editor)

    inner_agent = MagicMock()

    async def _astream(_inner_request, to_type=None):  # noqa: ARG001
        for chunk in inner_agent_chunks or []:
            yield chunk

    inner_agent.astream = _astream
    # Stash so tests can override or inspect.
    inner_agent._test_astream = _astream  # type: ignore[attr-defined]
    builder.get_function = AsyncMock(return_value=inner_agent)
    return builder


async def _drain(gen: AsyncGenerator) -> list:
    return [x async for x in gen]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestStreamingMemoryAgentConfig:
    def test_disables_mem0_posthog_telemetry_on_import(self):
        assert os.environ.get("MEM0_TELEMETRY") == "False"

    def test_is_subclass_of_auto_memory_agent_config(self):
        # Inheriting from upstream lets workflows swap _type between the two
        # wrappers without restating any other field.
        assert issubclass(StreamingMemoryAgentConfig, AutoMemoryAgentConfig)
        # Transitively still an AgentBaseConfig.
        assert issubclass(StreamingMemoryAgentConfig, AgentBaseConfig)

    def test_registered_name(self):
        assert StreamingMemoryAgentConfig.static_type() == "streaming_memory_agent"

    def test_only_discriminator_differs_from_upstream(self):
        # The whole point of inheriting is field parity with auto_memory_agent;
        # if a new field appears in only one of the two classes, this test
        # forces an explicit decision.
        parent_fields = set(AutoMemoryAgentConfig.model_fields) - {"type"}
        child_fields = set(StreamingMemoryAgentConfig.model_fields) - {"type"}
        assert parent_fields == child_fields

    def test_defaults(self):
        config = _make_config()
        assert config.inner_agent_name == "inner-agent"
        assert config.memory_name == "mem0"
        assert config.save_user_messages_to_memory is True
        assert config.retrieve_memory_for_every_response is True
        assert config.save_ai_messages_to_memory is True
        assert config.search_params == {}
        assert config.add_params == {}

    def test_inner_agent_name_required(self):
        with pytest.raises(Exception):
            StreamingMemoryAgentConfig(
                llm_name=LLMRef("x"),
                description="x",
                memory_name=MemoryRef("mem0"),
            )  # type: ignore[call-arg]

    def test_memory_name_required(self):
        with pytest.raises(Exception):
            StreamingMemoryAgentConfig(
                llm_name=LLMRef("x"),
                description="x",
                inner_agent_name=FunctionRef("inner-agent"),
            )  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# _user_id_from_context
# ---------------------------------------------------------------------------


class TestUserIdFromContext:
    def test_prefers_user_manager_get_id(self):
        with patch(f"{_MODULE}.Context") as mock_ctx:
            ctx = mock_ctx.get.return_value
            ctx.user_manager.get_id.return_value = "uid-from-manager"
            ctx.metadata.headers = {"x-user-id": "uid-from-header"}
            assert _user_id_from_context() == "uid-from-manager"

    def test_falls_back_to_header_when_no_user_manager(self):
        with patch(f"{_MODULE}.Context") as mock_ctx:
            ctx = mock_ctx.get.return_value
            ctx.user_manager = None
            ctx.metadata.headers = {"x-user-id": "uid-from-header"}
            assert _user_id_from_context() == "uid-from-header"

    def test_falls_back_to_header_when_user_manager_returns_empty(self):
        with patch(f"{_MODULE}.Context") as mock_ctx:
            ctx = mock_ctx.get.return_value
            ctx.user_manager.get_id.return_value = ""
            ctx.metadata.headers = {"x-user-id": "uid-from-header"}
            assert _user_id_from_context() == "uid-from-header"

    def test_falls_back_to_header_when_user_manager_raises(self, caplog):
        with patch(f"{_MODULE}.Context") as mock_ctx:
            ctx = mock_ctx.get.return_value
            ctx.user_manager.get_id.side_effect = RuntimeError("boom")
            ctx.metadata.headers = {"x-user-id": "uid-from-header"}
            with caplog.at_level(logging.DEBUG, logger=_MODULE):
                assert _user_id_from_context() == "uid-from-header"

    def test_default_user_when_nothing_resolves(self):
        with patch(f"{_MODULE}.Context") as mock_ctx:
            ctx = mock_ctx.get.return_value
            ctx.user_manager = None
            ctx.metadata.headers = {}
            assert _user_id_from_context() == "default_user"

    def test_default_user_when_no_metadata(self):
        with patch(f"{_MODULE}.Context") as mock_ctx:
            ctx = mock_ctx.get.return_value
            ctx.user_manager = None
            ctx.metadata = None
            assert _user_id_from_context() == "default_user"


# ---------------------------------------------------------------------------
# _last_user_text
# ---------------------------------------------------------------------------


class TestLastUserText:
    def test_returns_last_user_message(self):
        messages = [_user("first"), _assistant("reply"), _user("second")]
        assert _last_user_text(messages) == "second"

    def test_returns_empty_when_no_user_messages(self):
        assert _last_user_text([_assistant("reply"), _system("sys")]) == ""

    def test_returns_empty_for_empty_list(self):
        assert _last_user_text([]) == ""

    def test_skips_user_message_without_content(self):
        # content="" is falsy, so the function falls back to the next user message.
        messages = [_user("earlier"), _user("", msg_id="empty")]
        assert _last_user_text(messages) == "earlier"


# ---------------------------------------------------------------------------
# _with_memory_context
# ---------------------------------------------------------------------------


class TestWithMemoryContext:
    def test_inserts_system_before_last_user(self):
        messages = [_user("hi"), _assistant("hello"), _user("question")]
        out = _with_memory_context(messages, "remembered fact")

        assert len(out) == len(messages) + 1
        # Inserted system message is the one immediately before the final user message.
        assert isinstance(out[-1], AgUiUserMessage)
        assert out[-1].content == "question"
        assert isinstance(out[-2], AgUiSystemMessage)
        assert "Relevant context from memory:" in str(out[-2].content)
        assert "remembered fact" in str(out[-2].content)
        # Earlier messages are unchanged.
        assert out[0] is messages[0]
        assert out[1] is messages[1]

    def test_inserts_at_front_when_no_user_message(self):
        messages = [_assistant("hello"), _system("sys")]
        out = _with_memory_context(messages, "fact")

        assert isinstance(out[0], AgUiSystemMessage)
        assert "fact" in str(out[0].content)
        # Original messages follow, in order.
        assert out[1:] == messages

    def test_skips_user_message_with_empty_content(self):
        # Injection target must match `_last_user_text`'s choice: both functions
        # skip empty-content user messages so the system message lands next to
        # the user turn the memory search was actually keyed off of.
        messages = [_user("earlier"), _user("", msg_id="empty")]
        out = _with_memory_context(messages, "fact")

        # Inserted system message sits immediately before "earlier", not before "".
        assert len(out) == len(messages) + 1
        assert isinstance(out[0], AgUiSystemMessage)
        assert isinstance(out[1], AgUiUserMessage)
        assert out[1].content == "earlier"
        assert isinstance(out[2], AgUiUserMessage)
        assert out[2].content == ""

    def test_does_not_mutate_input_list(self):
        messages = [_user("hi")]
        _with_memory_context(messages, "fact")
        assert len(messages) == 1


# ---------------------------------------------------------------------------
# streaming_memory_agent — registered factory
# ---------------------------------------------------------------------------


@pytest.fixture
def context_user_id():
    """Patch Context to return a stable user id and avoid touching real state."""
    with patch(f"{_MODULE}.Context") as mock_ctx:
        ctx = mock_ctx.get.return_value
        ctx.user_manager.get_id.return_value = "test-user"
        ctx.metadata.headers = {}
        yield "test-user"


class TestStreamingMemoryAgentFactory:
    async def test_yields_function_info_with_stream_fn(self, context_user_id):
        config = _make_config()
        builder = _make_builder(memory_editor=AsyncMock())
        async with streaming_memory_agent(config, builder) as fn_info:
            assert fn_info is not None
            assert fn_info.stream_fn is not None
            assert fn_info.description == config.description

    async def test_yields_function_info_with_single_fn(self, context_user_id):
        # stream_to_single_fn is wired so `nat dragent run` (and other
        # non-streaming callers) can collapse the event stream to text.
        config = _make_config()
        builder = _make_builder(memory_editor=AsyncMock())
        async with streaming_memory_agent(config, builder) as fn_info:
            assert fn_info.single_fn is not None

    async def test_fetches_memory_client_and_inner_agent(self, context_user_id):
        config = _make_config(memory_name="mem0")
        memory_editor = AsyncMock()
        builder = _make_builder(memory_editor=memory_editor)
        async with streaming_memory_agent(config, builder):
            pass
        builder.get_memory_client.assert_awaited_once_with("mem0")
        builder.get_function.assert_awaited_once_with("inner-agent")

    async def test_passthroughs_when_memory_backend_unconfigured(self, context_user_id):
        config = _make_config(memory_name="mem0")
        chunks = [_chunk("Hello"), _chunk(" world")]
        builder = _make_builder(
            memory_editor=UnconfiguredMemoryEditor(),
            inner_agent_chunks=chunks,
        )

        async with streaming_memory_agent(config, builder) as fn_info:
            out = await _drain(fn_info.stream_fn(_input(_user("hi"))))

        assert _content_deltas(out) == ["Hello", " world"]
        builder.get_memory_client.assert_awaited_once_with("mem0")
        builder.get_function.assert_awaited_once_with("inner-agent")


# ---------------------------------------------------------------------------
# streaming_memory_agent — stream_fn behavior
# ---------------------------------------------------------------------------


def _content_deltas(responses: list[DRAgentEventResponse]) -> list[str]:
    """Pull TEXT_MESSAGE_CONTENT deltas out of an event-response list."""
    out: list[str] = []
    for r in responses:
        for ev in r.events:
            if isinstance(ev, TextMessageContentEvent):
                out.append(ev.delta)
    return out


class TestStreamFnAllFlagsOff:
    """All save/retrieve flags off → chunks still stream, no mem0 calls."""

    async def test_streams_chunks_unchanged(self, context_user_id):
        config = _make_config(
            save_user_messages_to_memory=False,
            retrieve_memory_for_every_response=False,
            save_ai_messages_to_memory=False,
        )
        memory_editor = AsyncMock()
        chunks = [_chunk("Hello"), _chunk(" world")]
        builder = _make_builder(memory_editor=memory_editor, inner_agent_chunks=chunks)

        async with streaming_memory_agent(config, builder) as fn_info:
            out = await _drain(fn_info.stream_fn(_input(_user("hi"))))

        assert _content_deltas(out) == ["Hello", " world"]
        # No memory side effects when every flag is off.
        memory_editor.add_items.assert_not_called()
        memory_editor.search.assert_not_called()


class TestStreamFnPassthroughWithoutMemory:
    """Unconfigured dr_mem0_memory → inner agent stream passes through unchanged."""

    async def test_streams_inner_agent_without_memory_calls(self, context_user_id):
        config = _make_config(memory_name="mem0")
        chunks = [_chunk("Hello"), _chunk(" world")]
        builder = _make_builder(
            memory_editor=UnconfiguredMemoryEditor(),
            inner_agent_chunks=chunks,
        )

        async with streaming_memory_agent(config, builder) as fn_info:
            out = await _drain(fn_info.stream_fn(_input(_user("hi"))))

        assert _content_deltas(out) == ["Hello", " world"]
        builder.get_memory_client.assert_awaited_once_with("mem0")

    async def test_forwards_input_unchanged(self, context_user_id):
        config = _make_config(memory_name="mem0")
        captured: dict = {}

        async def _astream(inner_request, to_type=None):
            captured["request"] = inner_request
            captured["to_type"] = to_type
            yield _chunk("ok")

        builder = _make_builder(memory_editor=UnconfiguredMemoryEditor())
        builder.get_function.return_value.astream = _astream

        async with streaming_memory_agent(config, builder) as fn_info:
            await _drain(fn_info.stream_fn(_input(_user("hi"))))

        assert captured["to_type"] is None
        assert isinstance(captured["request"], RunAgentInput)
        assert [m.content for m in captured["request"].messages] == ["hi"]


class TestStreamFnSavesUserMessage:
    async def test_saves_last_user_message_to_memory(self, context_user_id):
        config = _make_config(memory_name="mem0")
        memory_editor = AsyncMock()
        memory_editor.search.return_value = []
        builder = _make_builder(memory_editor=memory_editor, inner_agent_chunks=[_chunk("ok")])

        async with streaming_memory_agent(config, builder) as fn_info:
            await _drain(
                fn_info.stream_fn(_input(_user("earlier"), _assistant("a"), _user("latest")))
            )

        # First add_items call should persist the user's latest message.
        first_call = memory_editor.add_items.await_args_list[0]
        (items,) = first_call.args
        assert len(items) == 1
        item = items[0]
        assert isinstance(item, MemoryItem)
        assert item.conversation == [{"role": "user", "content": "latest"}]
        assert item.user_id == "test-user"

    async def test_skips_save_when_disabled(self, context_user_id):
        config = _make_config(memory_name="mem0", save_user_messages_to_memory=False)
        memory_editor = AsyncMock()
        memory_editor.search.return_value = []
        builder = _make_builder(memory_editor=memory_editor, inner_agent_chunks=[_chunk("ok")])

        async with streaming_memory_agent(config, builder) as fn_info:
            await _drain(fn_info.stream_fn(_input(_user("hi"))))

        # Only the assistant save (after streaming) is expected.
        user_saves = [
            call
            for call in memory_editor.add_items.await_args_list
            if call.args[0][0].conversation[0]["role"] == "user"
        ]
        assert user_saves == []

    async def test_skips_save_when_no_user_text(self, context_user_id):
        config = _make_config(memory_name="mem0")
        memory_editor = AsyncMock()
        memory_editor.search.return_value = []
        builder = _make_builder(memory_editor=memory_editor, inner_agent_chunks=[_chunk("ok")])

        async with streaming_memory_agent(config, builder) as fn_info:
            # No user messages at all.
            await _drain(fn_info.stream_fn(_input(_assistant("just chatting"))))

        user_saves = [
            call
            for call in memory_editor.add_items.await_args_list
            if call.args[0][0].conversation[0]["role"] == "user"
        ]
        assert user_saves == []

    async def test_forwards_add_params(self, context_user_id):
        config = _make_config(memory_name="mem0", add_params={"namespace": "foo"})
        memory_editor = AsyncMock()
        memory_editor.search.return_value = []
        builder = _make_builder(memory_editor=memory_editor, inner_agent_chunks=[_chunk("ok")])

        async with streaming_memory_agent(config, builder) as fn_info:
            await _drain(fn_info.stream_fn(_input(_user("hi"))))

        for call in memory_editor.add_items.await_args_list:
            assert call.kwargs == {"namespace": "foo"}

    async def test_save_failure_does_not_propagate(self, context_user_id, caplog):
        config = _make_config(memory_name="mem0", save_ai_messages_to_memory=False)
        memory_editor = AsyncMock()
        memory_editor.add_items.side_effect = RuntimeError("kapow")
        memory_editor.search.return_value = []
        builder = _make_builder(memory_editor=memory_editor, inner_agent_chunks=[_chunk("ok")])

        async with streaming_memory_agent(config, builder) as fn_info:
            with caplog.at_level(logging.WARNING, logger=_MODULE):
                # Should not raise even though add_items blew up.
                out = await _drain(fn_info.stream_fn(_input(_user("hi"))))

        assert out  # streaming still happened
        assert any("memory.add_items(user) failed" in rec.message for rec in caplog.records)


class TestStreamFnRetrievesMemory:
    async def test_injects_search_results_as_system_message(self, context_user_id):
        config = _make_config(
            memory_name="mem0",
            save_user_messages_to_memory=False,
            save_ai_messages_to_memory=False,
        )
        memory_editor = AsyncMock()
        memory_editor.search.return_value = [
            MemoryItem(memory="user likes cats", user_id="test-user"),
            MemoryItem(memory="user lives in NYC", user_id="test-user"),
        ]

        captured_request: dict = {}

        async def _astream(inner_request, to_type=None):  # noqa: ARG001
            captured_request["request"] = inner_request
            yield _chunk("ok")

        builder = _make_builder(memory_editor=memory_editor)
        builder.get_function.return_value.astream = _astream

        async with streaming_memory_agent(config, builder) as fn_info:
            await _drain(fn_info.stream_fn(_input(_user("first"), _assistant("ok"), _user("now"))))

        memory_editor.search.assert_awaited_once()
        assert memory_editor.search.await_args.kwargs["query"] == "now"
        assert memory_editor.search.await_args.kwargs["user_id"] == "test-user"

        inner_request: RunAgentInput = captured_request["request"]
        # System message with both memory snippets, inserted right before the last user message.
        assert isinstance(inner_request.messages[-1], AgUiUserMessage)
        assert isinstance(inner_request.messages[-2], AgUiSystemMessage)
        injected = str(inner_request.messages[-2].content)
        assert "user likes cats" in injected
        assert "user lives in NYC" in injected

    async def test_skips_search_when_disabled(self, context_user_id):
        config = _make_config(memory_name="mem0", retrieve_memory_for_every_response=False)
        memory_editor = AsyncMock()
        memory_editor.search.return_value = []
        builder = _make_builder(memory_editor=memory_editor, inner_agent_chunks=[_chunk("ok")])

        async with streaming_memory_agent(config, builder) as fn_info:
            await _drain(fn_info.stream_fn(_input(_user("hi"))))

        memory_editor.search.assert_not_called()

    async def test_no_system_message_when_search_returns_no_text(self, context_user_id):
        config = _make_config(
            memory_name="mem0",
            save_user_messages_to_memory=False,
            save_ai_messages_to_memory=False,
        )
        memory_editor = AsyncMock()
        # Items with no `memory` text get filtered out.
        memory_editor.search.return_value = [MemoryItem(memory=None, user_id="test-user")]

        captured: dict = {}

        async def _astream(inner_request, to_type=None):  # noqa: ARG001
            captured["request"] = inner_request
            yield _chunk("ok")

        builder = _make_builder(memory_editor=memory_editor)
        builder.get_function.return_value.astream = _astream

        async with streaming_memory_agent(config, builder) as fn_info:
            await _drain(fn_info.stream_fn(_input(_user("hi"))))

        assert not any(isinstance(m, AgUiSystemMessage) for m in captured["request"].messages)

    async def test_forwards_search_params(self, context_user_id):
        config = _make_config(
            memory_name="mem0",
            save_user_messages_to_memory=False,
            save_ai_messages_to_memory=False,
            search_params={"top_k": 5},
        )
        memory_editor = AsyncMock()
        memory_editor.search.return_value = []
        builder = _make_builder(memory_editor=memory_editor, inner_agent_chunks=[_chunk("ok")])

        async with streaming_memory_agent(config, builder) as fn_info:
            await _drain(fn_info.stream_fn(_input(_user("hi"))))

        assert memory_editor.search.await_args.kwargs["top_k"] == 5

    async def test_search_failure_does_not_propagate(self, context_user_id, caplog):
        config = _make_config(
            memory_name="mem0",
            save_user_messages_to_memory=False,
            save_ai_messages_to_memory=False,
        )
        memory_editor = AsyncMock()
        memory_editor.search.side_effect = RuntimeError("search down")
        builder = _make_builder(memory_editor=memory_editor, inner_agent_chunks=[_chunk("ok")])

        async with streaming_memory_agent(config, builder) as fn_info:
            with caplog.at_level(logging.WARNING, logger=_MODULE):
                out = await _drain(fn_info.stream_fn(_input(_user("hi"))))

        assert out
        assert any("memory.search failed" in rec.message for rec in caplog.records)


class TestStreamFnSavesAiResponse:
    async def test_accumulates_text_and_saves_after_stream(self, context_user_id):
        config = _make_config(
            memory_name="mem0",
            save_user_messages_to_memory=False,
            retrieve_memory_for_every_response=False,
        )
        memory_editor = AsyncMock()
        memory_editor.search.return_value = []
        builder = _make_builder(
            memory_editor=memory_editor,
            inner_agent_chunks=[_chunk("Hel"), _chunk("lo"), _chunk(" world")],
        )

        async with streaming_memory_agent(config, builder) as fn_info:
            out = await _drain(fn_info.stream_fn(_input(_user("hi"))))

        # All chunks streamed through.
        assert _content_deltas(out) == ["Hel", "lo", " world"]

        # Exactly one add_items call, for the assistant turn.
        assert memory_editor.add_items.await_count == 1
        (items,) = memory_editor.add_items.await_args.args
        assert items[0].conversation == [{"role": "assistant", "content": "Hello world"}]
        assert items[0].user_id == "test-user"

    async def test_skips_save_when_disabled(self, context_user_id):
        config = _make_config(
            memory_name="mem0",
            save_user_messages_to_memory=False,
            retrieve_memory_for_every_response=False,
            save_ai_messages_to_memory=False,
        )
        memory_editor = AsyncMock()
        builder = _make_builder(memory_editor=memory_editor, inner_agent_chunks=[_chunk("hi")])

        async with streaming_memory_agent(config, builder) as fn_info:
            await _drain(fn_info.stream_fn(_input(_user("hi"))))

        memory_editor.add_items.assert_not_called()

    async def test_skips_save_when_no_text_accumulated(self, context_user_id):
        config = _make_config(
            memory_name="mem0",
            save_user_messages_to_memory=False,
            retrieve_memory_for_every_response=False,
        )
        memory_editor = AsyncMock()
        # No chunks → no accumulated text → skip the assistant save.
        builder = _make_builder(memory_editor=memory_editor, inner_agent_chunks=[])

        async with streaming_memory_agent(config, builder) as fn_info:
            await _drain(fn_info.stream_fn(_input(_user("hi"))))

        memory_editor.add_items.assert_not_called()

    async def test_save_failure_does_not_propagate(self, context_user_id, caplog):
        config = _make_config(
            memory_name="mem0",
            save_user_messages_to_memory=False,
            retrieve_memory_for_every_response=False,
        )
        memory_editor = AsyncMock()
        memory_editor.add_items.side_effect = RuntimeError("save down")
        builder = _make_builder(memory_editor=memory_editor, inner_agent_chunks=[_chunk("ok")])

        async with streaming_memory_agent(config, builder) as fn_info:
            with caplog.at_level(logging.WARNING, logger=_MODULE):
                out = await _drain(fn_info.stream_fn(_input(_user("hi"))))

        assert out  # streaming itself succeeded
        assert any("memory.add_items(assistant) failed" in rec.message for rec in caplog.records)


class TestStreamFnInnerRequest:
    async def test_inner_request_carries_modified_messages(self, context_user_id):
        config = _make_config(
            save_user_messages_to_memory=False,
            retrieve_memory_for_every_response=False,
            save_ai_messages_to_memory=False,
        )
        captured: dict = {}

        async def _astream(inner_request, to_type=None):
            captured["request"] = inner_request
            captured["to_type"] = to_type
            yield _chunk("ok")

        builder = _make_builder(memory_editor=AsyncMock())
        builder.get_function.return_value.astream = _astream

        async with streaming_memory_agent(config, builder) as fn_info:
            await _drain(fn_info.stream_fn(_input(_user("hi"))))

        # No explicit to_type — the wrapper relies on the inner agent's native
        # DRAgentEventResponse stream rather than converting.
        assert captured["to_type"] is None
        # And RunAgentInput is forwarded (possibly with extra system messages, but
        # in this test no flags are on so nothing is injected).
        assert isinstance(captured["request"], RunAgentInput)
        assert [m.content for m in captured["request"].messages] == ["hi"]
