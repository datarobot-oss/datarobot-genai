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

"""Unit tests for the chat / message repositories against a fake Memory Service.

Every test drives the real :class:`ChatRepository` / :class:`MessageRepository`
against an in-memory, respx-backed fake of the REST API, exercising the create /
read / update / delete paths, the one-event-per-message nesting, the content
placeholder boundary, cache and registry behaviour, dedup idempotency and the
bounded retry on optimistic-concurrency conflicts.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from uuid import uuid4

import pytest
import respx

from datarobot_genai.application_utils.chat_history.constants import _ZW_PLACEHOLDER
from datarobot_genai.application_utils.chat_history.constants import participant_id
from datarobot_genai.application_utils.chat_history.models import ChatCreate
from datarobot_genai.application_utils.chat_history.models import MessageCreate
from datarobot_genai.application_utils.chat_history.models import MessageReasoningCreate
from datarobot_genai.application_utils.chat_history.models import MessageReasoningUpdate
from datarobot_genai.application_utils.chat_history.models import MessageToolCallCreate
from datarobot_genai.application_utils.chat_history.models import MessageToolCallUpdate
from datarobot_genai.application_utils.chat_history.models import MessageUpdate
from datarobot_genai.application_utils.chat_history.models import Role
from datarobot_genai.application_utils.chat_history.repositories import ChatRepository
from datarobot_genai.application_utils.chat_history.repositories import ChatSessionRegistry
from datarobot_genai.application_utils.chat_history.repositories import MessageRepository
from datarobot_genai.application_utils.persistence import DRMemorySpace
from datarobot_genai.application_utils.persistence import DRMemoryVersionConflictError
from tests.application_utils.chat_history.unit.conftest import ProjectChat
from tests.application_utils.chat_history.unit.conftest import RichMessage
from tests.application_utils.chat_history.unit.fake_memory import FakeMemoryService


@dataclass
class Env:
    """Bundle of a fake service plus repositories wired against it."""

    svc: FakeMemoryService
    space: DRMemorySpace
    registry: ChatSessionRegistry
    chat_repo: ChatRepository
    message_repo: MessageRepository


@pytest.fixture
def env() -> Iterator[Env]:
    """GIVEN a fake Memory Service WHEN a test runs THEN repositories target it."""
    with respx.mock:
        svc = FakeMemoryService()
        svc.install()
        space = svc.space()
        registry = ChatSessionRegistry(space)
        yield Env(
            svc=svc,
            space=space,
            registry=registry,
            chat_repo=ChatRepository(space, registry),
            message_repo=MessageRepository(space, registry),
        )


async def _make_chat(env: Env, *, thread_id: str = "thread-1", user=None):
    """Create a chat and return it."""
    user_uuid = user or uuid4()
    return await env.chat_repo.create_chat(
        ChatCreate(name="Chat", thread_id=thread_id, user_uuid=user_uuid)
    )


def _session_id(env: Env, chat_uuid) -> str:
    sid = env.registry.get_session_id(chat_uuid)
    assert sid is not None
    return sid


# ── ChatRepository ──────────────────────────────────────────────────────────────


async def test_create_chat_places_fields_and_registers(env: Env) -> None:
    """GIVEN user+thread WHEN creating a chat THEN wire slots + registry are populated."""
    user = uuid4()
    chat = await env.chat_repo.create_chat(
        ChatCreate(name="Billing", thread_id="thread-42", user_uuid=user)
    )

    sid = _session_id(env, chat.chat_uuid)
    stored = env.svc.sessions[sid]
    assert stored["description"] == "//thread/thread-42/"
    assert stored["deduplicationKey"]  # derived, non-empty
    assert stored["participants"] == [participant_id(user)]
    assert stored["metadata"]["name"] == "Billing"
    assert stored["metadata"]["user_uuid"] == str(user)
    assert chat.thread_id == "thread-42"
    assert chat.user_uuid == user


async def test_create_chat_is_idempotent_by_user_thread(env: Env) -> None:
    """GIVEN an existing (user, thread) WHEN creating again THEN the same chat is returned."""
    user = uuid4()
    first = await env.chat_repo.create_chat(ChatCreate(name="One", thread_id="t", user_uuid=user))
    second = await env.chat_repo.create_chat(ChatCreate(name="Two", thread_id="t", user_uuid=user))

    assert first.chat_uuid == second.chat_uuid
    assert len(env.svc.sessions) == 1  # no duplicate session created


async def test_create_chat_requires_user_and_thread(env: Env) -> None:
    """GIVEN a missing user or thread WHEN creating a chat THEN it raises ValueError."""
    with pytest.raises(ValueError, match="user_uuid is required"):
        await env.chat_repo.create_chat(ChatCreate(thread_id="t"))
    with pytest.raises(ValueError, match="thread_id is required"):
        await env.chat_repo.create_chat(ChatCreate(user_uuid=uuid4()))


async def test_get_chat_by_thread_id_hit_and_miss(env: Env) -> None:
    """GIVEN a stored chat WHEN looking up by thread THEN it is found; unknown → None."""
    user = uuid4()
    created = await _make_chat(env, thread_id="thread-x", user=user)

    found = await env.chat_repo.get_chat_by_thread_id(user, "thread-x")
    assert found is not None
    assert found.chat_uuid == created.chat_uuid

    assert await env.chat_repo.get_chat_by_thread_id(user, "nope") is None
    assert await env.chat_repo.get_chat_by_thread_id(uuid4(), "thread-x") is None


async def test_get_all_chats_scoped_to_user(env: Env) -> None:
    """GIVEN chats for two users WHEN scoping to one THEN only their chats are returned."""
    alice, bob = uuid4(), uuid4()
    await _make_chat(env, thread_id="a1", user=alice)
    await _make_chat(env, thread_id="a2", user=alice)
    await _make_chat(env, thread_id="b1", user=bob)

    alice_chats = await env.chat_repo.get_all_chats(alice)
    assert {c.thread_id for c in alice_chats} == {"a1", "a2"}

    everyone = await env.chat_repo.get_all_chats(None)
    assert len(everyone) == 3


async def test_update_chat_name(env: Env) -> None:
    """GIVEN a chat WHEN renaming THEN the new name is persisted."""
    chat = await _make_chat(env)
    updated = await env.chat_repo.update_chat_name(chat.chat_uuid, "Renamed")

    assert updated is not None
    assert updated.name == "Renamed"
    assert env.svc.sessions[_session_id(env, chat.chat_uuid)]["metadata"]["name"] == "Renamed"


async def test_update_chat_name_retries_on_version_conflict(env: Env) -> None:
    """GIVEN two transient 409s WHEN renaming THEN it retries and eventually succeeds."""
    chat = await _make_chat(env)
    env.svc.fail_session_patch_times = 2

    updated = await env.chat_repo.update_chat_name(chat.chat_uuid, "Retried")

    assert updated is not None
    assert updated.name == "Retried"
    assert env.svc.fail_session_patch_times == 0


async def test_update_chat_name_raises_when_retries_exhausted(env: Env) -> None:
    """GIVEN unrelenting 409s WHEN renaming THEN the conflict propagates after the bound."""
    chat = await _make_chat(env)
    env.svc.fail_session_patch_times = 99

    with pytest.raises(DRMemoryVersionConflictError):
        await env.chat_repo.update_chat_name(chat.chat_uuid, "Nope")


async def test_update_chat_name_unknown_returns_none(env: Env) -> None:
    """GIVEN an unknown chat WHEN renaming THEN None is returned."""
    assert await env.chat_repo.update_chat_name(uuid4(), "x") is None


async def test_delete_chat_unregisters(env: Env) -> None:
    """GIVEN a chat WHEN deleting THEN the session is gone and the registry entry cleared."""
    chat = await _make_chat(env)
    sid = _session_id(env, chat.chat_uuid)

    deleted = await env.chat_repo.delete_chat(chat.chat_uuid)

    assert deleted is not None
    assert sid not in env.svc.sessions
    assert env.registry.get_session_id(chat.chat_uuid) is None
    assert await env.chat_repo.delete_chat(uuid4()) is None


# ── MessageRepository — create / emitter / placeholder ──────────────────────────


async def test_create_user_message_sets_user_emitter(env: Env) -> None:
    """GIVEN a user message WHEN created THEN the emitter is (user, participant id)."""
    user = uuid4()
    chat = await _make_chat(env, user=user)

    msg = await env.message_repo.create_message(
        MessageCreate(role=Role.USER.value, content="hello", chat_id=chat.chat_uuid)
    )

    sid = _session_id(env, chat.chat_uuid)
    wire = env.svc.events[sid][0]
    assert wire["emitterType"] == "user"
    assert wire["emitterId"] == participant_id(user)
    assert msg.content == "hello"
    assert env.message_repo._msg_chat[msg.message_uuid] == chat.chat_uuid


async def test_create_agent_message_has_no_emitter_id(env: Env) -> None:
    """GIVEN an assistant message WHEN created THEN the emitter is (agent, no id)."""
    chat = await _make_chat(env)

    await env.message_repo.create_message(
        MessageCreate(role=Role.ASSISTANT.value, content="hi", chat_id=chat.chat_uuid)
    )

    wire = env.svc.events[_session_id(env, chat.chat_uuid)][0]
    assert wire["emitterType"] == "agent"
    assert wire["emitterId"] is None


async def test_create_message_empty_content_uses_placeholder_on_wire(env: Env) -> None:
    """GIVEN empty content WHEN created THEN the wire uses the placeholder, read decodes it."""
    chat = await _make_chat(env)

    msg = await env.message_repo.create_message(
        MessageCreate(role=Role.ASSISTANT.value, content="", chat_id=chat.chat_uuid)
    )

    wire = env.svc.events[_session_id(env, chat.chat_uuid)][0]
    assert wire["body"]["content"] == _ZW_PLACEHOLDER
    assert msg.content == ""


async def test_create_message_requires_chat_id(env: Env) -> None:
    """GIVEN no chat_id WHEN creating a message THEN it raises ValueError."""
    with pytest.raises(ValueError, match="chat_id is required"):
        await env.message_repo.create_message(MessageCreate(content="x"))


async def test_create_message_is_one_event(env: Env) -> None:
    """GIVEN a message with a tool call and reasoning WHEN created THEN it is a single event."""
    chat = await _make_chat(env)
    msg = await env.message_repo.create_message(
        MessageCreate(role=Role.ASSISTANT.value, content="a", chat_id=chat.chat_uuid)
    )
    await env.message_repo.create_message_tool_call(
        MessageToolCallCreate(message_uuid=msg.message_uuid, name="search")
    )
    await env.message_repo.create_message_reasoning(
        MessageReasoningCreate(message_uuid=msg.message_uuid, content="think")
    )

    assert env.svc.event_count(_session_id(env, chat.chat_uuid)) == 1


# ── MessageRepository — update ──────────────────────────────────────────────────


async def test_update_message_applies_set_fields(env: Env) -> None:
    """GIVEN a live message WHEN updated THEN status/in_progress/content change."""
    chat = await _make_chat(env)
    msg = await env.message_repo.create_message(
        MessageCreate(role=Role.ASSISTANT.value, content="draft", chat_id=chat.chat_uuid)
    )

    updated = await env.message_repo.update_message(
        msg.message_uuid,
        MessageUpdate(content="final", in_progress=False, status="complete"),
    )

    assert updated is not None
    assert updated.content == "final"
    assert updated.in_progress is False
    assert updated.status == "complete"
    wire = env.svc.events[_session_id(env, chat.chat_uuid)][0]
    assert wire["body"]["status"] == "complete"
    assert wire["body"]["in_progress"] is False


async def test_update_message_unknown_returns_none(env: Env) -> None:
    """GIVEN an unknown message WHEN updated THEN None is returned."""
    assert await env.message_repo.update_message(uuid4(), MessageUpdate(content="x")) is None


# ── MessageRepository — tool calls / reasonings (nested body rewrite) ────────────


async def test_create_and_update_tool_call_rewrites_body(env: Env) -> None:
    """GIVEN a message WHEN a tool call is appended/updated THEN the event body is rewritten."""
    chat = await _make_chat(env)
    msg = await env.message_repo.create_message(
        MessageCreate(role=Role.ASSISTANT.value, content="", chat_id=chat.chat_uuid)
    )

    tc = await env.message_repo.create_message_tool_call(
        MessageToolCallCreate(
            message_uuid=msg.message_uuid, name="search", arguments='{"q":1}', agui_id="tc-1"
        )
    )
    wire = env.svc.events[_session_id(env, chat.chat_uuid)][0]
    assert wire["body"]["tool_calls"][0]["name"] == "search"
    assert wire["body"]["tool_calls"][0]["arguments"] == '{"q":1}'

    updated = await env.message_repo.update_message_tool_call(
        tc.uuid, MessageToolCallUpdate(content="result", in_progress=False, status="complete")
    )
    assert updated is not None
    assert updated.content == "result"
    assert updated.in_progress is False
    wire = env.svc.events[_session_id(env, chat.chat_uuid)][0]
    assert wire["body"]["tool_calls"][0]["content"] == "result"


async def test_create_and_update_reasoning_rewrites_body(env: Env) -> None:
    """GIVEN a message WHEN a reasoning is appended/updated THEN the event body is rewritten."""
    chat = await _make_chat(env)
    msg = await env.message_repo.create_message(
        MessageCreate(role=Role.ASSISTANT.value, content="", chat_id=chat.chat_uuid)
    )

    reasoning = await env.message_repo.create_message_reasoning(
        MessageReasoningCreate(message_uuid=msg.message_uuid, content="planning", agui_id="r-1")
    )
    wire = env.svc.events[_session_id(env, chat.chat_uuid)][0]
    assert wire["body"]["reasonings"][0]["content"] == "planning"

    updated = await env.message_repo.update_message_reasoning(
        reasoning.uuid, MessageReasoningUpdate(content="planned", in_progress=False)
    )
    assert updated is not None
    assert updated.content == "planned"
    assert updated.in_progress is False


async def test_update_tool_call_discovers_on_cold_cache(env: Env) -> None:
    """GIVEN a fresh repo WHEN updating a tool call THEN its parent is discovered by scan."""
    chat = await _make_chat(env)
    msg = await env.message_repo.create_message(
        MessageCreate(role=Role.ASSISTANT.value, content="", chat_id=chat.chat_uuid)
    )
    tc = await env.message_repo.create_message_tool_call(
        MessageToolCallCreate(message_uuid=msg.message_uuid, name="search")
    )

    cold_repo = MessageRepository(env.space, ChatSessionRegistry(env.space))
    updated = await cold_repo.update_message_tool_call(
        tc.uuid, MessageToolCallUpdate(status="complete")
    )

    assert updated is not None
    assert updated.status == "complete"


async def test_update_reasoning_discovers_on_cold_cache(env: Env) -> None:
    """GIVEN a fresh repo WHEN updating a reasoning THEN its parent is discovered by scan."""
    chat = await _make_chat(env)
    msg = await env.message_repo.create_message(
        MessageCreate(role=Role.ASSISTANT.value, content="", chat_id=chat.chat_uuid)
    )
    reasoning = await env.message_repo.create_message_reasoning(
        MessageReasoningCreate(message_uuid=msg.message_uuid, content="a")
    )

    cold_repo = MessageRepository(env.space, ChatSessionRegistry(env.space))
    updated = await cold_repo.update_message_reasoning(
        reasoning.uuid, MessageReasoningUpdate(content="b")
    )

    assert updated is not None
    assert updated.content == "b"


async def test_create_tool_call_for_missing_message_raises(env: Env) -> None:
    """GIVEN no such message WHEN appending a tool call THEN it raises ValueError."""
    with pytest.raises(ValueError, match="does not exist"):
        await env.message_repo.create_message_tool_call(
            MessageToolCallCreate(message_uuid=uuid4(), name="x")
        )


async def test_update_tool_call_unknown_returns_none(env: Env) -> None:
    """GIVEN an unknown tool call WHEN updated THEN None is returned."""
    assert (
        await env.message_repo.update_message_tool_call(uuid4(), MessageToolCallUpdate(content="x"))
        is None
    )


async def test_update_tool_call_retries_on_event_conflict(env: Env) -> None:
    """GIVEN transient 422s WHEN updating a tool call THEN it retries and succeeds."""
    chat = await _make_chat(env)
    msg = await env.message_repo.create_message(
        MessageCreate(role=Role.ASSISTANT.value, content="", chat_id=chat.chat_uuid)
    )
    tc = await env.message_repo.create_message_tool_call(
        MessageToolCallCreate(message_uuid=msg.message_uuid, name="search")
    )
    env.svc.fail_event_patch_times = 2

    updated = await env.message_repo.update_message_tool_call(
        tc.uuid, MessageToolCallUpdate(status="complete")
    )

    assert updated is not None
    assert updated.status == "complete"
    assert env.svc.fail_event_patch_times == 0


# ── MessageRepository — reads / ordering / discovery ────────────────────────────


async def test_get_chat_messages_ordered_by_sequence(env: Env) -> None:
    """GIVEN several messages WHEN listing a chat THEN they come back in creation order."""
    chat = await _make_chat(env)
    for i in range(3):
        await env.message_repo.create_message(
            MessageCreate(role=Role.ASSISTANT.value, content=f"m{i}", chat_id=chat.chat_uuid)
        )

    messages = await env.message_repo.get_chat_messages(chat.chat_uuid)
    assert [m.content for m in messages] == ["m0", "m1", "m2"]


async def test_get_last_messages_returns_latest_per_chat(env: Env) -> None:
    """GIVEN two chats WHEN fetching last messages THEN each maps to its newest message."""
    chat_a = await _make_chat(env, thread_id="a")
    chat_b = await _make_chat(env, thread_id="b")
    await env.message_repo.create_message(
        MessageCreate(role=Role.ASSISTANT.value, content="a-old", chat_id=chat_a.chat_uuid)
    )
    await env.message_repo.create_message(
        MessageCreate(role=Role.ASSISTANT.value, content="a-new", chat_id=chat_a.chat_uuid)
    )
    await env.message_repo.create_message(
        MessageCreate(role=Role.ASSISTANT.value, content="b-only", chat_id=chat_b.chat_uuid)
    )

    last = await env.message_repo.get_last_messages([chat_a.chat_uuid, chat_b.chat_uuid])

    assert last[chat_a.chat_uuid].content == "a-new"
    assert last[chat_b.chat_uuid].content == "b-only"


async def test_get_message_by_agui_id(env: Env) -> None:
    """GIVEN a message with an AG-UI id WHEN looked up by it THEN the message is returned."""
    chat = await _make_chat(env)
    await env.message_repo.create_message(
        MessageCreate(
            role=Role.ASSISTANT.value, content="x", agui_id="m-77", chat_id=chat.chat_uuid
        )
    )

    found = await env.message_repo.get_message_by_agui_id(chat.chat_uuid, "m-77")
    assert found is not None
    assert found.agui_id == "m-77"
    assert await env.message_repo.get_message_by_agui_id(chat.chat_uuid, "absent") is None


async def test_get_tool_call_by_agui_id(env: Env) -> None:
    """GIVEN a tool call with an AG-UI id WHEN looked up by it THEN the tool call is returned."""
    chat = await _make_chat(env)
    msg = await env.message_repo.create_message(
        MessageCreate(role=Role.ASSISTANT.value, content="", chat_id=chat.chat_uuid)
    )
    await env.message_repo.create_message_tool_call(
        MessageToolCallCreate(message_uuid=msg.message_uuid, name="s", agui_id="tc-9")
    )

    found = await env.message_repo.get_tool_call_by_agui_id(msg.message_uuid, "tc-9")
    assert found is not None
    assert found.agui_id == "tc-9"


async def test_get_message_discovers_on_cold_cache(env: Env) -> None:
    """GIVEN a fresh repo/registry WHEN getting a message THEN it is discovered by scan."""
    chat = await _make_chat(env)
    msg = await env.message_repo.create_message(
        MessageCreate(role=Role.ASSISTANT.value, content="hi", chat_id=chat.chat_uuid)
    )

    # A brand-new registry and repository have empty caches — force the scan path.
    cold_registry = ChatSessionRegistry(env.space)
    cold_repo = MessageRepository(env.space, cold_registry)

    found = await cold_repo.get_message(msg.message_uuid)
    assert found is not None
    assert found.content == "hi"
    # The scan registered the chat → session mapping.
    assert cold_registry.get_session_id(chat.chat_uuid) is not None


async def test_get_message_unknown_returns_none(env: Env) -> None:
    """GIVEN an unknown message uuid WHEN getting it THEN None is returned."""
    assert await env.message_repo.get_message(uuid4()) is None


# ── Registry ──────────────────────────────────────────────────────────────────────


async def test_registry_resolve_scans_on_miss(env: Env) -> None:
    """GIVEN an unseen chat WHEN resolving THEN a metadata scan finds and caches the session."""
    chat = await _make_chat(env)
    cold_registry = ChatSessionRegistry(env.space)

    sid = await cold_registry.resolve(chat.chat_uuid)

    assert sid == _session_id(env, chat.chat_uuid)
    assert cold_registry.get_session_id(chat.chat_uuid) == sid  # now cached


async def test_registry_resolve_unknown_returns_none(env: Env) -> None:
    """GIVEN no matching session WHEN resolving THEN None is returned."""
    assert await env.registry.resolve(uuid4()) is None


async def test_transaction_is_a_noop_context(env: Env) -> None:
    """GIVEN the message repo WHEN entering transaction() THEN it is an inert async scope."""
    async with env.message_repo.transaction():
        chat = await _make_chat(env)
        msg = await env.message_repo.create_message(
            MessageCreate(role=Role.USER.value, content="tx", chat_id=chat.chat_uuid)
        )
    assert msg.content == "tx"


# ── Subclass extensibility (repos thread the injected chat_cls / message_cls) ─────


async def test_repositories_thread_injected_subclasses_end_to_end(env: Env) -> None:
    """GIVEN Chat/Message subclasses WHEN driven through the repos THEN those types flow back.

    Proves Decision 4 / deviation #3: the repositories route every ORM call
    through the injected ``chat_cls`` / ``message_cls`` (they are not hardcoded to
    the base :class:`Chat` / :class:`Message`), so a consumer's subclass is what
    ``create_*`` and the read paths construct and return.
    """
    registry = ChatSessionRegistry(env.space, chat_cls=ProjectChat)
    chat_repo = ChatRepository(env.space, registry, chat_cls=ProjectChat)
    message_repo = MessageRepository(
        env.space, registry, chat_cls=ProjectChat, message_cls=RichMessage
    )

    user = uuid4()
    chat = await chat_repo.create_chat(ChatCreate(name="C", thread_id="t", user_uuid=user))
    assert isinstance(chat, ProjectChat)

    # The registry (also subclass-parameterised) resolves back to the subclass too.
    assert isinstance(await chat_repo.get_chat_by_thread_id(user, "t"), ProjectChat)

    msg = await message_repo.create_message(
        MessageCreate(role=Role.ASSISTANT.value, content="hi", chat_id=chat.chat_uuid)
    )
    assert isinstance(msg, RichMessage)

    fetched = await message_repo.get_chat_messages(chat.chat_uuid)
    assert len(fetched) == 1
    assert isinstance(fetched[0], RichMessage)
    assert fetched[0].content == "hi"
