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

"""Unit tests for the chat-history ORM models and DTOs.

Covers field routing (dedup / description / metadata / body placement), the
zero-width placeholder round-trip on nested models, enum/status handling, the
public read view, and the subclass-extensibility contract (extra scalar and
nested fields persisted and hydrated through the ORM).
"""

from __future__ import annotations

from typing import Any
from uuid import UUID
from uuid import uuid4

import respx

from datarobot_genai.application_utils.chat_history.constants import _ZW_PLACEHOLDER
from datarobot_genai.application_utils.chat_history.models import Chat
from datarobot_genai.application_utils.chat_history.models import Message
from datarobot_genai.application_utils.chat_history.models import MessageStatus
from datarobot_genai.application_utils.chat_history.models import Reasoning
from datarobot_genai.application_utils.chat_history.models import Role
from datarobot_genai.application_utils.chat_history.models import ToolCall
from tests.application_utils.chat_history.unit.conftest import EVENTS_URL
from tests.application_utils.chat_history.unit.conftest import SESSIONS_URL
from tests.application_utils.chat_history.unit.conftest import Attachment
from tests.application_utils.chat_history.unit.conftest import ProjectChat
from tests.application_utils.chat_history.unit.conftest import RichMessage
from tests.application_utils.chat_history.unit.conftest import echo_event
from tests.application_utils.chat_history.unit.conftest import echo_session
from tests.application_utils.chat_history.unit.conftest import make_chat
from tests.application_utils.chat_history.unit.conftest import make_space

# ── Enums ─────────────────────────────────────────────────────────────────────


def test_role_enum_values() -> None:
    """GIVEN the Role enum WHEN inspected THEN it mirrors the AG-UI message roles."""
    assert [r.value for r in Role] == [
        "developer",
        "system",
        "assistant",
        "user",
        "tool",
        "reasoning",
    ]


def test_message_status_enum_values() -> None:
    """GIVEN the MessageStatus enum WHEN inspected THEN the four lifecycle states exist."""
    assert [s.value for s in MessageStatus] == ["active", "complete", "interrupted", "errored"]


# ── Chat field routing ──────────────────────────────────────────────────────────


def test_chat_field_routing_places_fields_on_the_wire() -> None:
    """GIVEN a Chat WHEN building the create payload THEN each field lands in its wire slot."""
    chat_uuid = uuid4()
    user_uuid = uuid4()
    payload = Chat._to_wire_create(
        {
            "thread_id": "thread-1",
            "dedup_key": "dedup-1",
            "name": "My Chat",
            "chat_uuid": chat_uuid,
            "user_uuid": user_uuid,
            "participants": ["pid-1"],
        }
    )

    assert payload["deduplicationKey"] == "dedup-1"
    assert payload["description"] == "//thread/thread-1/"
    assert payload["participants"] == ["pid-1"]
    assert payload["metadata"] == {
        "name": "My Chat",
        "chat_uuid": str(chat_uuid),
        "user_uuid": str(user_uuid),
    }
    # Routing markers keep the range/dedup keys out of metadata.
    assert "thread_id" not in payload["metadata"]
    assert "dedup_key" not in payload["metadata"]


def test_chat_metadata_hydrates_typed_uuids() -> None:
    """GIVEN a session wire dict WHEN hydrating a Chat THEN metadata UUID strings become UUIDs."""
    chat_uuid = uuid4()
    user_uuid = uuid4()
    wire = {
        "id": "sid",
        "participants": ["pid-1"],
        "description": "//thread/thread-1/",
        "deduplicationKey": "dedup-1",
        "metadata": {
            "name": "Hydrated",
            "chat_uuid": str(chat_uuid),
            "user_uuid": str(user_uuid),
        },
        "version": 3,
        "createdAt": "2026-06-30T00:00:00Z",
    }

    chat: Chat = Chat._from_wire(make_space(), wire)  # type: ignore[assignment]

    assert chat.thread_id == "thread-1"
    assert chat.dedup_key == "dedup-1"
    assert chat.name == "Hydrated"
    assert chat.chat_uuid == chat_uuid
    assert isinstance(chat.chat_uuid, UUID)
    assert chat.user_uuid == user_uuid


# ── Message field routing ────────────────────────────────────────────────────────


def test_message_body_routing_uses_v_key_literally() -> None:
    """GIVEN a Message WHEN building the body THEN declared fields (incl. `v`) land in body."""
    message_uuid = uuid4()
    body = Message._to_wire_body(
        {
            "content": "hello",
            "v": 1,
            "message_uuid": message_uuid,
            "role": "assistant",
            "status": MessageStatus.ACTIVE.value,
        }
    )

    assert body["content"] == "hello"
    assert body["v"] == 1
    assert body["message_uuid"] == str(message_uuid)
    assert body["role"] == "assistant"
    assert body["status"] == "active"


def test_message_content_is_not_placeholder_encoded_at_model_level() -> None:
    """GIVEN an empty base `content` WHEN building the body THEN it stays empty (repo boundary)."""
    body = Message._to_wire_body({"content": ""})
    assert body["content"] == ""


async def test_message_round_trips_typed_nested_models() -> None:
    """GIVEN a Message with nested tool call and reasoning WHEN posted THEN they hydrate typed."""
    with respx.mock:
        captured: dict[str, Any] = {}
        respx.post(EVENTS_URL).mock(side_effect=echo_event(captured))

        tool_call = ToolCall(name="search", arguments='{"q": "x"}', content="found", agui_id="tc1")
        reasoning = Reasoning(name="plan", content="thinking", agui_id="r1")

        message: Message = await Message.post(
            make_chat(),
            content="Hi there",
            emitter_type="agent",
            role="assistant",
            tool_calls=[tool_call],
            reasonings=[reasoning],
        )

    # Wire body carries nested models as JSON dicts.
    wire_body = captured["body"]["body"]
    assert wire_body["tool_calls"][0]["name"] == "search"
    assert wire_body["reasonings"][0]["content"] == "thinking"

    # Read back as typed models.
    assert isinstance(message.tool_calls[0], ToolCall)
    assert message.tool_calls[0].arguments == '{"q": "x"}'
    assert message.tool_calls[0].content == "found"
    assert isinstance(message.reasonings[0], Reasoning)
    assert message.reasonings[0].content == "thinking"


# ── Placeholder round-trip on nested models ────────────────────────────────────


async def test_empty_nested_strings_round_trip_via_placeholder() -> None:
    """GIVEN empty tool-call/reasoning strings WHEN posted THEN the wire uses the placeholder."""
    with respx.mock:
        captured: dict[str, Any] = {}
        respx.post(EVENTS_URL).mock(side_effect=echo_event(captured))

        message: Message = await Message.post(
            make_chat(),
            content="assistant reply",
            emitter_type="agent",
            role="assistant",
            tool_calls=[ToolCall(arguments="", content="")],
            reasonings=[Reasoning(content="")],
        )

    wire_tc = captured["body"]["body"]["tool_calls"][0]
    wire_rs = captured["body"]["body"]["reasonings"][0]
    assert wire_tc["arguments"] == _ZW_PLACEHOLDER
    assert wire_tc["content"] == _ZW_PLACEHOLDER
    assert wire_rs["content"] == _ZW_PLACEHOLDER

    # On read the placeholder is stripped back to the empty string.
    assert message.tool_calls[0].arguments == ""
    assert message.tool_calls[0].content == ""
    assert message.reasonings[0].content == ""


def test_non_empty_nested_strings_are_not_placeholder_encoded() -> None:
    """GIVEN non-empty nested strings WHEN JSON-serialised THEN they are left unchanged."""
    tool_call = ToolCall(arguments="{}", content="done")
    dumped = tool_call.model_dump(mode="json")
    assert dumped["arguments"] == "{}"
    assert dumped["content"] == "done"


def test_python_mode_dump_keeps_empty_strings_clean() -> None:
    """GIVEN an empty nested string WHEN dumped in python mode THEN it stays empty (no leak)."""
    dumped = ToolCall(arguments="", content="").model_dump()
    assert dumped["arguments"] == ""
    assert dumped["content"] == ""


# ── Defaults and status transitions ─────────────────────────────────────────────


def test_nested_model_defaults() -> None:
    """GIVEN freshly-built nested models WHEN inspected THEN roles/status/flags default sensibly."""
    tool_call = ToolCall()
    reasoning = Reasoning()

    assert tool_call.role == Role.TOOL
    assert reasoning.role == Role.REASONING
    assert tool_call.status == MessageStatus.ACTIVE
    assert reasoning.status == MessageStatus.ACTIVE
    assert tool_call.in_progress is True
    assert isinstance(tool_call.uuid, UUID)
    # default_factory yields a distinct id per instance.
    assert ToolCall().uuid != ToolCall().uuid


def test_message_defaults() -> None:
    """GIVEN a new Message WHEN inspected THEN payload version, role and status default."""
    message = Message(content="hi", emitter_type="user")
    assert message.v == 1
    assert message.role == Role.USER
    assert message.status == MessageStatus.ACTIVE
    assert message.in_progress is True
    assert message.tool_calls == []
    assert message.reasonings == []


async def test_status_transition_to_interrupted_round_trips() -> None:
    """GIVEN an interrupted terminal state WHEN posted THEN status/in_progress round-trip."""
    with respx.mock:
        captured: dict[str, Any] = {}
        respx.post(EVENTS_URL).mock(side_effect=echo_event(captured))

        message: Message = await Message.post(
            make_chat(),
            content="partial",
            emitter_type="agent",
            role="assistant",
            status=MessageStatus.INTERRUPTED.value,
            in_progress=False,
        )

    assert captured["body"]["body"]["status"] == "interrupted"
    assert captured["body"]["body"]["in_progress"] is False
    assert message.status == "interrupted"
    assert message.in_progress is False


# ── Public read view ─────────────────────────────────────────────────────────────


def test_message_public_from_message_is_flat_and_non_recursive() -> None:
    """GIVEN a Message WHEN projected to MessagePublic THEN nested models are carried flat."""
    from datarobot_genai.application_utils.chat_history.models import MessagePublic

    message = Message(
        content="answer",
        emitter_type="agent",
        role="assistant",
        tool_calls=[ToolCall(name="search")],
        reasonings=[Reasoning(name="plan")],
    )

    public = MessagePublic.from_message(message)

    assert public.content == "answer"
    assert public.role == "assistant"
    assert public.timestamp == message.timestamp
    assert isinstance(public.tool_calls[0], ToolCall)
    assert public.tool_calls[0].name == "search"
    assert public.reasonings[0].name == "plan"


# ── Subclass extensibility ───────────────────────────────────────────────────────


async def test_message_subclass_persists_scalar_and_nested_field() -> None:
    """GIVEN a Message subclass with extra scalar+nested fields WHEN posted THEN both persist."""
    with respx.mock:
        captured: dict[str, Any] = {}
        respx.post(EVENTS_URL).mock(side_effect=echo_event(captured))

        message: RichMessage = await RichMessage.post(
            make_chat(),
            content="hi",
            emitter_type="agent",
            role="assistant",
            priority=5,
            attachments=[Attachment(filename="a.txt", size=3)],
        )

    # Extra fields are serialised into the event body...
    assert captured["body"]["body"]["priority"] == 5
    assert captured["body"]["body"]["attachments"] == [{"filename": "a.txt", "size": 3}]
    # ...and hydrated back into the typed subclass.
    assert message.priority == 5
    assert isinstance(message.attachments[0], Attachment)
    assert message.attachments[0].filename == "a.txt"


def test_chat_subclass_routes_extra_fields_to_metadata() -> None:
    """GIVEN a Chat subclass with extra fields WHEN serialised THEN both land in metadata."""
    payload = ProjectChat._to_wire_create(
        {
            "thread_id": "thread-9",
            "dedup_key": "dedup-9",
            "project_code": "PX-1",
            "labels": [Attachment(filename="spec", size=1)],
        }
    )

    assert payload["description"] == "//thread/thread-9/"
    assert payload["deduplicationKey"] == "dedup-9"
    assert payload["metadata"]["project_code"] == "PX-1"
    assert payload["metadata"]["labels"] == [{"filename": "spec", "size": 1}]


async def test_chat_subclass_round_trips_nested_field_via_respx() -> None:
    """GIVEN a Chat subclass WHEN created against a mocked API THEN the nested field hydrates."""
    with respx.mock:
        captured: dict[str, Any] = {}
        respx.post(SESSIONS_URL).mock(side_effect=echo_session(captured))

        chat: ProjectChat = await ProjectChat.post(
            make_space(),
            thread_id="thread-9",
            dedup_key="dedup-9",
            project_code="PX-1",
            labels=[Attachment(filename="spec", size=2)],
        )

    assert chat.project_code == "PX-1"
    assert isinstance(chat.labels[0], Attachment)
    assert chat.labels[0].filename == "spec"
    assert chat.labels[0].size == 2
