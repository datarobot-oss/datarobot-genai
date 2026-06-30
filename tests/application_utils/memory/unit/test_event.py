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

"""Unit tests for DREvent (event.py)."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from datarobot_genai.application_utils.memory import DRMemorySpace
from datarobot_genai.application_utils.memory import MemoryBadRequestError
from datarobot_genai.application_utils.memory import MemoryServiceClient
from datarobot_genai.application_utils.memory import MemoryVersionConflictError
from datarobot_genai.application_utils.memory._encoding import build_description
from tests.application_utils.memory.unit.conftest import PARTICIPANT
from tests.application_utils.memory.unit.conftest import SESSION_ID
from tests.application_utils.memory.unit.conftest import SPACE_ID
from tests.application_utils.memory.unit.conftest import SYSTEM_OID
from tests.application_utils.memory.unit.conftest import ChatMessage
from tests.application_utils.memory.unit.conftest import ChatSession

BASE = "https://app.datarobot.com/api/v2"
MEMORY_BASE = f"{BASE}/memory"
SESSIONS_URL = f"{MEMORY_BASE}/{SPACE_ID}/sessions/"
SESSION_URL = f"{SESSIONS_URL}{SESSION_ID}/"
EVENTS_URL = f"{SESSION_URL}events/"
EVENTS_BATCH_URL = f"{EVENTS_URL}batch/"


# ── Test helpers ──────────────────────────────────────────────────────────────


def _client() -> MemoryServiceClient:
    return MemoryServiceClient(
        endpoint=BASE,
        api_token="test-token",
        http_client=httpx.AsyncClient(),
    )


def _space_wire(space_id: str = SPACE_ID) -> dict:
    return {
        "memorySpaceId": space_id,
        "userId": "u1",
        "tenantId": "t1",
        "description": None,
        "deduplicationKey": None,
        "createdAt": "2026-06-30T00:00:00Z",
    }


def _session_wire(
    *,
    participant: str = PARTICIPANT,
    version: int = 1,
) -> dict:
    return {
        "id": SESSION_ID,
        "participants": [participant],
        "description": build_description("chat", ["acme", "billing"]),
        "deduplicationKey": "chat-001",
        "metadata": {"title": "Test", "rev": version},
        "lifecycleStrategies": [],
        "version": version,
        "createdAt": "2026-06-30T00:00:00Z",
    }


def _event_wire(
    *,
    seq_id: int = 0,
    content: str = "Hello!",
    emitter_type: str = "user",
    emitter_id: str | None = None,
    score: float = 0.9,
    created_at: str = "2026-06-30T00:00:01Z",
) -> dict:
    return {
        "sequenceId": seq_id,
        "createdAt": created_at,
        "eventType": "message",
        "emitterType": emitter_type,
        "emitterId": emitter_id,
        "body": {"content": content, "score": score},
    }


def _make_space(client: MemoryServiceClient | None = None) -> DRMemorySpace:
    return DRMemorySpace._from_wire(client or _client(), _space_wire())


def _make_session(
    space: DRMemorySpace | None = None,
    participant: str = PARTICIPANT,
) -> ChatSession:
    sp = space or _make_space()
    return ChatSession._from_wire(sp, _session_wire(participant=participant))  # type: ignore[return-value]


def _make_event(
    session: ChatSession | None = None,
    seq_id: int = 0,
    created_at: str = "2026-06-30T00:00:01Z",
) -> ChatMessage:
    sess = session or _make_session()
    return ChatMessage._from_wire(  # type: ignore[return-value]
        sess,
        _event_wire(seq_id=seq_id, created_at=created_at),
    )


# ── DREvent.post ──────────────────────────────────────────────────────────────


@respx.mock
async def test_post_creates_event_and_returns_instance() -> None:
    """GIVEN post(content, emitter_type, ...) WHEN POST /events/ THEN DREvent returned."""
    respx.post(EVENTS_URL).mock(return_value=httpx.Response(201, json=_event_wire()))
    session = _make_session()
    event = await ChatMessage.post(
        session,
        content="Hello!",
        emitter_type="user",
        emitter_id=PARTICIPANT,
        score=0.9,
    )
    assert event.sequence_id == 0
    assert event.content == "Hello!"
    assert event.score == 0.9


@respx.mock
async def test_post_sends_event_type_in_payload() -> None:
    """GIVEN __event_type__ = 'message' WHEN post THEN type='message' in wire body."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(201, json=_event_wire())

    respx.post(EVENTS_URL).mock(side_effect=_capture)
    session = _make_session()
    await ChatMessage.post(
        session,
        content="Hello!",
        emitter_type="agent",
    )
    assert captured["body"]["type"] == "message"


@respx.mock
async def test_post_sends_extra_body_fields() -> None:
    """GIVEN score=0.75 (extra body field) WHEN post THEN score present in body."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(201, json=_event_wire(score=0.75))

    respx.post(EVENTS_URL).mock(side_effect=_capture)
    session = _make_session()
    await ChatMessage.post(
        session,
        content="Hello!",
        emitter_type="agent",
        score=0.75,
    )
    assert captured["body"]["body"]["score"] == 0.75


@respx.mock
async def test_post_sends_emitter_in_payload() -> None:
    """GIVEN emitter_type='user', emitter_id=oid WHEN post THEN emitter dict in wire."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(201, json=_event_wire())

    respx.post(EVENTS_URL).mock(side_effect=_capture)
    session = _make_session()
    await ChatMessage.post(
        session,
        content="Hello!",
        emitter_type="user",
        emitter_id=PARTICIPANT,
    )
    assert captured["body"]["emitter"]["type"] == "user"
    assert captured["body"]["emitter"]["id"] == PARTICIPANT


def test_post_emitter_not_in_participants_raises_bad_request() -> None:
    """GIVEN user emitter not in session.participants WHEN post THEN MemoryBadRequestError."""
    import asyncio

    session = _make_session(participant=SYSTEM_OID)

    async def _run() -> None:
        await ChatMessage.post(
            session,
            content="Hello!",
            emitter_type="user",
            emitter_id=PARTICIPANT,  # PARTICIPANT is not in session.participants
        )

    with pytest.raises(MemoryBadRequestError):
        asyncio.get_event_loop().run_until_complete(_run())


# ── Body round-trip ───────────────────────────────────────────────────────────


def test_from_wire_round_trips_extra_body_fields() -> None:
    """GIVEN event wire with body.score WHEN _from_wire THEN event.score is set."""
    session = _make_session()
    event = ChatMessage._from_wire(session, _event_wire(score=0.85))  # type: ignore[assignment]
    assert event.score == 0.85  # type: ignore[attr-defined]
    assert event.content == "Hello!"
    assert event.emitter_type == "user"
    assert event.sequence_id == 0
    assert event.created_at == "2026-06-30T00:00:01Z"


def test_from_wire_emitter_id_is_none_when_absent() -> None:
    """GIVEN event wire without emitterId WHEN _from_wire THEN emitter_id is None."""
    session = _make_session()
    wire = _event_wire()
    wire["emitterId"] = None
    event = ChatMessage._from_wire(session, wire)  # type: ignore[assignment]
    assert event.emitter_id is None


def test_from_wire_sets_session_back_reference() -> None:
    """GIVEN _from_wire WHEN inspecting private attrs THEN _session is set correctly."""
    session = _make_session()
    event = ChatMessage._from_wire(session, _event_wire())  # type: ignore[assignment]
    assert event._session is session


# ── DREvent.post_batch ────────────────────────────────────────────────────────


@respx.mock
async def test_post_batch_sends_events_list_payload() -> None:
    """GIVEN two events WHEN post_batch THEN wire payload has events array."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        items = [_event_wire(seq_id=0), _event_wire(seq_id=1)]
        return httpx.Response(200, json={"items": items})

    respx.post(EVENTS_BATCH_URL).mock(side_effect=_capture)
    session = _make_session()
    events = await ChatMessage.post_batch(
        session,
        events=[
            {"content": "First", "emitter_type": "agent", "score": 0.1},
            {"content": "Second", "emitter_type": "agent", "score": 0.2},
        ],
    )
    assert len(events) == 2
    assert "events" in captured["body"]
    assert len(captured["body"]["events"]) == 2


@respx.mock
async def test_post_batch_up_to_200_events_is_accepted() -> None:
    """GIVEN 200 events WHEN post_batch THEN no error (at the batch boundary)."""
    items = [_event_wire(seq_id=i) for i in range(200)]
    respx.post(EVENTS_BATCH_URL).mock(return_value=httpx.Response(200, json={"items": items}))
    session = _make_session()
    batch = [{"content": f"msg-{i}", "emitter_type": "agent"} for i in range(200)]
    events = await ChatMessage.post_batch(session, events=batch)
    assert len(events) == 200


def test_post_batch_over_200_raises_value_error() -> None:
    """GIVEN 201 events WHEN post_batch THEN ValueError raised before HTTP call."""
    import asyncio

    session = _make_session()
    batch = [{"content": f"msg-{i}", "emitter_type": "agent"} for i in range(201)]

    async def _run() -> None:
        await ChatMessage.post_batch(session, events=batch)

    with pytest.raises(ValueError, match="at most 200"):
        asyncio.get_event_loop().run_until_complete(_run())


# ── DREvent.list ──────────────────────────────────────────────────────────────


@respx.mock
async def test_list_returns_events() -> None:
    """GIVEN GET /events/ returns events WHEN list THEN events with body fields."""
    respx.get(EVENTS_URL).mock(
        return_value=httpx.Response(
            200,
            json={"items": [_event_wire(seq_id=0), _event_wire(seq_id=1)], "total": 2},
        )
    )
    session = _make_session()
    events = await ChatMessage.list(session)
    assert len(events) == 2
    assert events[0].sequence_id == 0
    assert events[1].sequence_id == 1


@respx.mock
async def test_list_by_type_sends_event_type_param() -> None:
    """GIVEN list(type='message') WHEN GET THEN eventType=message query param sent."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["params"] = dict(req.url.params)
        return httpx.Response(200, json={"items": [], "total": 0})

    respx.get(EVENTS_URL).mock(side_effect=_capture)
    session = _make_session()
    await ChatMessage.list(session, type="message")
    assert captured["params"]["eventType"] == "message"


@respx.mock
async def test_list_sends_offset_and_limit() -> None:
    """GIVEN list(offset=10, limit=25) WHEN GET THEN offset + limit params sent."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["params"] = dict(req.url.params)
        return httpx.Response(200, json={"items": [], "total": 0})

    respx.get(EVENTS_URL).mock(side_effect=_capture)
    session = _make_session()
    await ChatMessage.list(session, offset=10, limit=25)
    assert captured["params"]["offset"] == "10"
    assert captured["params"]["limit"] == "25"


# ── DREvent.last ──────────────────────────────────────────────────────────────


@respx.mock
async def test_last_sends_last_n_param_without_offset() -> None:
    """GIVEN last(n=5) WHEN GET THEN lastN=5 sent without offset param."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["params"] = dict(req.url.params)
        return httpx.Response(200, json={"items": [], "total": 0})

    respx.get(EVENTS_URL).mock(side_effect=_capture)
    session = _make_session()
    await ChatMessage.last(session, n=5)
    assert captured["params"]["lastN"] == "5"
    assert "offset" not in captured["params"]


@respx.mock
async def test_last_with_type_filter_sends_event_type_param() -> None:
    """GIVEN last(n=10, type='message') WHEN GET THEN both lastN and eventType sent."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["params"] = dict(req.url.params)
        return httpx.Response(200, json={"items": [], "total": 0})

    respx.get(EVENTS_URL).mock(side_effect=_capture)
    session = _make_session()
    await ChatMessage.last(session, n=10, type="message")
    assert captured["params"]["lastN"] == "10"
    assert captured["params"]["eventType"] == "message"


# ── DREvent.patch ─────────────────────────────────────────────────────────────


@respx.mock
async def test_patch_sends_created_at_as_query_param() -> None:
    """GIVEN event with created_at token WHEN patch THEN createdAt query param sent."""
    captured: dict = {}
    seq_id = 0
    created_at = "2026-06-30T00:00:01Z"
    event_patch_url = f"{EVENTS_URL}{seq_id}/"

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["params"] = dict(req.url.params)
        return httpx.Response(200, json=_event_wire(seq_id=seq_id))

    respx.patch(event_patch_url).mock(side_effect=_capture)
    event = _make_event(seq_id=seq_id, created_at=created_at)
    await event.patch(content="Updated content")
    assert captured["params"]["createdAt"] == created_at


@respx.mock
async def test_patch_updates_event_in_place() -> None:
    """GIVEN patch(content='Updated') WHEN PATCH response THEN event.content updated."""
    seq_id = 0
    event_patch_url = f"{EVENTS_URL}{seq_id}/"
    respx.patch(event_patch_url).mock(
        return_value=httpx.Response(
            200,
            json=_event_wire(seq_id=seq_id, content="Updated content"),
        )
    )
    event = _make_event()
    await event.patch(content="Updated content")
    assert event.content == "Updated content"


@respx.mock
async def test_patch_updates_extra_body_field_in_place() -> None:
    """GIVEN patch(score=0.5) WHEN PATCH response THEN event.score updated."""
    seq_id = 0
    event_patch_url = f"{EVENTS_URL}{seq_id}/"
    respx.patch(event_patch_url).mock(
        return_value=httpx.Response(
            200,
            json=_event_wire(seq_id=seq_id, score=0.5),
        )
    )
    event = _make_event()
    await event.patch(score=0.5)
    assert event.score == 0.5  # type: ignore[attr-defined]


@respx.mock
async def test_patch_stale_created_at_raises_version_conflict_error() -> None:
    """GIVEN stale createdAt WHEN PATCH returns 422 event-version THEN VersionConflict."""
    seq_id = 0
    event_patch_url = f"{EVENTS_URL}{seq_id}/"
    respx.patch(event_patch_url).mock(
        return_value=httpx.Response(
            422,
            json={"detail": "Patch of incorrect version of event"},
        )
    )
    event = _make_event()
    with pytest.raises(MemoryVersionConflictError):
        await event.patch(content="Stale update")


def test_patch_with_no_fields_raises_value_error() -> None:
    """GIVEN patch() with no kwargs WHEN patch THEN ValueError raised."""
    import asyncio

    event = _make_event()

    async def _run() -> None:
        await event.patch()

    with pytest.raises(ValueError, match="at least one field"):
        asyncio.get_event_loop().run_until_complete(_run())


# ── DREvent.delete ────────────────────────────────────────────────────────────


@respx.mock
async def test_delete_sends_delete_request() -> None:
    """GIVEN an event WHEN delete() THEN DELETE .../events/{seq_id}/ is sent."""
    seq_id = 0
    event_delete_url = f"{EVENTS_URL}{seq_id}/"
    delete_route = respx.delete(event_delete_url).mock(return_value=httpx.Response(204))
    event = _make_event(seq_id=seq_id)
    await event.delete()
    assert delete_route.called


# ── DREvent.patch_batch ───────────────────────────────────────────────────────


@respx.mock
async def test_patch_batch_sends_events_with_sequence_ids() -> None:
    """GIVEN patch_batch(updates) WHEN PATCH /events/batch/ THEN sequenceIds in payload."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={"items": [_event_wire(seq_id=0), _event_wire(seq_id=1)]},
        )

    respx.patch(EVENTS_BATCH_URL).mock(side_effect=_capture)
    session = _make_session()
    event_a = _make_event(session=session, seq_id=0, created_at="2026-01-01T00:00:00Z")
    event_b = _make_event(session=session, seq_id=1, created_at="2026-01-01T00:00:01Z")
    await ChatMessage.patch_batch(
        session,
        updates=[
            (event_a, {"score": 0.9}),
            (event_b, {"content": "Updated"}),
        ],
    )
    assert "events" in captured["body"]
    seq_ids = [e["sequenceId"] for e in captured["body"]["events"]]
    assert 0 in seq_ids
    assert 1 in seq_ids


@respx.mock
async def test_patch_batch_sends_created_at_token_per_event() -> None:
    """GIVEN patch_batch with events having createdAt WHEN PATCH THEN tokens included."""
    captured: dict = {}
    created_at_a = "2026-01-01T00:00:00Z"

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={"items": [_event_wire(seq_id=0)]})

    respx.patch(EVENTS_BATCH_URL).mock(side_effect=_capture)
    session = _make_session()
    event_a = _make_event(session=session, seq_id=0, created_at=created_at_a)
    await ChatMessage.patch_batch(session, updates=[(event_a, {"score": 0.5})])
    event_payload = captured["body"]["events"][0]
    assert event_payload.get("createdAt") == created_at_a


def test_patch_batch_over_200_raises_value_error() -> None:
    """GIVEN 201 updates WHEN patch_batch THEN ValueError raised before HTTP call."""
    import asyncio

    session = _make_session()
    events = [_make_event(seq_id=i) for i in range(201)]
    updates = [(e, {"score": 0.5}) for e in events]

    async def _run() -> None:
        await ChatMessage.patch_batch(session, updates=updates)

    with pytest.raises(ValueError, match="at most 200"):
        asyncio.get_event_loop().run_until_complete(_run())


def test_patch_batch_item_with_no_fields_raises_value_error() -> None:
    """GIVEN patch_batch with one item having empty kwargs WHEN patch_batch THEN ValueError."""
    import asyncio

    session = _make_session()
    event_a = _make_event(session=session, seq_id=0)

    async def _run() -> None:
        await ChatMessage.patch_batch(session, updates=[(event_a, {})])

    with pytest.raises(ValueError):
        asyncio.get_event_loop().run_until_complete(_run())


# ── Unknown-field rejection ───────────────────────────────────────────────────


def test_post_unknown_field_raises_value_error() -> None:
    """GIVEN post(scoree=0.5) WHEN ChatMessage.post THEN ValueError naming the bad field."""
    import asyncio

    session = _make_session()

    async def _run() -> None:
        await ChatMessage.post(session, content="x", emitter_type="agent", scoree=0.5)

    with pytest.raises(ValueError, match="scoree"):
        asyncio.get_event_loop().run_until_complete(_run())


def test_post_batch_unknown_field_raises_value_error() -> None:
    """GIVEN post_batch with a typo'd field WHEN ChatMessage.post_batch THEN ValueError."""
    import asyncio

    session = _make_session()

    async def _run() -> None:
        await ChatMessage.post_batch(
            session,
            events=[{"content": "x", "emitter_type": "agent", "scoree": 0.5}],
        )

    with pytest.raises(ValueError, match="scoree"):
        asyncio.get_event_loop().run_until_complete(_run())


def test_patch_unknown_field_raises_value_error() -> None:
    """GIVEN patch(scoree=0.5) WHEN event.patch THEN ValueError naming the bad field."""
    import asyncio

    event = _make_event()

    async def _run() -> None:
        await event.patch(scoree=0.5)

    with pytest.raises(ValueError, match="scoree"):
        asyncio.get_event_loop().run_until_complete(_run())


def test_patch_batch_unknown_field_raises_value_error() -> None:
    """GIVEN patch_batch updates with a typo'd field WHEN ChatMessage.patch_batch THEN ValueError."""
    import asyncio

    session = _make_session()
    event = _make_event(session=session)

    async def _run() -> None:
        await ChatMessage.patch_batch(session, updates=[(event, {"scoree": 0.5})])

    with pytest.raises(ValueError, match="scoree"):
        asyncio.get_event_loop().run_until_complete(_run())


# ── Session-binding type check ────────────────────────────────────────────────


def test_event_session_type_is_bound_to_chat_session() -> None:
    """GIVEN class ChatMessage(DREvent, session=ChatSession) THEN __dr_session_type__ is set."""
    assert ChatMessage.__dr_session_type__ is ChatSession


def test_event_without_session_binding_has_no_session_type() -> None:
    """GIVEN a DREvent subclass with no session= arg THEN __dr_session_type__ is not set."""
    from datarobot_genai.application_utils.memory import DREvent

    class UnboundEvent(DREvent):
        pass

    assert not hasattr(UnboundEvent, "__dr_session_type__")
