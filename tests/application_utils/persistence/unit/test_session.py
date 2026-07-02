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

"""Unit tests for DRSession (session.py)."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from datarobot_genai.application_utils.persistence import DRMemoryNotFoundError
from datarobot_genai.application_utils.persistence import DRMemoryServiceClient
from datarobot_genai.application_utils.persistence import DRMemorySpace
from datarobot_genai.application_utils.persistence import DRMemoryVersionConflictError
from datarobot_genai.application_utils.persistence._encoding import build_description
from datarobot_genai.application_utils.persistence.markers import SYSTEM_PARTICIPANT
from tests.application_utils.persistence.unit.conftest import PARTICIPANT
from tests.application_utils.persistence.unit.conftest import SESSION_ID
from tests.application_utils.persistence.unit.conftest import SPACE_ID
from tests.application_utils.persistence.unit.conftest import SYSTEM_OID
from tests.application_utils.persistence.unit.conftest import ChatSession

BASE = "https://app.datarobot.com/api/v2"
MEMORY_BASE = f"{BASE}/memory"
SESSIONS_URL = f"{MEMORY_BASE}/{SPACE_ID}/sessions/"
SESSION_URL = f"{SESSIONS_URL}{SESSION_ID}/"


# ── Test helpers ──────────────────────────────────────────────────────────────


def _client() -> DRMemoryServiceClient:
    return DRMemoryServiceClient(
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
    session_id: str = SESSION_ID,
    participant: str = PARTICIPANT,
    dedup_key: str = "chat-001",
    tenant: str = "acme",
    topic: str = "billing",
    title: str = "Billing enquiry",
    version: int = 1,
) -> dict:
    return {
        "id": session_id,
        "participants": [participant],
        "description": build_description("chat", [tenant, topic]),
        "deduplicationKey": dedup_key,
        "metadata": {"title": title, "rev": version},
        "lifecycleStrategies": [],
        "version": version,
        "createdAt": "2026-06-30T00:00:00Z",
    }


def _make_space(client: DRMemoryServiceClient | None = None) -> DRMemorySpace:
    return DRMemorySpace._from_wire(client or _client(), _space_wire())


def _make_session(space: DRMemorySpace | None = None) -> ChatSession:
    sp = space or _make_space()
    return ChatSession._from_wire(sp, _session_wire())  # type: ignore[return-value]


# ── DRSession.post ────────────────────────────────────────────────────────────


@respx.mock
async def test_post_creates_session_with_system_participant() -> None:
    """GIVEN no participants kwarg WHEN post THEN SYSTEM_PARTICIPANT is used."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(201, json=_session_wire())

    respx.post(SESSIONS_URL).mock(side_effect=_capture)
    space = _make_space()
    session = await ChatSession.post(
        space,
        tenant="acme",
        topic="billing",
        chat_id="chat-001",
        title="Billing enquiry",
        rev=1,
    )
    assert session.id == SESSION_ID
    assert SYSTEM_OID in captured["body"]["participants"]


@respx.mock
async def test_post_uses_explicit_participant() -> None:
    """GIVEN participants=[oid] WHEN post THEN the explicit participant is sent."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(201, json=_session_wire(participant=PARTICIPANT))

    respx.post(SESSIONS_URL).mock(side_effect=_capture)
    space = _make_space()
    await ChatSession.post(
        space,
        participants=[PARTICIPANT],
        tenant="acme",
        topic="billing",
        chat_id="chat-001",
        title="Test",
        rev=1,
    )
    # The request payload must carry the explicit participant. The response echo is not
    # evidence of what was sent, so assert on the captured request body.
    assert captured["body"]["participants"] == [PARTICIPANT]


@respx.mock
async def test_post_adopts_existing_session_on_409_dedup() -> None:
    """GIVEN POST returns 409 dedup WHEN post THEN session is fetched by existing_id."""
    existing_id = "cccc-0001"
    respx.post(SESSIONS_URL).mock(
        return_value=httpx.Response(
            409,
            json={
                "errorName": "SessionDeduplicationConflict",
                "detail": "Duplicate",
                "deduplicationKey": "chat-001",
                "existingSessionId": existing_id,
                "existingSessionUrl": f"/{SPACE_ID}/sessions/{existing_id}/",
            },
        )
    )
    respx.get(f"{SESSIONS_URL}{existing_id}/").mock(
        return_value=httpx.Response(200, json=_session_wire(session_id=existing_id))
    )
    space = _make_space()
    session = await ChatSession.post(
        space,
        tenant="acme",
        topic="billing",
        chat_id="chat-001",
        title="Test",
        rev=1,
    )
    assert session.id == existing_id


@respx.mock
async def test_post_sends_dedup_key_in_payload() -> None:
    """GIVEN a chat_id (DRDeduplicationKey) WHEN post THEN deduplicationKey in wire body."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(201, json=_session_wire())

    respx.post(SESSIONS_URL).mock(side_effect=_capture)
    space = _make_space()
    await ChatSession.post(
        space,
        tenant="acme",
        topic="billing",
        chat_id="my-chat-001",
        title="Test",
        rev=1,
    )
    assert captured["body"]["deduplicationKey"] == "my-chat-001"


@respx.mock
async def test_post_encodes_range_keys_as_description() -> None:
    """GIVEN range keys tenant + topic WHEN post THEN description is encoded correctly."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(201, json=_session_wire())

    respx.post(SESSIONS_URL).mock(side_effect=_capture)
    space = _make_space()
    await ChatSession.post(
        space,
        tenant="acme",
        topic="billing",
        chat_id="chat-001",
        title="Test",
        rev=1,
    )
    assert captured["body"]["description"] == "//chat/acme/billing/"


@respx.mock
async def test_post_puts_metadata_fields_in_metadata() -> None:
    """GIVEN title (metadata field) WHEN post THEN metadata dict contains title."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(201, json=_session_wire())

    respx.post(SESSIONS_URL).mock(side_effect=_capture)
    space = _make_space()
    await ChatSession.post(
        space,
        tenant="acme",
        topic="billing",
        chat_id="chat-001",
        title="My Title",
        rev=1,
    )
    assert "metadata" in captured["body"]
    assert captured["body"]["metadata"]["title"] == "My Title"


# ── DRSession.get ─────────────────────────────────────────────────────────────


@respx.mock
async def test_get_by_id_fetches_correct_session() -> None:
    """GIVEN get(id=...) WHEN GET /{id}/ THEN returns session with correct id."""
    respx.get(SESSION_URL).mock(return_value=httpx.Response(200, json=_session_wire()))
    space = _make_space()
    session = await ChatSession.get(space, id=SESSION_ID)
    assert session.id == SESSION_ID
    assert session.version == 1


@respx.mock
async def test_get_by_dedup_key_uses_list_endpoint() -> None:
    """GIVEN get(chat_id='chat-001') WHEN GET THEN deduplicationKey query param sent."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["params"] = dict(req.url.params)
        return httpx.Response(200, json={"items": [_session_wire()], "total": 1})

    respx.get(SESSIONS_URL).mock(side_effect=_capture)
    space = _make_space()
    session = await ChatSession.get(space, chat_id="chat-001")
    assert session.id == SESSION_ID
    assert captured["params"]["deduplicationKey"] == "chat-001"


@respx.mock
async def test_get_by_dedup_key_raises_not_found_when_empty() -> None:
    """GIVEN dedup key with no match WHEN get(chat_id=...) THEN DRMemoryNotFoundError."""
    respx.get(SESSIONS_URL).mock(return_value=httpx.Response(200, json={"items": [], "total": 0}))
    space = _make_space()
    with pytest.raises(DRMemoryNotFoundError):
        await ChatSession.get(space, chat_id="nonexistent")


def test_get_without_id_or_dedup_key_raises_value_error() -> None:
    """GIVEN get() with no positional id and no dedup kwarg THEN ValueError."""
    import asyncio

    space = _make_space()

    async def _run() -> None:
        await ChatSession.get(space)

    with pytest.raises(ValueError, match="get\\(\\) requires"):
        asyncio.get_event_loop().run_until_complete(_run())


# ── DRSession.list ────────────────────────────────────────────────────────────


@respx.mock
async def test_list_returns_all_sessions() -> None:
    """GIVEN GET sessions/ WHEN list THEN all items returned."""
    respx.get(SESSIONS_URL).mock(
        return_value=httpx.Response(
            200,
            json={"items": [_session_wire(), _session_wire(session_id="other")], "total": 2},
        )
    )
    space = _make_space()
    sessions = await ChatSession.list(space)
    assert len(sessions) == 2


@respx.mock
async def test_list_with_range_key_sends_encoded_description() -> None:
    """GIVEN list(tenant='acme') WHEN GET THEN description=//chat/acme/ query param."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["params"] = dict(req.url.params)
        return httpx.Response(200, json={"items": [], "total": 0})

    respx.get(SESSIONS_URL).mock(side_effect=_capture)
    space = _make_space()
    await ChatSession.list(space, tenant="acme")
    assert captured["params"]["description"] == "//chat/acme/"


@respx.mock
async def test_list_with_two_range_keys_sends_full_prefix() -> None:
    """GIVEN list(tenant='acme', topic='billing') WHEN GET THEN full encoded description."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["params"] = dict(req.url.params)
        return httpx.Response(200, json={"items": [], "total": 0})

    respx.get(SESSIONS_URL).mock(side_effect=_capture)
    space = _make_space()
    await ChatSession.list(space, tenant="acme", topic="billing")
    assert captured["params"]["description"] == "//chat/acme/billing/"


@respx.mock
async def test_list_with_participant_sends_participants_param() -> None:
    """GIVEN list(participant=oid) WHEN GET THEN participants query param set."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["params"] = dict(req.url.params)
        return httpx.Response(200, json={"items": [], "total": 0})

    respx.get(SESSIONS_URL).mock(side_effect=_capture)
    space = _make_space()
    await ChatSession.list(space, participant=PARTICIPANT)
    assert captured["params"]["participants"] == PARTICIPANT


@respx.mock
async def test_list_combined_range_and_participant() -> None:
    """GIVEN list(tenant='acme', participant=oid) WHEN GET THEN both params present."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["params"] = dict(req.url.params)
        return httpx.Response(200, json={"items": [], "total": 0})

    respx.get(SESSIONS_URL).mock(side_effect=_capture)
    space = _make_space()
    await ChatSession.list(space, tenant="acme", participant=PARTICIPANT)
    assert "description" in captured["params"]
    assert "participants" in captured["params"]


def test_list_non_leading_range_key_raises_value_error() -> None:
    """GIVEN list(topic='billing') without tenant WHEN list THEN ValueError (gaps not allowed)."""
    import asyncio

    space = _make_space()

    async def _run() -> None:
        await ChatSession.list(space, topic="billing")

    with pytest.raises(ValueError, match="contiguous leading prefix"):
        asyncio.get_event_loop().run_until_complete(_run())


async def test_list_empty_range_key_value_raises_value_error() -> None:
    """GIVEN list(tenant='') WHEN list THEN ValueError (empty range key, like post/patch)."""
    space = _make_space()
    with pytest.raises(ValueError, match="must not be empty"):
        await ChatSession.list(space, tenant="")


@respx.mock
async def test_list_paginates_when_total_field_absent() -> None:
    """GIVEN paginated responses that omit 'total' WHEN list THEN all pages are still fetched."""
    call_count = 0

    def _paginate(req: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        offset = int(dict(req.url.params).get("offset", 0))
        if offset == 0:
            # Full page (== limit) with NO 'total' key — must not short-circuit.
            items = [_session_wire(session_id=f"s-{i}") for i in range(100)]
            return httpx.Response(200, json={"items": items})
        # Short page (< limit) signals the last page.
        items = [_session_wire(session_id=f"s-{i}") for i in range(100, 130)]
        return httpx.Response(200, json={"items": items})

    respx.get(SESSIONS_URL).mock(side_effect=_paginate)
    space = _make_space()
    sessions = await ChatSession.list(space)
    assert len(sessions) == 130
    assert call_count == 2


@respx.mock
async def test_list_auto_paginates_when_total_exceeds_limit() -> None:
    """GIVEN total=150 with limit=100 WHEN list THEN two requests made."""
    call_count = 0

    def _paginate(req: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        params = dict(req.url.params)
        offset = int(params.get("offset", 0))
        if offset == 0:
            items = [_session_wire(session_id=f"sess-{i}") for i in range(100)]
            return httpx.Response(200, json={"items": items, "total": 150})
        else:
            items = [_session_wire(session_id=f"sess-{i}") for i in range(100, 150)]
            return httpx.Response(200, json={"items": items, "total": 150})

    respx.get(SESSIONS_URL).mock(side_effect=_paginate)
    space = _make_space()
    sessions = await ChatSession.list(space)
    assert len(sessions) == 150
    assert call_count == 2


# ── DRSession.patch ───────────────────────────────────────────────────────────


@respx.mock
async def test_patch_sends_if_match_header_with_current_version() -> None:
    """GIVEN session at version=3 WHEN patch THEN If-Match: 3 header sent."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["if_match"] = req.headers.get("if-match")
        return httpx.Response(200, json=_session_wire(version=4))

    respx.patch(SESSION_URL).mock(side_effect=_capture)
    session = _make_session()
    # Override version to 3
    session._version = 3
    await session.patch(title="New title")
    assert captured["if_match"] == "3"


@respx.mock
async def test_patch_updates_instance_from_response() -> None:
    """GIVEN patch(title='X') WHEN PATCH response THEN session.title updated."""
    respx.patch(SESSION_URL).mock(
        return_value=httpx.Response(200, json=_session_wire(title="Updated", version=2))
    )
    session = _make_session()
    await session.patch(title="Updated")
    assert session.title == "Updated"
    assert session.version == 2


@respx.mock
async def test_patch_updates_range_key_in_description() -> None:
    """GIVEN patch(topic='support') WHEN PATCH THEN description is rebuilt and updated."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json=_session_wire(topic="support"))

    respx.patch(SESSION_URL).mock(side_effect=_capture)
    session = _make_session()
    await session.patch(topic="support")
    # Description should be rebuilt with updated topic
    assert "description" in captured["body"]
    assert "support" in captured["body"]["description"]


@respx.mock
async def test_patch_stale_version_raises_version_conflict_error() -> None:
    """GIVEN stale If-Match WHEN PATCH returns 409 THEN DRMemoryVersionConflictError."""
    respx.patch(SESSION_URL).mock(
        return_value=httpx.Response(
            409, json={"detail": "Session version mismatch: expected 1, current 3"}
        )
    )
    session = _make_session()
    with pytest.raises(DRMemoryVersionConflictError):
        await session.patch(title="Too late")


def test_patch_with_unsupported_kwarg_raises_value_error() -> None:
    """GIVEN patch(participants=[...]) WHEN patch THEN ValueError (reserved field)."""
    import asyncio

    session = _make_session()

    async def _run() -> None:
        await session.patch(participants=["new_participant"])

    with pytest.raises(ValueError):
        asyncio.get_event_loop().run_until_complete(_run())


# ── DRSession.delete ──────────────────────────────────────────────────────────


@respx.mock
async def test_delete_sends_delete_request() -> None:
    """GIVEN a session WHEN delete() THEN DELETE /{space}/{session}/ is sent."""
    delete_route = respx.delete(SESSION_URL).mock(return_value=httpx.Response(204))
    session = _make_session()
    await session.delete()
    assert delete_route.called


# ── _from_wire field mapping ──────────────────────────────────────────────────


def test_from_wire_populates_all_fields() -> None:
    """GIVEN a session wire dict WHEN _from_wire THEN all fields are correctly mapped."""
    space = _make_space()
    session = ChatSession._from_wire(  # type: ignore[assignment]
        space,
        _session_wire(
            tenant="acme",
            topic="billing",
            dedup_key="chat-xyz",
            title="My Chat",
            version=5,
        ),
    )
    assert session.id == SESSION_ID
    assert session.version == 5
    assert session.tenant == "acme"
    assert session.topic == "billing"
    assert session.chat_id == "chat-xyz"
    assert session.title == "My Chat"
    assert session.rev == 5  # DRConcurrencyField mirrors version


def test_from_wire_sets_default_participants_when_missing() -> None:
    """GIVEN wire dict without participants WHEN _from_wire THEN defaults to system sentinel."""
    space = _make_space()
    wire = _session_wire()
    del wire["participants"]
    session = ChatSession._from_wire(space, wire)  # type: ignore[assignment]
    assert session.participants == [SYSTEM_PARTICIPANT]


def test_from_wire_preserves_participants() -> None:
    """GIVEN wire dict with explicit participant WHEN _from_wire THEN participant preserved."""
    space = _make_space()
    session = ChatSession._from_wire(  # type: ignore[assignment]
        space, _session_wire(participant=PARTICIPANT)
    )
    assert session.participants == [PARTICIPANT]


def test_from_wire_sets_space_back_reference() -> None:
    """GIVEN _from_wire WHEN inspecting private attrs THEN _space is set correctly."""
    space = _make_space()
    session = ChatSession._from_wire(space, _session_wire())  # type: ignore[assignment]
    assert session._space is space


# ── Metadata round-trip via encoding ─────────────────────────────────────────


def test_metadata_title_round_trips() -> None:
    """GIVEN title in metadata WHEN _from_wire THEN title accessible via session.title."""
    space = _make_space()
    session = ChatSession._from_wire(space, _session_wire(title="Round-trip title"))  # type: ignore[assignment]
    assert session.title == "Round-trip title"


def test_range_key_encoding_round_trips() -> None:
    """GIVEN tenant/topic WHEN build_description THEN parse_description recovers values."""
    from datarobot_genai.application_utils.persistence._encoding import parse_description

    desc = build_description("chat", ["acme", "billing"])
    values = parse_description("chat", desc, 2)
    assert values == ["acme", "billing"]


# ── Unknown-field rejection ───────────────────────────────────────────────────


def test_post_unknown_field_raises_value_error() -> None:
    """GIVEN post(titel='x') WHEN ChatSession.post THEN ValueError naming the bad field."""
    import asyncio

    space = _make_space()

    async def _run() -> None:
        await ChatSession.post(
            space,
            tenant="acme",
            topic="billing",
            chat_id="c",
            titel="typo",  # should be 'title'
        )

    with pytest.raises(ValueError, match="titel"):
        asyncio.get_event_loop().run_until_complete(_run())


def test_get_unknown_kwarg_raises_value_error() -> None:
    """GIVEN get(chat_idd='x') WHEN ChatSession.get THEN ValueError naming the bad field."""
    import asyncio

    space = _make_space()

    async def _run() -> None:
        await ChatSession.get(space, chat_idd="billing-chat-001")  # typo for 'chat_id'

    with pytest.raises(ValueError, match="chat_idd"):
        asyncio.get_event_loop().run_until_complete(_run())


def test_get_unknown_kwarg_with_id_raises_value_error() -> None:
    """GIVEN get(id=..., titel='x') WHEN ChatSession.get THEN ValueError before HTTP call."""
    import asyncio

    space = _make_space()

    async def _run() -> None:
        await ChatSession.get(space, id="some-uuid", titel="typo")

    with pytest.raises(ValueError, match="titel"):
        asyncio.get_event_loop().run_until_complete(_run())
