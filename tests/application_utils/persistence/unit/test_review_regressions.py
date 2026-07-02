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

"""Regression tests for code-review fixes to the persistence ORM."""

from __future__ import annotations

import json
from typing import Annotated

import httpx
import pytest
import respx

from datarobot_genai.application_utils.persistence import DRDeduplicationKey
from datarobot_genai.application_utils.persistence import DREvent
from datarobot_genai.application_utils.persistence import DRMemoryConflictError
from datarobot_genai.application_utils.persistence import DRMemoryNotFoundError
from datarobot_genai.application_utils.persistence import DRMemoryServiceClient
from datarobot_genai.application_utils.persistence import DRMemorySpace
from datarobot_genai.application_utils.persistence import DRRangeKey
from datarobot_genai.application_utils.persistence import DRSession
from tests.application_utils.persistence.unit.conftest import PARTICIPANT
from tests.application_utils.persistence.unit.conftest import SESSION_ID
from tests.application_utils.persistence.unit.conftest import SPACE_ID
from tests.application_utils.persistence.unit.conftest import ChatMessage
from tests.application_utils.persistence.unit.conftest import ChatSession

BASE = "https://app.datarobot.com/api/v2"
MEMORY_BASE = f"{BASE}/memory"
SESSIONS_URL = f"{MEMORY_BASE}/{SPACE_ID}/sessions/"
SESSION_URL = f"{SESSIONS_URL}{SESSION_ID}/"
EVENT_URL = f"{SESSION_URL}events/5/"


def _client() -> DRMemoryServiceClient:
    return DRMemoryServiceClient(endpoint=BASE, api_token="t", http_client=httpx.AsyncClient())


def _space() -> DRMemorySpace:
    return DRMemorySpace._from_wire(
        _client(),
        {"memorySpaceId": SPACE_ID, "userId": "u", "tenantId": "t", "createdAt": ""},
    )


def _session_wire(**overrides: object) -> dict:
    wire: dict = {
        "id": SESSION_ID,
        "participants": [PARTICIPANT],
        "description": "//chat/acme/billing/",
        "deduplicationKey": "chat-001",
        "metadata": {"title": "Billing enquiry"},
        "version": 3,
        "createdAt": "2026-06-30T00:00:00Z",
    }
    wire.update(overrides)
    return wire


def _make_session() -> ChatSession:
    return ChatSession._from_wire(_space(), _session_wire())  # type: ignore[return-value]


# ── _routing: multiple markers on one field ───────────────────────────────────


def test_multiple_markers_on_one_field_raises_type_error() -> None:
    """GIVEN a field carrying two ORM markers WHEN routing is built THEN TypeError is raised."""

    class BadSession(DRSession):
        key: Annotated[str, DRRangeKey, DRDeduplicationKey]

    with pytest.raises(TypeError, match="multiple ORM markers"):
        BadSession._get_routing()


# ── session.patch: metadata / description gating ──────────────────────────────


@respx.mock
async def test_patch_metadata_only_does_not_send_description() -> None:
    """GIVEN a metadata-only patch WHEN PATCH is sent THEN the payload omits description."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json=_session_wire(version=4))

    respx.patch(SESSION_URL).mock(side_effect=_capture)
    session = _make_session()
    await session.patch(title="New title")
    assert "description" not in captured["body"]
    assert captured["body"]["metadata"]["title"] == "New title"


@respx.mock
async def test_patch_metadata_only_does_not_raise_when_range_field_unset() -> None:
    """GIVEN an undecodable description WHEN patching only metadata THEN no ValueError is raised."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json=_session_wire(version=4))

    respx.patch(SESSION_URL).mock(side_effect=_capture)
    # A description that does not decode for prefix "chat" leaves tenant/topic unset.
    session = ChatSession._from_wire(  # type: ignore[assignment]
        _space(), _session_wire(description="//other/x/")
    )
    await session.patch(title="New title")  # must not raise on unset range fields
    assert "description" not in captured["body"]


@respx.mock
async def test_patch_can_clear_metadata_field_with_none() -> None:
    """GIVEN an explicit None for a metadata field WHEN patching THEN it is sent (cleared)."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json=_session_wire(version=4))

    respx.patch(SESSION_URL).mock(side_effect=_capture)

    class NoteSession(DRSession):
        __description_prefix__ = "note"
        note: str | None = None

    wire = {
        "id": SESSION_ID,
        "participants": [PARTICIPANT],
        "metadata": {"note": "keep"},
        "version": 1,
        "createdAt": "",
    }
    session = NoteSession._from_wire(_space(), wire)  # type: ignore[assignment]
    await session.patch(note=None)
    assert captured["body"]["metadata"] == {"note": None}


# ── session._from_wire: null version + absent required field ───────────────────


def test_from_wire_tolerates_null_version() -> None:
    """GIVEN a wire dict with version=None WHEN _from_wire THEN version defaults to 1 (no crash)."""
    session = ChatSession._from_wire(_space(), _session_wire(version=None))  # type: ignore[assignment]
    assert session.version == 1


def test_from_wire_absent_required_metadata_field_is_none_not_attribute_error() -> None:
    """GIVEN metadata missing a required field WHEN _from_wire THEN access returns None."""
    session = ChatSession._from_wire(_space(), _session_wire(metadata={}))  # type: ignore[assignment]
    assert session.title is None


# ── session.get: exact dedup-key match ────────────────────────────────────────


@respx.mock
async def test_get_by_dedup_key_ignores_non_exact_matches() -> None:
    """GIVEN a non-exact match first WHEN get() by dedup key THEN the exact key wins."""
    respx.get(SESSIONS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    _session_wire(session_id="other", deduplicationKey="chat-0010"),
                    _session_wire(deduplicationKey="chat-001"),
                ],
                "total": 2,
            },
        )
    )
    session = await ChatSession.get(_space(), chat_id="chat-001")
    assert session.id == SESSION_ID


@respx.mock
async def test_get_by_dedup_key_raises_when_only_non_exact_matches() -> None:
    """GIVEN only non-exact matches WHEN get() THEN DRMemoryNotFoundError is raised."""
    respx.get(SESSIONS_URL).mock(
        return_value=httpx.Response(
            200,
            json={"items": [_session_wire(deduplicationKey="chat-0010")], "total": 1},
        )
    )
    with pytest.raises(DRMemoryNotFoundError):
        await ChatSession.get(_space(), chat_id="chat-001")


# ── session._to_wire_create: None range/dedup key rejected ─────────────────────


async def test_post_none_dedup_key_raises_value_error() -> None:
    """GIVEN chat_id=None WHEN post() THEN ValueError (not a literal 'None' key)."""
    with pytest.raises(ValueError, match="must not be None"):
        await ChatSession.post(_space(), tenant="acme", topic="billing", chat_id=None, title="t")


async def test_post_none_range_key_raises_value_error() -> None:
    """GIVEN tenant=None WHEN post() THEN ValueError (not a literal 'None' segment)."""
    with pytest.raises(ValueError, match="must not be None"):
        await ChatSession.post(_space(), tenant=None, topic="billing", chat_id="c1", title="t")


# ── event.patch: unpatched body fields preserved ──────────────────────────────


def _event_wire(**overrides: object) -> dict:
    wire: dict = {
        "sequenceId": 5,
        "createdAt": "2026-06-30T00:00:01Z",
        "emitterType": "agent",
        "emitterId": None,
        "body": {"content": "hi", "score": 0.9},
    }
    wire.update(overrides)
    return wire


class LabeledMessage(DREvent, session=ChatSession):
    """Event with two body fields to exercise partial patch."""

    __event_type__ = "message"

    score: float = 0.0
    label: str = ""


@respx.mock
async def test_event_patch_preserves_unpatched_body_fields() -> None:
    """GIVEN an event with score+label WHEN patching only score THEN label is preserved."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json=_event_wire(body={"content": "hi"}))

    respx.patch(EVENT_URL).mock(side_effect=_capture)
    session = _make_session()
    event = LabeledMessage._from_wire(
        session, _event_wire(body={"content": "hi", "score": 0.9, "label": "important"})
    )
    await event.patch(score=0.5)
    assert captured["body"]["body"]["label"] == "important"
    assert captured["body"]["body"]["score"] == 0.5


def test_event_from_wire_absent_required_field_is_none() -> None:
    """GIVEN an event body missing a required field WHEN _from_wire THEN access returns None."""

    class RequiredScore(DREvent, session=ChatSession):
        score: float

    event = RequiredScore._from_wire(_make_session(), _event_wire(body={"content": "hi"}))
    assert event.score is None  # type: ignore[attr-defined]


def test_event_from_wire_tolerates_null_sequence_id() -> None:
    """GIVEN sequenceId=None WHEN _from_wire THEN sequence_id defaults to -1 (no crash)."""
    event = ChatMessage._from_wire(_make_session(), _event_wire(sequenceId=None))
    assert event.sequence_id == -1


# ── event.post_batch: required fields enforced ────────────────────────────────


async def test_post_batch_missing_content_raises_value_error() -> None:
    """GIVEN a batch item missing content WHEN post_batch THEN ValueError (not KeyError)."""
    with pytest.raises(ValueError, match="missing required field"):
        await ChatMessage.post_batch(
            _make_session(), events=[{"emitter_type": "agent", "score": 0.5}]
        )


# ── _client: 409 dedup detection robust to errorName drift ────────────────────


@respx.mock
async def test_space_post_adopts_on_409_with_drifted_error_name() -> None:
    """GIVEN a 409 with existing id but a drifted errorName WHEN post THEN it still adopts."""
    existing = "eeeeeeee-0000-0000-0000-000000000009"
    respx.post(f"{MEMORY_BASE}/new/").mock(
        return_value=httpx.Response(
            409,
            json={"errorName": "DuplicateKey", "detail": "dup", "existingMemorySpaceId": existing},
        )
    )
    respx.get(f"{MEMORY_BASE}/{existing}/").mock(
        return_value=httpx.Response(
            200,
            json={"memorySpaceId": existing, "userId": "u", "tenantId": "t", "createdAt": ""},
        )
    )
    space = await DRMemorySpace.post(_client(), deduplication_key="k")
    assert space.id == existing


@respx.mock
async def test_client_populates_location_from_header() -> None:
    """GIVEN a 409 with the pointer only in the Location header WHEN raised THEN location is set."""
    location = f"/{SPACE_ID}/sessions/xyz/"
    respx.post(SESSIONS_URL).mock(
        return_value=httpx.Response(
            409,
            headers={"Location": location},
            json={"errorName": "SessionDeduplicationConflict", "detail": "dup"},
        )
    )
    with pytest.raises(DRMemoryConflictError) as exc:
        await ChatSession.post(_space(), tenant="acme", topic="billing", chat_id="c1", title="t")
    assert exc.value.location == location
