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

"""Integration tests for the Memory Service Light ORM.

These tests run against a live DataRobot Memory Service endpoint.  They are
**skipped by default** and require three environment variables to be set:

.. code-block:: bash

    export DATAROBOT_ENDPOINT="https://app.datarobot.com/api/v2"
    export DATAROBOT_API_TOKEN="<your-token>"
    export DR_MEMORY_LIVE_INTEGRATION="1"

    uv run pytest tests/application_utils/memory/integration -vv

The tests clean up after themselves (deleting created spaces) and are designed
to be idempotent so they can be re-run.
"""

from __future__ import annotations

import os
import uuid
from typing import Annotated

import pytest

from datarobot_genai.application_utils.memory import DRConcurrencyField
from datarobot_genai.application_utils.memory import DRDeduplicationKey
from datarobot_genai.application_utils.memory import DREvent
from datarobot_genai.application_utils.memory import DRMemorySpace
from datarobot_genai.application_utils.memory import DRRangeKey
from datarobot_genai.application_utils.memory import DRSession
from datarobot_genai.application_utils.memory import MemoryServiceClient
from datarobot_genai.application_utils.memory import MemoryVersionConflictError

# ── Integration opt-in guard ──────────────────────────────────────────────────

_LIVE = (
    os.environ.get("DATAROBOT_ENDPOINT")
    and os.environ.get("DATAROBOT_API_TOKEN")
    and os.environ.get("DR_MEMORY_LIVE_INTEGRATION", "").lower() in {"1", "true", "yes"}
)

pytestmark = pytest.mark.integration

_SKIP_REASON = (
    "Live integration tests require DATAROBOT_ENDPOINT, DATAROBOT_API_TOKEN, and "
    "DR_MEMORY_LIVE_INTEGRATION=1 to be set."
)

skip_unless_live = pytest.mark.skipif(not _LIVE, reason=_SKIP_REASON)

# ── Domain models used across all scenarios ───────────────────────────────────

PARTICIPANT_OID = "aabbccddeeff001122334455"  # 24-hex valid ObjectId for testing


class ChatSession(DRSession):
    """ORM session model used in integration tests."""

    __description_prefix__ = "it-chat"

    tenant: Annotated[str, DRRangeKey]
    topic: Annotated[str, DRRangeKey]
    chat_id: Annotated[str, DRDeduplicationKey]
    rev: Annotated[int, DRConcurrencyField]
    title: str = ""


class ChatMessage(DREvent, session=ChatSession):
    """ORM event model used in integration tests."""

    __event_type__ = "message"

    score: float = 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────


def _unique(prefix: str = "") -> str:
    """Return a short unique string for test isolation."""
    return f"{prefix}{uuid.uuid4().hex[:8]}"


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
async def client() -> MemoryServiceClient:  # type: ignore[return]
    """Return a ``MemoryServiceClient`` configured from environment variables."""
    async with MemoryServiceClient() as c:
        yield c


@pytest.fixture
async def space(client: MemoryServiceClient) -> DRMemorySpace:  # type: ignore[return]
    """Create a unique test memory space; delete after the test."""
    key = _unique("it-space-")
    sp = await DRMemorySpace.post(
        client,
        description="Integration test space",
        deduplication_key=key,
    )
    yield sp
    try:
        await sp.delete()
    except Exception:
        pass


# ── Scenario 1: Space lifecycle ───────────────────────────────────────────────


@skip_unless_live
async def test_space_lifecycle(client: MemoryServiceClient) -> None:
    """Create, get, list, patch, and delete a memory space."""
    key = _unique("lifecycle-")
    # Create
    sp = await DRMemorySpace.post(client, description="Initial desc", deduplication_key=key)
    assert sp.id
    assert sp.deduplication_key == key

    # Get by id
    sp2 = await DRMemorySpace.get(client, sp.id)
    assert sp2.id == sp.id

    # List — should appear
    spaces = await DRMemorySpace.list(client, deduplication_key=key)
    assert any(s.id == sp.id for s in spaces)

    # Patch description
    await sp.patch(description="Updated description")
    assert sp.description == "Updated description"

    # Delete
    await sp.delete()

    # After delete, should not appear in list
    spaces_after = await DRMemorySpace.list(client, deduplication_key=key)
    assert not any(s.id == sp.id for s in spaces_after)


# ── Scenario 2: Session round-trip ───────────────────────────────────────────


@skip_unless_live
async def test_session_round_trip(space: DRMemorySpace) -> None:
    """Post a session; fields survive the wire round-trip."""
    chat_id = _unique("sess-")
    session = await ChatSession.post(
        space,
        tenant="acme",
        topic="billing",
        chat_id=chat_id,
        title="Integration test chat",
        rev=1,
    )
    assert session.id
    assert session.tenant == "acme"
    assert session.topic == "billing"
    assert session.chat_id == chat_id
    assert session.title == "Integration test chat"

    # Get by id
    fetched = await ChatSession.get(space, id=session.id)
    assert fetched.id == session.id
    assert fetched.tenant == "acme"
    assert fetched.title == "Integration test chat"

    # Get by dedup key
    by_key = await ChatSession.get(space, chat_id=chat_id)
    assert by_key.id == session.id

    # Cleanup
    await session.delete()


@skip_unless_live
async def test_session_with_explicit_participant(space: DRMemorySpace) -> None:
    """Post a session scoped to a specific participant; participant survives the wire."""
    session = await ChatSession.post(
        space,
        participants=[PARTICIPANT_OID],
        tenant="acme",
        topic="support",
        chat_id=_unique("explicit-"),
        title="Explicit participant test",
        rev=1,
    )
    assert session.participants == [PARTICIPANT_OID]
    await session.delete()


# ── Scenario 3: Idempotent create (dedup key) ─────────────────────────────────


@skip_unless_live
async def test_idempotent_create(space: DRMemorySpace) -> None:
    """Creating a session with the same dedup key twice returns the same session."""
    key = _unique("dedup-")
    first = await ChatSession.post(
        space,
        tenant="acme",
        topic="billing",
        chat_id=key,
        title="First create",
        rev=1,
    )
    second = await ChatSession.post(
        space,
        tenant="acme",
        topic="billing",
        chat_id=key,
        title="Second create",
        rev=1,
    )
    # Both should have the same server-assigned id
    assert first.id == second.id
    await first.delete()


# ── Scenario 4: Range-key prefix queries ─────────────────────────────────────


@skip_unless_live
async def test_range_key_prefix_queries(space: DRMemorySpace) -> None:
    """Sessions under different topics; list by leading range-key prefix."""
    tenant = _unique("t-")
    s1 = await ChatSession.post(
        space, tenant=tenant, topic="billing", chat_id=_unique("s1-"), title="Billing", rev=1
    )
    s2 = await ChatSession.post(
        space, tenant=tenant, topic="support", chat_id=_unique("s2-"), title="Support", rev=1
    )
    # Query by tenant only → both sessions
    both = await ChatSession.list(space, tenant=tenant)
    ids = {s.id for s in both}
    assert s1.id in ids
    assert s2.id in ids

    # Query by tenant + topic=billing → only s1
    billing = await ChatSession.list(space, tenant=tenant, topic="billing")
    assert any(s.id == s1.id for s in billing)
    assert not any(s.id == s2.id for s in billing)

    # Partial segment (only part of "billing") should not match s1
    partial = await ChatSession.list(space, tenant=tenant, topic="bill")
    assert not any(s.id == s1.id for s in partial)

    await s1.delete()
    await s2.delete()


# ── Scenario 5: Participant scoping ──────────────────────────────────────────


@skip_unless_live
async def test_participant_scoping(space: DRMemorySpace) -> None:
    """Sessions for different participants; list filters correctly."""
    tenant = _unique("p-")
    sa = await ChatSession.post(
        space,
        participants=[PARTICIPANT_OID],
        tenant=tenant,
        topic="a",
        chat_id=_unique("pa-"),
        title="User A",
        rev=1,
    )
    sb = await ChatSession.post(
        space,
        tenant=tenant,
        topic="b",
        chat_id=_unique("pb-"),
        title="System",
        rev=1,
    )
    # Filter by participant → only sa
    user_sessions = await ChatSession.list(space, participant=PARTICIPANT_OID, tenant=tenant)
    assert any(s.id == sa.id for s in user_sessions)
    assert not any(s.id == sb.id for s in user_sessions)

    await sa.delete()
    await sb.delete()


# ── Scenario 6: Optimistic concurrency (session) ─────────────────────────────


@skip_unless_live
async def test_optimistic_concurrency_session(space: DRMemorySpace) -> None:
    """Patching a session bumps version; a second patch from a stale copy raises."""
    session = await ChatSession.post(
        space,
        tenant="acme",
        topic="billing",
        chat_id=_unique("occ-"),
        title="Original",
        rev=1,
    )
    # First patch bumps version
    await session.patch(title="Updated")
    v1 = session.version
    assert v1 > 1

    # Stale copy at old version — should raise
    stale = await ChatSession.get(space, id=session.id)
    stale._version = 0  # force stale version
    with pytest.raises(MemoryVersionConflictError):
        await stale.patch(title="Stale update")

    await session.delete()


# ── Scenario 7: Event log ─────────────────────────────────────────────────────


@skip_unless_live
async def test_event_log(space: DRMemorySpace) -> None:
    """Post events, list, last(n), patch, stale-patch, delete."""
    session = await ChatSession.post(
        space,
        participants=[PARTICIPANT_OID],
        tenant="acme",
        topic="events",
        chat_id=_unique("ev-"),
        title="Event test",
        rev=1,
    )

    # Single post
    msg = await ChatMessage.post(
        session,
        content="Hello!",
        emitter_type="user",
        emitter_id=PARTICIPANT_OID,
        score=0.9,
    )
    assert msg.sequence_id >= 0
    assert msg.content == "Hello!"
    assert msg.score == 0.9  # type: ignore[attr-defined]

    # Batch post
    batch = await ChatMessage.post_batch(
        session,
        events=[
            {"content": "Batch one", "emitter_type": "agent", "score": 0.5},
            {"content": "Batch two", "emitter_type": "agent", "score": 0.6},
        ],
    )
    assert len(batch) == 2

    # List by type
    all_msgs = await ChatMessage.list(session, type="message")
    assert len(all_msgs) >= 3

    # Last(n) — no offset param
    recent = await ChatMessage.last(session, n=2)
    assert len(recent) <= 2

    # Patch an event
    await msg.patch(content="Updated Hello!")
    assert msg.content == "Updated Hello!"

    # Stale patch — force an old token
    msg_stale_copy = await ChatMessage.list(session)
    # Corrupt the token to simulate a stale update
    msg_stale_copy[0]._created_at = "1970-01-01T00:00:00Z"
    with pytest.raises(MemoryVersionConflictError):
        await msg_stale_copy[0].patch(content="Stale edit")

    # Delete
    await msg.delete()
    remaining = await ChatMessage.list(session)
    assert not any(e.sequence_id == msg.sequence_id for e in remaining)

    await session.delete()


# ── Scenario 8: Emitter validation ───────────────────────────────────────────


@skip_unless_live
async def test_emitter_validation(space: DRMemorySpace) -> None:
    """An emitter not in participants returns a 400 bad-request error from the service."""
    from datarobot_genai.application_utils.memory import MemoryBadRequestError

    session = await ChatSession.post(
        space,
        tenant="acme",
        topic="validation",
        chat_id=_unique("val-"),
        title="Validation test",
        rev=1,
    )
    stranger_oid = "111111111111111111111111"  # valid format, not a participant
    with pytest.raises(MemoryBadRequestError):
        await ChatMessage.post(
            session,
            content="Unauthorised!",
            emitter_type="user",
            emitter_id=stranger_oid,
        )
    await session.delete()
