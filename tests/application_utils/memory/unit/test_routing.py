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

"""Unit tests for the field routing table builder (_routing.py)."""

from __future__ import annotations

from typing import Annotated

import pytest

from datarobot_genai.application_utils.memory import DRConcurrencyField
from datarobot_genai.application_utils.memory import DRDeduplicationKey
from datarobot_genai.application_utils.memory import DREvent
from datarobot_genai.application_utils.memory import DRSession
from tests.application_utils.memory.unit.conftest import ChatMessage
from tests.application_utils.memory.unit.conftest import ChatSession
from tests.application_utils.memory.unit.conftest import DedupeOnlySession
from tests.application_utils.memory.unit.conftest import MinimalSession

# ── Session routing: field classification ────────────────────────────────────


def test_chat_session_dedup_field_is_chat_id() -> None:
    """GIVEN ChatSession WHEN routing is built THEN dedup_field == 'chat_id'."""
    routing = ChatSession._get_routing()
    assert routing.dedup_field == "chat_id"


def test_chat_session_range_fields_in_declaration_order() -> None:
    """GIVEN ChatSession WHEN routing is built THEN range_fields preserves order."""
    routing = ChatSession._get_routing()
    assert routing.range_fields == ["tenant", "topic"]


def test_chat_session_concurrency_field_is_rev() -> None:
    """GIVEN ChatSession WHEN routing is built THEN concurrency_field == 'rev'."""
    routing = ChatSession._get_routing()
    assert routing.concurrency_field == "rev"


def test_chat_session_metadata_fields_is_title() -> None:
    """GIVEN ChatSession WHEN routing is built THEN metadata_fields == ['title']."""
    routing = ChatSession._get_routing()
    assert routing.metadata_fields == ["title"]


def test_minimal_session_all_fields_go_to_metadata() -> None:
    """GIVEN MinimalSession (no markers) WHEN routing THEN all fields are metadata."""
    routing = MinimalSession._get_routing()
    assert routing.dedup_field is None
    assert routing.range_fields == []
    assert routing.concurrency_field is None
    assert "label" in routing.metadata_fields


def test_dedup_only_session_no_range_fields() -> None:
    """GIVEN DedupeOnlySession WHEN routing THEN no range fields, has dedup."""
    routing = DedupeOnlySession._get_routing()
    assert routing.dedup_field == "key"
    assert routing.range_fields == []
    assert "notes" in routing.metadata_fields


def test_description_prefix_defaults_to_class_name() -> None:
    """GIVEN MinimalSession with no __description_prefix__ WHEN _prefix() THEN class name."""
    assert MinimalSession._prefix() == "MinimalSession"


def test_description_prefix_uses_declared_value() -> None:
    """GIVEN ChatSession with __description_prefix__ = 'chat' WHEN _prefix() THEN 'chat'."""
    assert ChatSession._prefix() == "chat"


# ── Session routing: error conditions ────────────────────────────────────────


def test_two_dedup_fields_raises_type_error_at_first_use() -> None:
    """GIVEN a session class with two DRDeduplicationKey fields WHEN routing THEN TypeError."""
    with pytest.raises(TypeError, match="at most one field may carry DRDeduplicationKey"):

        class _BadSession(DRSession):
            k1: Annotated[str, DRDeduplicationKey]
            k2: Annotated[str, DRDeduplicationKey]

        # Force routing table construction
        _BadSession._get_routing()


def test_two_concurrency_fields_raises_type_error_at_first_use() -> None:
    """GIVEN two DRConcurrencyField fields WHEN routing THEN TypeError."""
    with pytest.raises(TypeError, match="at most one field may carry DRConcurrencyField"):

        class _BadSession(DRSession):
            v1: Annotated[int, DRConcurrencyField]
            v2: Annotated[int, DRConcurrencyField]

        _BadSession._get_routing()


def test_reserved_field_name_is_filtered_by_base_class() -> None:
    """GIVEN a subclass redeclaring 'participants' WHEN routing THEN it is treated as base field.

    The routing builder receives only OWN fields (filtered against base model_fields), so
    a redeclared 'participants' is silently excluded from routing — Pydantic handles it at
    the model level.  No routing ValueError is raised; the table simply has no extra fields.
    """

    class _RedeclaredParticipants(DRSession):
        participants: str  # type: ignore[assignment]

    # Routing succeeds (no ValueError); participants is excluded from own fields
    routing = _RedeclaredParticipants._get_routing()
    assert routing.dedup_field is None
    assert routing.range_fields == []
    assert routing.metadata_fields == []


# ── Event routing ─────────────────────────────────────────────────────────────


def test_chat_message_body_fields_is_score() -> None:
    """GIVEN ChatMessage WHEN event routing THEN body_fields contains 'score'."""
    routing = ChatMessage._get_routing()
    assert routing.body_fields == ["score"]


def test_event_reserved_field_is_filtered_by_base_class() -> None:
    """GIVEN a DREvent subclass redeclaring 'content' WHEN routing THEN treated as base field.

    The routing builder sees only own fields (filtered against DREvent.model_fields), so
    'content' — already a base field — is excluded.  No routing ValueError is raised.
    """

    class _RedeclaredContent(DREvent, session=ChatSession):
        content: str  # type: ignore[assignment]

    # Routing succeeds; 'content' is excluded from own fields
    routing = _RedeclaredContent._get_routing()
    assert routing.body_fields == []


def test_event_without_extra_fields_has_empty_body_fields() -> None:
    """GIVEN a DREvent subclass with no extra fields WHEN routing THEN body_fields empty."""

    class _MinEvent(DREvent, session=ChatSession):
        pass

    routing = _MinEvent._get_routing()
    assert routing.body_fields == []


def test_event_session_type_is_bound() -> None:
    """GIVEN class MyEvent(DREvent, session=ChatSession) WHEN inspected THEN session type set."""
    assert ChatMessage.__dr_session_type__ is ChatSession


# ── Routing table is class-specific (not shared across subclasses) ────────────


def test_routing_tables_are_independent() -> None:
    """GIVEN two different session subclasses WHEN routing THEN tables are independent."""
    r1 = ChatSession._get_routing()
    r2 = MinimalSession._get_routing()
    assert r1 is not r2
    assert r1.dedup_field != r2.dedup_field
