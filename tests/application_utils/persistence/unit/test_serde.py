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

"""Nested (de)serialisation tests for the persistence ORM.

Declared ``DREvent`` body fields and ``DRSession`` metadata fields must round-trip
through any Pydantic-expressible type — nested models, ``list[Model]``,
``Optional[Model]``, enums and dicts — while remaining byte-identical for scalars and
robust (raw fallback) against malformed wire payloads.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from enum import StrEnum
from typing import Any

import httpx
import pytest
import respx
from pydantic import BaseModel

from datarobot_genai.application_utils.persistence import DREvent
from datarobot_genai.application_utils.persistence import DRMemoryServiceClient
from datarobot_genai.application_utils.persistence import DRMemorySpace
from datarobot_genai.application_utils.persistence import DRSession
from tests.application_utils.persistence.unit.conftest import PARTICIPANT
from tests.application_utils.persistence.unit.conftest import SESSION_ID
from tests.application_utils.persistence.unit.conftest import SPACE_ID
from tests.application_utils.persistence.unit.conftest import ChatSession

BASE = "https://app.datarobot.com/api/v2"
MEMORY_BASE = f"{BASE}/memory"
SESSIONS_URL = f"{MEMORY_BASE}/{SPACE_ID}/sessions/"
SESSION_URL = f"{SESSIONS_URL}{SESSION_ID}/"
EVENTS_URL = f"{SESSION_URL}events/"
EVENT_URL = f"{EVENTS_URL}5/"


# ── Nested types under test ───────────────────────────────────────────────────


class Color(StrEnum):
    """Simple string enum for enum round-trip coverage."""

    red = "red"
    green = "green"


class Widget(BaseModel):
    """Nested Pydantic model persisted inside a body/metadata field."""

    name: str
    size: int
    color: Color = Color.red


class RichMessage(DREvent, session=ChatSession):
    """Event whose body carries nested models, lists, optionals, enums and dicts."""

    __event_type__ = "message"

    widget: Widget | None = None
    widgets: list[Widget] = []
    color: Color = Color.red
    attrs: dict[str, Any] = {}


class RichSession(DRSession):
    """Session whose metadata carries nested models, lists, optionals, enums and dicts."""

    __description_prefix__ = "rich"

    widget: Widget | None = None
    widgets: list[Widget] = []
    color: Color = Color.red
    attrs: dict[str, Any] = {}


class ScalarSession(DRSession):
    """Session with only scalar metadata fields (byte-identity check)."""

    __description_prefix__ = "scalar"

    s: str = ""
    n: int = 0
    f: float = 0.0
    b: bool = False


# ── Test infrastructure ────────────────────────────────────────────────────────


def _client() -> DRMemoryServiceClient:
    return DRMemoryServiceClient(endpoint=BASE, api_token="t", http_client=httpx.AsyncClient())


def _space() -> DRMemorySpace:
    return DRMemorySpace._from_wire(
        _client(),
        {"memorySpaceId": SPACE_ID, "userId": "u", "tenantId": "t", "createdAt": ""},
    )


def _session() -> ChatSession:
    wire = {
        "id": SESSION_ID,
        "participants": [PARTICIPANT],
        "description": "//chat/acme/billing/",
        "deduplicationKey": "chat-001",
        "metadata": {"title": "T"},
        "version": 1,
        "createdAt": "2026-06-30T00:00:00Z",
    }
    return ChatSession._from_wire(_space(), wire)  # type: ignore[return-value]


def _echo_event(captured: dict[str, Any]) -> Callable[[httpx.Request], httpx.Response]:
    """Return a respx side-effect that echoes the posted body back as an EventResponse."""

    def _handler(req: httpx.Request) -> httpx.Response:
        payload = json.loads(req.content)
        captured["body"] = payload
        return httpx.Response(
            201,
            json={
                "sequenceId": 5,
                "createdAt": "2026-06-30T00:00:01Z",
                "eventType": payload["type"],
                "emitterType": payload["emitter"]["type"],
                "emitterId": payload["emitter"].get("id"),
                "body": payload["body"],
            },
        )

    return _handler


def _echo_session(captured: dict[str, Any]) -> Callable[[httpx.Request], httpx.Response]:
    """Return a respx side-effect that echoes the posted metadata back as a SessionResponse."""

    def _handler(req: httpx.Request) -> httpx.Response:
        payload = json.loads(req.content)
        captured["body"] = payload
        return httpx.Response(
            201,
            json={
                "id": SESSION_ID,
                "participants": payload.get("participants", [PARTICIPANT]),
                "metadata": payload.get("metadata", {}),
                "version": 1,
                "createdAt": "2026-06-30T00:00:00Z",
            },
        )

    return _handler


# ── Body: nested single model ───────────────────────────────────────────────────


@respx.mock
async def test_body_nested_model_round_trips() -> None:
    """GIVEN a nested Widget body field WHEN post THEN wire is JSON and read is typed."""
    captured: dict[str, Any] = {}
    respx.post(EVENTS_URL).mock(side_effect=_echo_event(captured))

    widget = Widget(name="gizmo", size=3, color=Color.green)
    event = await RichMessage.post(_session(), content="hi", emitter_type="agent", widget=widget)

    assert captured["body"]["body"]["widget"] == {"name": "gizmo", "size": 3, "color": "green"}
    assert isinstance(event.widget, Widget)
    assert event.widget == widget


# ── Body: list[Model] ────────────────────────────────────────────────────────────


@respx.mock
async def test_body_list_of_models_round_trips() -> None:
    """GIVEN a list[Widget] body field WHEN post THEN wire is a list of dicts, read is typed."""
    captured: dict[str, Any] = {}
    respx.post(EVENTS_URL).mock(side_effect=_echo_event(captured))

    widgets = [Widget(name="a", size=1), Widget(name="b", size=2, color=Color.green)]
    event = await RichMessage.post(_session(), content="hi", emitter_type="agent", widgets=widgets)

    assert captured["body"]["body"]["widgets"] == [
        {"name": "a", "size": 1, "color": "red"},
        {"name": "b", "size": 2, "color": "green"},
    ]
    assert event.widgets == widgets
    assert all(isinstance(w, Widget) for w in event.widgets)


# ── Body: Optional[Model] ────────────────────────────────────────────────────────


@respx.mock
async def test_body_optional_model_round_trips_value_and_none() -> None:
    """GIVEN an Optional[Widget] WHEN post with a value and with None THEN both round-trip."""
    captured: dict[str, Any] = {}
    respx.post(EVENTS_URL).mock(side_effect=_echo_event(captured))

    with_value = await RichMessage.post(
        _session(),
        content="hi",
        emitter_type="agent",
        widget=Widget(name="w", size=9),
    )
    assert isinstance(with_value.widget, Widget)

    with_none = await RichMessage.post(_session(), content="hi", emitter_type="agent", widget=None)
    assert captured["body"]["body"]["widget"] is None
    assert with_none.widget is None


# ── Body: enum ───────────────────────────────────────────────────────────────────


@respx.mock
async def test_body_enum_round_trips() -> None:
    """GIVEN an enum body field WHEN post THEN wire stores the value and read is the enum."""
    captured: dict[str, Any] = {}
    respx.post(EVENTS_URL).mock(side_effect=_echo_event(captured))

    event = await RichMessage.post(
        _session(), content="hi", emitter_type="agent", color=Color.green
    )

    assert captured["body"]["body"]["color"] == "green"
    assert event.color is Color.green


# ── Body: dict ───────────────────────────────────────────────────────────────────


@respx.mock
async def test_body_dict_round_trips() -> None:
    """GIVEN a dict body field WHEN post THEN it round-trips verbatim."""
    captured: dict[str, Any] = {}
    respx.post(EVENTS_URL).mock(side_effect=_echo_event(captured))

    attrs = {"k": 1, "nested": {"x": [1, 2, 3]}}
    event = await RichMessage.post(_session(), content="hi", emitter_type="agent", attrs=attrs)

    assert captured["body"]["body"]["attrs"] == attrs
    assert event.attrs == attrs


# ── Metadata: nested model / list / optional / enum / dict ──────────────────────


@respx.mock
async def test_metadata_nested_model_round_trips() -> None:
    """GIVEN a nested Widget metadata field WHEN post THEN wire is JSON and read is typed."""
    captured: dict[str, Any] = {}
    respx.post(SESSIONS_URL).mock(side_effect=_echo_session(captured))

    widget = Widget(name="cfg", size=7, color=Color.green)
    session = await RichSession.post(_space(), widget=widget)

    assert captured["body"]["metadata"]["widget"] == {"name": "cfg", "size": 7, "color": "green"}
    assert isinstance(session.widget, Widget)
    assert session.widget == widget


@respx.mock
async def test_metadata_list_enum_and_dict_round_trip() -> None:
    """GIVEN list/enum/dict metadata fields WHEN post THEN each round-trips through metadata."""
    captured: dict[str, Any] = {}
    respx.post(SESSIONS_URL).mock(side_effect=_echo_session(captured))

    widgets = [Widget(name="a", size=1)]
    attrs = {"flag": True, "count": 2}
    session = await RichSession.post(_space(), widgets=widgets, color=Color.green, attrs=attrs)

    assert captured["body"]["metadata"]["widgets"] == [{"name": "a", "size": 1, "color": "red"}]
    assert captured["body"]["metadata"]["color"] == "green"
    assert captured["body"]["metadata"]["attrs"] == attrs
    assert session.widgets == widgets
    assert session.color is Color.green
    assert session.attrs == attrs


@respx.mock
async def test_metadata_optional_model_none_round_trips() -> None:
    """GIVEN an Optional[Widget] metadata field WHEN post with None THEN it round-trips as None."""
    captured: dict[str, Any] = {}
    respx.post(SESSIONS_URL).mock(side_effect=_echo_session(captured))

    session = await RichSession.post(_space(), widget=None)

    assert captured["body"]["metadata"]["widget"] is None
    assert session.widget is None


# ── Graceful raw fallback on malformed wire ─────────────────────────────────────


def test_body_malformed_wire_falls_back_to_raw() -> None:
    """GIVEN a body value that cannot validate WHEN _from_wire THEN the raw value is kept."""
    wire = {
        "sequenceId": 5,
        "createdAt": "2026-06-30T00:00:01Z",
        "emitterType": "agent",
        "emitterId": None,
        "body": {"content": "hi", "widgets": "not-a-list"},
    }
    event = RichMessage._from_wire(_session(), wire)  # type: ignore[assignment]
    assert event.widgets == "not-a-list"  # raw preserved, no ValidationError raised


def test_metadata_malformed_wire_falls_back_to_raw() -> None:
    """GIVEN a metadata value that cannot validate WHEN _from_wire THEN the raw value is kept."""
    wire = {
        "id": SESSION_ID,
        "participants": [PARTICIPANT],
        "metadata": {"widget": "not-a-widget"},
        "version": 1,
        "createdAt": "",
    }
    session = RichSession._from_wire(_space(), wire)  # type: ignore[assignment]
    assert session.widget == "not-a-widget"  # raw preserved, no ValidationError raised


def test_malformed_wire_fallback_logs_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    """GIVEN an uncoercible field WHEN _from_wire THEN the raw fallback is logged at WARNING.

    The raw value crashes downstream consumers that expect the declared type, so
    the fallback must be observable rather than silent.
    """
    wire = {
        "sequenceId": 5,
        "createdAt": "2026-06-30T00:00:01Z",
        "emitterType": "agent",
        "emitterId": None,
        "body": {"content": "hi", "widgets": "not-a-list"},
    }
    serde_logger = "datarobot_genai.application_utils.persistence._serde"
    with caplog.at_level(logging.WARNING, logger=serde_logger):
        event = RichMessage._from_wire(_session(), wire)  # type: ignore[assignment]

    assert event.widgets == "not-a-list"
    assert any(
        r.name == serde_logger and "widgets" in r.getMessage() and "raw value" in r.getMessage()
        for r in caplog.records
    )


# ── patch preserves an unpatched nested field ───────────────────────────────────


@respx.mock
async def test_patch_preserves_unpatched_nested_field() -> None:
    """GIVEN an event with a nested widget WHEN patching only the enum THEN widget is resent."""
    captured: dict[str, Any] = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "sequenceId": 5,
                "createdAt": "2026-06-30T00:00:02Z",
                "emitterType": "agent",
                "emitterId": None,
                "body": {"content": "hi", "widget": {"name": "w", "size": 1, "color": "red"}},
            },
        )

    respx.patch(EVENT_URL).mock(side_effect=_capture)

    wire = {
        "sequenceId": 5,
        "createdAt": "2026-06-30T00:00:01Z",
        "emitterType": "agent",
        "emitterId": None,
        "body": {"content": "hi", "widget": {"name": "w", "size": 1, "color": "green"}},
    }
    event = RichMessage._from_wire(_session(), wire)  # type: ignore[assignment]

    await event.patch(color=Color.red)

    # The unpatched nested widget is re-serialised and resent (JSON dict, not a model).
    assert captured["body"]["body"]["widget"] == {"name": "w", "size": 1, "color": "green"}
    assert captured["body"]["body"]["color"] == "red"


# ── Scalars are byte-identical to a raw passthrough ─────────────────────────────


def test_scalar_metadata_serialisation_is_identical_to_raw() -> None:
    """GIVEN scalar metadata fields WHEN building the wire THEN values and types are unchanged."""
    payload = ScalarSession._to_wire_create({"s": "x", "n": 3, "f": 0.5, "b": True})

    metadata = payload["metadata"]
    assert metadata == {"s": "x", "n": 3, "f": 0.5, "b": True}
    assert type(metadata["s"]) is str
    assert type(metadata["n"]) is int
    assert type(metadata["f"]) is float
    assert type(metadata["b"]) is bool


def test_scalar_body_serialisation_is_identical_to_raw() -> None:
    """GIVEN a scalar body field WHEN building the wire THEN the value and type are unchanged."""
    from tests.application_utils.persistence.unit.conftest import ChatMessage

    body = ChatMessage._to_wire_body({"content": "hi", "score": 0.9})
    assert body["score"] == 0.9
    assert type(body["score"]) is float


# ── Full serialise → deserialise cycle equality ─────────────────────────────────


def test_full_cycle_equality_for_all_field_kinds() -> None:
    """GIVEN mixed field kinds WHEN serialised to the wire and read back THEN they compare equal."""
    original = RichMessage.model_construct(
        content="hi",
        emitter_type="agent",
        emitter_id=None,
        widget=Widget(name="w", size=1, color=Color.green),
        widgets=[Widget(name="a", size=2)],
        color=Color.green,
        attrs={"k": "v"},
    )

    wire_body = RichMessage._to_wire_body(
        {
            "content": original.content,
            "widget": original.widget,
            "widgets": original.widgets,
            "color": original.color,
            "attrs": original.attrs,
        }
    )
    event = RichMessage._from_wire(  # type: ignore[assignment]
        _session(),
        {
            "sequenceId": 5,
            "createdAt": "2026-06-30T00:00:01Z",
            "emitterType": "agent",
            "emitterId": None,
            "body": wire_body,
        },
    )

    assert event.widget == original.widget
    assert event.widgets == original.widgets
    assert event.color is Color.green
    assert event.attrs == original.attrs


# ── Adapter caching ──────────────────────────────────────────────────────────────


def test_field_type_adapter_is_cached_per_field() -> None:
    """GIVEN repeated adapter requests WHEN fetched twice THEN the same instance is returned."""
    from datarobot_genai.application_utils.persistence._serde import field_type_adapter

    first = field_type_adapter(RichMessage, "widget")
    second = field_type_adapter(RichMessage, "widget")
    assert first is second


def test_deserialize_field_returns_raw_on_validation_error() -> None:
    """GIVEN a value that cannot validate WHEN deserialize_field THEN the raw value is returned."""
    from datarobot_genai.application_utils.persistence._serde import deserialize_field

    assert deserialize_field(RichMessage, "widgets", 123) == 123


@pytest.mark.parametrize("value", ["plain", 7, 1.5, True, None])
def test_deserialize_field_scalar_identity(value: object) -> None:
    """GIVEN a matching-shaped scalar WHEN round-tripped THEN it survives serialise+deserialise."""
    from datarobot_genai.application_utils.persistence._serde import deserialize_field
    from datarobot_genai.application_utils.persistence._serde import serialize_field

    wire = serialize_field(RichSession, "attrs", {"v": value})
    assert deserialize_field(RichSession, "attrs", wire) == {"v": value}
