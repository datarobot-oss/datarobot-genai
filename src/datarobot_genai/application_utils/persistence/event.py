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

"""``DREvent`` — abstract Pydantic v2 base for Memory Service event models.

Events are an append-only log under a session.  Bind a ``DREvent`` subclass to
a session type with the ``session=`` class argument:

.. code-block:: python

    class ChatMessage(DREvent, session=ChatSession):
        __event_type__ = "message"  # "message" | "tool_output" | "status"

        score: float   # -> body.score  (round-trips; not queryable)

All plain declared fields (those not in the base ``content``, ``emitter_type``,
``emitter_id``) map to ``body`` additional properties.

Examples
--------
.. code-block:: python

    # Create a single event
    msg = await ChatMessage.post(
        session=my_session,
        content="Hello!",
        emitter_type="user",
        emitter_id=participant_oid,
        score=0.95,
    )
    print(msg.sequence_id)    # server-assigned integer address
    print(msg.created_at)     # ISO-8601; also the concurrency token for patch

    # Atomic batch create (up to 200 events)
    msgs = await ChatMessage.post_batch(
        session=my_session,
        events=[
            {"content": "Hi", "emitter_type": "user", "emitter_id": oid, "score": 0.9},
            {"content": "Hello", "emitter_type": "agent", "score": 0.8},
        ],
    )

    # List by type
    all_msgs = await ChatMessage.list(session=my_session, type="message")

    # Tail window (last N events)
    recent = await ChatMessage.last(session=my_session, n=20)

    # Update a single event (guarded by createdAt token)
    await msg.patch(score=0.5)

    # Atomic batch patch
    await ChatMessage.patch_batch(
        session=my_session,
        updates=[(msg, {"score": 0.5}), (msgs[0], {"score": 0.7})],
    )

    # Delete
    await msg.delete()
"""

from __future__ import annotations

import builtins
import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import PrivateAttr

from datarobot_genai.application_utils.persistence._routing import _EVENT_RESERVED
from datarobot_genai.application_utils.persistence._routing import EventRoutingTable
from datarobot_genai.application_utils.persistence._routing import build_event_routing
from datarobot_genai.application_utils.persistence.exceptions import DRMemoryBadRequestError

if TYPE_CHECKING:
    from datarobot_genai.application_utils.persistence.session import DRSession

logger = logging.getLogger(__name__)

_MAX_BATCH_SIZE = 200


class DREvent(BaseModel):
    """Abstract base class for Memory Service ORM event models.

    Do not instantiate directly; subclass with ``session=<SessionClass>``.

    Parameters
    ----------
    content : str
        Event text (1–100 000 characters).
    emitter_type : Literal["user", "agent"]
        Who produced the event.
    emitter_id : str | None
        ObjectId of the emitting user.  Required when ``emitter_type="user"``
        and must be a member of the session's ``participants``.

    Read-only properties
    --------------------
    sequence_id : int
        Server-assigned monotonic integer address (−1 before the event is posted).
    created_at : str
        ISO-8601 timestamp; also serves as the concurrency token for ``patch()``.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    content: str
    emitter_type: Literal["user", "agent"]
    emitter_id: str | None = None

    # ── Class-level configuration ─────────────────────────────────────────
    __event_type__: ClassVar[str] = "message"
    __dr_session_type__: ClassVar[type]  # set by class arg; unset on base

    # Routing table — built lazily on first ORM call
    _dr_routing: ClassVar[EventRoutingTable | None] = None

    # ── Private attributes (server-assigned) ──────────────────────────────
    _sequence_id: int = PrivateAttr(default=-1)
    _created_at: str = PrivateAttr(default="")
    _session: Any = PrivateAttr(default=None)  # DRSession — avoids circular import

    # ── __init_subclass__ ─────────────────────────────────────────────────

    def __init_subclass__(cls, session: type | None = None, **kwargs: Any) -> None:
        """Bind the event class to a session type and reset the routing cache."""
        super().__init_subclass__(**kwargs)
        if session is not None:
            cls.__dr_session_type__ = session
        cls._dr_routing = None  # force rebuild on next _get_routing() call

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def sequence_id(self) -> int:
        """Server-assigned event address (read-only; −1 before posting)."""
        return self._sequence_id

    @property
    def created_at(self) -> str:
        """ISO-8601 creation timestamp (read-only; also the concurrency token)."""
        return self._created_at

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return (
            f"{type(self).__name__}(sequence_id={self._sequence_id}, "
            f"emitter_type={self.emitter_type!r})"
        )

    # ── Routing table ─────────────────────────────────────────────────────

    @classmethod
    def _get_routing(cls) -> EventRoutingTable:
        """Return the routing table for this subclass (built once, lazily)."""
        if cls._dr_routing is None:
            base_fields = set(DREvent.model_fields)
            own_fields = {k: v for k, v in cls.model_fields.items() if k not in base_fields}
            cls._dr_routing = build_event_routing(cls.__name__, own_fields)
        return cls._dr_routing

    # ── Wire serialisation helpers ────────────────────────────────────────

    @classmethod
    def _to_wire_body(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Build the ``body`` object from constructor kwargs."""
        routing = cls._get_routing()
        body: dict[str, Any] = {"content": kwargs["content"]}
        for fname in routing.body_fields:
            if fname in kwargs:
                body[fname] = kwargs[fname]
        return body

    @staticmethod
    def _build_emitter(emitter_type: str, emitter_id: str | None) -> dict[str, Any]:
        """Build the wire ``emitter`` object, omitting ``id`` when it is ``None``."""
        emitter: dict[str, Any] = {"type": emitter_type}
        if emitter_id is not None:
            emitter["id"] = emitter_id
        return emitter

    @classmethod
    def _to_wire_create(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Build the ``CreateEventRequest`` payload."""
        return {
            "type": cls.__event_type__,
            "body": cls._to_wire_body(kwargs),
            "emitter": cls._build_emitter(kwargs["emitter_type"], kwargs.get("emitter_id")),
        }

    @classmethod
    def _from_wire(cls, session: DRSession, data: dict[str, Any]) -> DREvent:
        """Construct an instance from an ``EventResponse`` wire dict."""
        routing = cls._get_routing()
        body = data.get("body") or {}
        init_kwargs: dict[str, Any] = {
            "content": body.get("content", ""),
            "emitter_type": data.get("emitterType", "agent"),
            "emitter_id": data.get("emitterId"),
        }
        # A required body field absent from the wire is set to None so attribute
        # access does not raise AttributeError after model_construct; fields with a
        # default are left for model_construct to fill.
        for fname in routing.body_fields:
            if fname in body:
                init_kwargs[fname] = body[fname]
            elif cls.model_fields[fname].is_required():
                init_kwargs[fname] = None

        raw_seq = data.get("sequenceId")
        obj: DREvent = cls.model_construct(**init_kwargs)
        obj._sequence_id = int(raw_seq) if raw_seq is not None else -1
        obj._created_at = str(data.get("createdAt", ""))
        obj._session = session
        return obj

    def _update_from_wire(self, data: dict[str, Any]) -> None:
        """Update in-place from a patch response."""
        routing = type(self)._get_routing()
        self._created_at = str(data.get("createdAt", self._created_at))
        body = data.get("body") or {}
        if "content" in body:
            self.content = body["content"]
        for fname in routing.body_fields:
            if fname in body:
                setattr(self, fname, body[fname])
        emitter_type = data.get("emitterType")
        if emitter_type is not None:
            self.emitter_type = emitter_type
        self.emitter_id = data.get("emitterId", self.emitter_id)

    # ── Client-side kwargs validators ─────────────────────────────────────

    @classmethod
    def _validate_body_kwargs(cls, kwargs: dict[str, Any], *, allow_reserved: bool) -> None:
        """Raise ``ValueError`` if any kwarg is not a declared body field.

        Parameters
        ----------
        kwargs : dict[str, Any]
            Caller-supplied keyword arguments to validate.
        allow_reserved : bool
            When ``True``, the base-managed keys ``content``, ``emitter_type``,
            and ``emitter_id`` are also accepted (used for batch-style dicts
            that carry all fields inline).  When ``False``, only subclass-declared
            body field names are valid (used where the base fields are already
            consumed as explicit keyword parameters).

        Raises
        ------
        ValueError
            If any key in *kwargs* is not in the allowed set.
        """
        routing = cls._get_routing()
        valid: set[str] = set(routing.body_fields)
        if allow_reserved:
            valid |= _EVENT_RESERVED
        unexpected = sorted(set(kwargs) - valid)
        if unexpected:
            raise ValueError(
                f"{cls.__name__} received unexpected field(s): {unexpected!r}. "
                f"Declared body fields: {sorted(routing.body_fields)!r}."
            )

    # ── Participant pre-check ─────────────────────────────────────────────

    @classmethod
    def _check_emitter(cls, session: DRSession, kwargs: dict[str, Any]) -> None:
        """Raise ``DRMemoryBadRequestError`` early if the emitter is not a participant."""
        if kwargs.get("emitter_type") == "user":
            eid = kwargs.get("emitter_id")
            if eid is not None and eid not in session.participants:
                raise DRMemoryBadRequestError(
                    f"Event emitter {eid!r} is not in session participants "
                    f"{session.participants!r}."
                )

    # ── Class-method operations ───────────────────────────────────────────

    @classmethod
    async def post(
        cls,
        session: DRSession,
        *,
        content: str,
        emitter_type: Literal["user", "agent"],
        emitter_id: str | None = None,
        **kwargs: Any,
    ) -> DREvent:
        """Append a single event to the session.

        Parameters
        ----------
        session : DRSession
            Session to append the event to.
        content : str
            Event text (1–100 000 characters).
        emitter_type : Literal["user", "agent"]
            Who produced the event.
        emitter_id : str | None
            ObjectId of the emitter.  Required when ``emitter_type="user"``
            and must be in ``session.participants``.
        **kwargs
            Any declared body fields (e.g. ``score=0.9``).

        Returns
        -------
        DREvent
        """
        cls._validate_body_kwargs(kwargs, allow_reserved=False)
        all_kwargs = {
            "content": content,
            "emitter_type": emitter_type,
            "emitter_id": emitter_id,
            **kwargs,
        }
        cls._check_emitter(session, all_kwargs)
        payload = cls._to_wire_create(all_kwargs)
        space = session._space
        resp = await space._client.request(
            "POST",
            f"{space.id}/sessions/{session.id}/events/",
            json=payload,
        )
        return cls._from_wire(session, resp.json())

    @classmethod
    async def post_batch(
        cls,
        session: DRSession,
        events: builtins.list[dict[str, Any]],
    ) -> builtins.list[DREvent]:
        """Atomically append up to 200 events to the session.

        All events are appended in list order.  If any event fails validation
        the entire batch is rolled back.

        Parameters
        ----------
        session : DRSession
            Session to append the events to.
        events : list[dict[str, Any]]
            Each dict contains the same keyword arguments as :meth:`post`
            (``content``, ``emitter_type``, optionally ``emitter_id`` and body
            fields).

        Returns
        -------
        list[DREvent]

        Raises
        ------
        ValueError
            If the batch exceeds 200 events.

        Examples
        --------
        .. code-block:: python

            msgs = await ChatMessage.post_batch(
                session=my_session,
                events=[
                    {"content": "Hi", "emitter_type": "user", "emitter_id": oid},
                    {"content": "Hello", "emitter_type": "agent"},
                ],
            )
        """
        if len(events) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"post_batch() accepts at most {_MAX_BATCH_SIZE} events; received {len(events)}."
            )
        for item in events:
            cls._validate_body_kwargs(item, allow_reserved=True)
            missing = [key for key in ("content", "emitter_type") if key not in item]
            if missing:
                raise ValueError(
                    f"post_batch() event is missing required field(s) {missing!r}: {item!r}"
                )
            cls._check_emitter(session, item)
        wire_events = [cls._to_wire_create(e) for e in events]
        space = session._space
        resp = await space._client.request(
            "POST",
            f"{space.id}/sessions/{session.id}/events/batch/",
            json={"events": wire_events},
        )
        return [cls._from_wire(session, item) for item in resp.json().get("items", [])]

    @classmethod
    async def list(
        cls,
        session: DRSession,
        *,
        type: str | None = None,  # noqa: A002
        offset: int = 0,
        limit: int = 100,
    ) -> builtins.list[DREvent]:
        """List events under a session, optionally filtered by type.

        Parameters
        ----------
        session : DRSession
            Session to query.
        type : str | None
            Event type filter: ``"message"``, ``"tool_output"``, or
            ``"status"``.  ``None`` returns all types.
        offset : int
            Number of events to skip.
        limit : int
            Maximum number of events to return (1–100).

        Returns
        -------
        list[DREvent]
        """
        params: dict[str, Any] = {"offset": offset, "limit": limit}
        if type is not None:
            params["eventType"] = type
        space = session._space
        resp = await space._client.request(
            "GET",
            f"{space.id}/sessions/{session.id}/events/",
            params=params,
        )
        return [cls._from_wire(session, item) for item in resp.json().get("items", [])]

    @classmethod
    async def last(
        cls,
        session: DRSession,
        *,
        n: int,
        type: str | None = None,  # noqa: A002
    ) -> builtins.list[DREvent]:
        """Return the last ``n`` events in chronological order.

        ``lastN`` and ``offset`` are mutually exclusive on the service API; this
        method never sends ``offset``.

        Parameters
        ----------
        session : DRSession
            Session to query.
        n : int
            Number of tail events to return (1–100).
        type : str | None
            Optional event-type filter.

        Returns
        -------
        list[DREvent]
        """
        params: dict[str, Any] = {"lastN": n}
        if type is not None:
            params["eventType"] = type
        space = session._space
        resp = await space._client.request(
            "GET",
            f"{space.id}/sessions/{session.id}/events/",
            params=params,
        )
        return [cls._from_wire(session, item) for item in resp.json().get("items", [])]

    # ── Instance operations ───────────────────────────────────────────────

    async def patch(
        self,
        *,
        content: str | None = None,
        emitter_type: Literal["user", "agent"] | None = None,
        emitter_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Update this event in place, guarded by its ``created_at`` token.

        At least one field must be supplied.

        Parameters
        ----------
        content : str | None
            New event text.
        emitter_type : Literal["user", "agent"] | None
            New emitter type.
        emitter_id : str | None
            New emitter ObjectId.
        **kwargs
            Any declared body fields to update.

        Raises
        ------
        DRMemoryVersionConflictError
            If the event was updated concurrently (stale ``createdAt`` token).
        """
        routing = type(self)._get_routing()
        type(self)._validate_body_kwargs(kwargs, allow_reserved=False)
        body: dict[str, Any] | None = None
        if content is not None or kwargs:
            # The service replaces the whole body object, so resend every declared
            # body field (with kwargs applied on top) — otherwise a partial patch
            # silently drops the body fields the caller did not mention.
            body = {"content": content if content is not None else self.content}
            for fname in routing.body_fields:
                new_val = kwargs.get(fname, getattr(self, fname, None))
                if new_val is not None:
                    body[fname] = new_val

        payload: dict[str, Any] = {}
        if body is not None:
            payload["body"] = body
        if emitter_type is not None or emitter_id is not None:
            payload["emitter"] = self._build_emitter(
                emitter_type if emitter_type is not None else self.emitter_type,
                emitter_id if emitter_id is not None else self.emitter_id,
            )

        if not payload:
            raise ValueError("patch() requires at least one field to update.")

        session: DRSession = self._session
        space = session._space
        params: dict[str, Any] = {}
        if self._created_at:
            params["createdAt"] = self._created_at

        resp = await space._client.request(
            "PATCH",
            f"{space.id}/sessions/{session.id}/events/{self._sequence_id}/",
            json=payload,
            params=params if params else None,
        )
        self._update_from_wire(resp.json())

    async def delete(self) -> None:
        """Soft-delete this event.

        After deletion the event is no longer returned by ``list`` or ``last``.
        """
        session: DRSession = self._session
        space = session._space
        await space._client.request(
            "DELETE",
            f"{space.id}/sessions/{session.id}/events/{self._sequence_id}/",
        )

    # ── Batch patch (extras) ──────────────────────────────────────────────

    @classmethod
    async def patch_batch(
        cls,
        session: DRSession,
        updates: builtins.list[tuple[DREvent, dict[str, Any]]],
    ) -> builtins.list[DREvent]:
        """Atomically update up to 200 events, each guarded by its ``created_at`` token.

        Parameters
        ----------
        session : DRSession
            Session containing the events.  All events must belong to this session.
        updates : list[tuple[DREvent, dict[str, Any]]]
            Each tuple is ``(event_instance, kwargs)``.  The kwargs accept the
            same arguments as :meth:`patch`.

        Returns
        -------
        list[DREvent]
            Updated event instances (in the order of the input list).

        Raises
        ------
        ValueError
            If the batch exceeds 200 updates or no fields are provided for an item.
        DRMemoryVersionConflictError
            If any event was updated concurrently.

        Examples
        --------
        .. code-block:: python

            await ChatMessage.patch_batch(
                session=my_session,
                updates=[
                    (event_a, {"score": 0.9}),
                    (event_b, {"content": "Updated text"}),
                ],
            )
        """
        if len(updates) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"patch_batch() accepts at most {_MAX_BATCH_SIZE} updates; received {len(updates)}."
            )

        routing = cls._get_routing()
        wire_items: list[dict[str, Any]] = []

        for event, kwargs in updates:
            cls._validate_body_kwargs(kwargs, allow_reserved=True)
            content = kwargs.get("content")
            emitter_type = kwargs.get("emitter_type")
            emitter_id = kwargs.get("emitter_id")
            body_kwargs = {k: v for k, v in kwargs.items() if k in routing.body_fields}

            item: dict[str, Any] = {"sequenceId": event.sequence_id}
            if event.created_at:
                item["createdAt"] = event.created_at

            if content is not None or body_kwargs:
                body: dict[str, Any] = {
                    "content": content if content is not None else event.content
                }
                body.update(body_kwargs)
                item["body"] = body

            if emitter_type is not None or emitter_id is not None:
                item["emitter"] = cls._build_emitter(
                    emitter_type if emitter_type is not None else event.emitter_type,
                    emitter_id if emitter_id is not None else event.emitter_id,
                )

            if not any(k in item for k in ("body", "emitter")):
                raise ValueError(
                    f"patch_batch() item for sequence_id={event.sequence_id} has no "
                    "fields to update."
                )

            wire_items.append(item)

        space = session._space
        resp = await space._client.request(
            "PATCH",
            f"{space.id}/sessions/{session.id}/events/batch/",
            json={"events": wire_items},
        )
        # Index the server results by sequenceId so each original event is updated in place
        # and the returned list follows the caller's input order, regardless of the order
        # (or count) in which the service responds.
        result_items = resp.json().get("items", [])
        items_by_seq: dict[int, dict[str, Any]] = {
            item["sequenceId"]: item for item in result_items if "sequenceId" in item
        }

        updated_events: builtins.list[DREvent] = []
        for event, _ in updates:
            result_item = items_by_seq.get(event.sequence_id)
            if result_item is not None:
                event._update_from_wire(result_item)
            else:
                logger.warning(
                    "patch_batch: no server result for event sequence_id=%s; returning "
                    "the stale local copy (its createdAt concurrency token may be invalid).",
                    event.sequence_id,
                )
            updated_events.append(event)

        return updated_events
