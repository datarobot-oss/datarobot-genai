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

"""``DRSession`` — abstract Pydantic v2 base for Memory Service session models.

Subclass ``DRSession`` to define a typed document stored as a Memory Service
session.  Field markers control how each declared field maps to the wire format:

* ``Annotated[T, DRDeduplicationKey]`` → ``deduplicationKey`` (point-lookup key)
* ``Annotated[T, DRRangeKey]`` → ``description`` segment (ordered; prefix queries)
* ``Annotated[T, DRConcurrencyField]`` → mirrors server ``version`` integer
* *(plain field)* → ``metadata`` (payload; not queryable)

Examples
--------
.. code-block:: python

    from typing import Annotated
    from datarobot_genai.application_utils.persistence import (
        DRSession,
        DRDeduplicationKey,
        DRRangeKey,
        DRConcurrencyField,
        SYSTEM_PARTICIPANT,
    )

    class ChatSession(DRSession):
        __description_prefix__ = "chat"          # anchors all description queries

        tenant: Annotated[str, DRRangeKey]        # description segment 1
        topic:  Annotated[str, DRRangeKey]        # description segment 2
        chat_id: Annotated[str, DRDeduplicationKey]  # point-lookup key
        rev:     Annotated[int, DRConcurrencyField]  # user-visible version
        title:   str                              # metadata.title (payload only)

    async def demo(space: DRMemorySpace) -> None:
        # Create (idempotent via dedup key)
        session = await ChatSession.post(
            space,
            tenant="acme",
            topic="billing",
            chat_id="billing-chat-001",
            title="Billing enquiry",
        )

        # Point lookup by dedup key
        same = await ChatSession.get(space, chat_id="billing-chat-001")

        # Prefix query — all topics under "acme"
        all_acme = await ChatSession.list(space, tenant="acme")

        # Update metadata + range key in one call
        await session.patch(title="Updated title", topic="support")
"""

from __future__ import annotations

import builtins
import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PrivateAttr

from datarobot_genai.application_utils.persistence._encoding import build_description
from datarobot_genai.application_utils.persistence._encoding import build_query_description
from datarobot_genai.application_utils.persistence._encoding import parse_description
from datarobot_genai.application_utils.persistence._encoding import validate_range_key
from datarobot_genai.application_utils.persistence._routing import SessionRoutingTable
from datarobot_genai.application_utils.persistence._routing import build_session_routing
from datarobot_genai.application_utils.persistence.exceptions import DRMemoryConflictError
from datarobot_genai.application_utils.persistence.exceptions import DRMemoryNotFoundError
from datarobot_genai.application_utils.persistence.markers import DEFAULT_SESSION_TTL_SECONDS
from datarobot_genai.application_utils.persistence.markers import SYSTEM_PARTICIPANT

if TYPE_CHECKING:
    from datarobot_genai.application_utils.persistence.space import DRMemorySpace

logger = logging.getLogger(__name__)


class DRSession(BaseModel):
    """Abstract base class for Memory Service ORM session models.

    Do not instantiate directly; subclass and declare fields with ORM markers.

    Class variables
    ---------------
    __description_prefix__ : str
        Prefix injected at the start of the encoded ``description``.  Defaults
        to the subclass name.  Keep it short and stable; it is part of every
        stored description and every list-query filter.
    __lifecycle_strategies__ : list[dict[str, Any]]
        Lifecycle strategy objects sent on session creation.  Defaults to a single
        ``soft_delete`` strategy with a ``DEFAULT_SESSION_TTL_SECONDS`` (2 year) TTL
        trigger, so sessions auto-clean unless a subclass overrides this.  Override
        with a different strategy list to change the TTL/strategy, or set to ``[]``
        to send no lifecycle strategies at all.

    Read-only properties
    --------------------
    id : str
        Server-assigned session UUID.
    created_at : str
        ISO-8601 creation timestamp.
    version : int
        Server-assigned version integer (optimistic-concurrency token).
    """

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    participants: builtins.list[str] = Field(
        default_factory=lambda: [SYSTEM_PARTICIPANT],
        description=(
            "Participant ObjectIds (exactly 1).  Defaults to the system sentinel "
            f"({SYSTEM_PARTICIPANT!r}) for sessions not owned by a specific user."
        ),
    )

    # ── Class-level configuration ─────────────────────────────────────────
    __description_prefix__: ClassVar[str] = ""
    __lifecycle_strategies__: ClassVar[builtins.list[dict[str, Any]]] = [
        {"type": "soft_delete", "trigger": {"ttl": DEFAULT_SESSION_TTL_SECONDS}}
    ]

    # Routing table — built lazily on first ORM call; shared across instances.
    _dr_routing: ClassVar[SessionRoutingTable | None] = None

    # ── Private attributes (server-assigned; not Pydantic fields) ─────────
    _id: str = PrivateAttr(default="")
    _created_at: str = PrivateAttr(default="")
    _version: int = PrivateAttr(default=0)
    _space: Any = PrivateAttr(default=None)  # DRMemorySpace — avoids circular import

    # ── __init_subclass__ ─────────────────────────────────────────────────

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Reset the routing cache so each subclass builds (and caches) its own table.

        Without this, a subclass of an already-used concrete ``DRSession`` subclass would
        inherit the parent's cached ``_dr_routing`` via the MRO and silently drop its own
        declared fields from wire serialisation.
        """
        super().__init_subclass__(**kwargs)
        cls._dr_routing = None

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def id(self) -> str:
        """Server-assigned session UUID (read-only)."""
        return self._id

    @property
    def created_at(self) -> str:
        """ISO-8601 creation timestamp (read-only)."""
        return self._created_at

    @property
    def version(self) -> int:
        """Current server version integer (read-only; updated on every patch)."""
        return self._version

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return (
            f"{type(self).__name__}(id={self._id!r}, version={self._version}, "
            f"participants={self.participants!r})"
        )

    # ── Routing table ─────────────────────────────────────────────────────

    @classmethod
    def _get_routing(cls) -> SessionRoutingTable:
        """Return the routing table for this subclass (built once, lazily)."""
        if cls._dr_routing is None:
            base_fields = set(DRSession.model_fields)
            own_fields = {k: v for k, v in cls.model_fields.items() if k not in base_fields}
            cls._dr_routing = build_session_routing(cls.__name__, own_fields)
        return cls._dr_routing

    # ── Wire serialisation helpers ────────────────────────────────────────

    @classmethod
    def _prefix(cls) -> str:
        """Return the description prefix, defaulting to the class name."""
        return cls.__description_prefix__ or cls.__name__

    @classmethod
    def _to_wire_create(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Build the ``CreateSessionRequest`` payload from constructor kwargs."""
        routing = cls._get_routing()

        participants: list[str] = kwargs.get("participants", [SYSTEM_PARTICIPANT])

        # Dedup key
        dedup_key: str | None = None
        if routing.dedup_field and routing.dedup_field in kwargs:
            dedup_value = kwargs[routing.dedup_field]
            if dedup_value is None:
                raise ValueError(f"Deduplication key {routing.dedup_field!r} must not be None.")
            dedup_key = str(dedup_value)

        # Description from range keys
        description: str | None = None
        if routing.range_fields:
            range_values: list[str] = []
            for fname in routing.range_fields:
                val = kwargs.get(fname, "")
                validate_range_key(fname, val)
                range_values.append(str(val))
            description = build_description(cls._prefix(), range_values)

        # Metadata from plain fields
        metadata: dict[str, Any] = {}
        for fname in routing.metadata_fields:
            if fname in kwargs:
                metadata[fname] = kwargs[fname]

        payload: dict[str, Any] = {"participants": participants}
        if dedup_key is not None:
            payload["deduplicationKey"] = dedup_key
        if description is not None:
            payload["description"] = description
        if metadata:
            payload["metadata"] = metadata
        payload["lifecycleStrategies"] = cls.__lifecycle_strategies__

        return payload

    @classmethod
    def _decode_range_fields(
        cls, routing: SessionRoutingTable, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Decode ``DRRangeKey`` field values from a wire ``description`` string.

        Returns a mapping of range-field name → value, or an empty mapping when there are
        no range fields, no description, or the description cannot be parsed for this prefix.
        """
        description = data.get("description") or ""
        if not (routing.range_fields and description):
            return {}
        try:
            values = parse_description(cls._prefix(), description, len(routing.range_fields))
        except ValueError:
            logger.debug(
                "Session %s description %r could not be parsed for prefix %r; "
                "range fields will be absent.",
                data.get("id"),
                description,
                cls._prefix(),
            )
            return {}
        return dict(zip(routing.range_fields, values))

    @classmethod
    def _from_wire(cls, space: DRMemorySpace, data: dict[str, Any]) -> DRSession:
        """Construct an instance from a ``SessionResponse`` wire dict."""
        routing = cls._get_routing()
        init_kwargs: dict[str, Any] = {
            "participants": data.get("participants", [SYSTEM_PARTICIPANT]),
        }

        # Dedup field ← deduplicationKey
        if routing.dedup_field is not None:
            raw_dedup = data.get("deduplicationKey")
            if raw_dedup is not None:
                init_kwargs[routing.dedup_field] = raw_dedup

        # Range fields ← description
        init_kwargs.update(cls._decode_range_fields(routing, data))

        # Metadata fields ← metadata.  A required field absent from the wire is
        # set to None so attribute access does not raise AttributeError after
        # model_construct; fields with a default are left for model_construct to fill.
        metadata = data.get("metadata") or {}
        for fname in routing.metadata_fields:
            if fname in metadata:
                init_kwargs[fname] = metadata[fname]
            elif cls.model_fields[fname].is_required():
                init_kwargs[fname] = None

        # Server version (tolerate a missing or explicitly-null "version")
        raw_version = data.get("version")
        version = int(raw_version) if raw_version is not None else 1

        # Concurrency field ← version
        if routing.concurrency_field is not None:
            init_kwargs[routing.concurrency_field] = version

        # Bypass Pydantic validation (server is the source of truth)
        obj: DRSession = cls.model_construct(**init_kwargs)

        # Set server-assigned private attrs
        obj._id = str(data.get("id", ""))
        obj._created_at = str(data.get("createdAt", ""))
        obj._version = version
        obj._space = space

        return obj

    def _update_from_wire(self, data: dict[str, Any]) -> None:
        """Update in-place from a ``SessionResponse`` wire dict after a patch."""
        routing = type(self)._get_routing()

        raw_version = data.get("version")
        self._version = int(raw_version) if raw_version is not None else self._version

        # Sync concurrency field if declared
        if routing.concurrency_field is not None:
            setattr(self, routing.concurrency_field, self._version)

        # Sync metadata fields
        metadata = data.get("metadata") or {}
        for fname in routing.metadata_fields:
            if fname in metadata:
                setattr(self, fname, metadata[fname])

        # Sync range fields from description
        for fname, val in type(self)._decode_range_fields(routing, data).items():
            setattr(self, fname, val)

        # Sync dedup field
        if routing.dedup_field is not None:
            raw_dedup = data.get("deduplicationKey")
            if raw_dedup is not None:
                setattr(self, routing.dedup_field, raw_dedup)

    # ── Class-method operations ───────────────────────────────────────────

    @classmethod
    async def post(
        cls,
        space: DRMemorySpace,
        **kwargs: Any,
    ) -> DRSession:
        """Create a session, or adopt the existing one on a deduplication conflict.

        Parameters
        ----------
        space : DRMemorySpace
            Memory space to create the session in.
        **kwargs
            Session field values.  Pass ``participants=["<objectid>"]`` to
            scope to a user; omit to use the system sentinel.

        Returns
        -------
        DRSession
            The newly created or adopted session.

        Examples
        --------
        .. code-block:: python

            session = await ChatSession.post(
                space,
                tenant="acme",
                topic="billing",
                chat_id="chat-001",
                title="Billing enquiry",
            )
        """
        routing = cls._get_routing()
        cls._validate_post_kwargs(routing, kwargs)
        payload = cls._to_wire_create(kwargs)
        try:
            resp = await space._client.request("POST", f"{space.id}/sessions/", json=payload)
            return cls._from_wire(space, resp.json())
        except DRMemoryConflictError as exc:
            # Adopt the existing session
            if exc.existing_id:
                resp = await space._client.request("GET", f"{space.id}/sessions/{exc.existing_id}/")
                return cls._from_wire(space, resp.json())
            raise

    @classmethod
    async def get(
        cls,
        space: DRMemorySpace,
        id: str | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> DRSession:
        """Fetch a session by its server-assigned ``id`` or by ``deduplication_key``.

        Exactly one of ``id=`` or the subclass's ``DRDeduplicationKey`` field
        name must be supplied.

        Parameters
        ----------
        space : DRMemorySpace
            Memory space containing the session.
        id : str | None
            Server-assigned session UUID.
        **kwargs
            Pass the ``DRDeduplicationKey`` field name as a keyword argument
            for an exact-match point lookup (e.g. ``chat_id="billing-chat-001"``).

        Returns
        -------
        DRSession

        Raises
        ------
        DRMemoryNotFoundError
            If no matching session is found.
        ValueError
            If neither ``id`` nor a dedup key is supplied, or the subclass has
            no ``DRDeduplicationKey`` field and a keyword arg is provided.

        Examples
        --------
        .. code-block:: python

            # By server id
            session = await ChatSession.get(space, id="uuid-string")

            # By dedup key
            session = await ChatSession.get(space, chat_id="billing-chat-001")
        """
        routing = cls._get_routing()
        cls._validate_get_kwargs(routing, kwargs)

        if id is not None:
            resp = await space._client.request("GET", f"{space.id}/sessions/{id}/")
            return cls._from_wire(space, resp.json())

        # Dedup-key point lookup via the list endpoint
        if kwargs and routing.dedup_field and routing.dedup_field in kwargs:
            dedup_value = str(kwargs[routing.dedup_field])
            resp = await space._client.request(
                "GET",
                f"{space.id}/sessions/",
                params={"deduplicationKey": dedup_value},
            )
            items = resp.json().get("items", [])
            # The service deduplicationKey filter is not guaranteed to be an exact
            # match, so confirm the key client-side instead of trusting items[0].
            for item in items:
                if item.get("deduplicationKey") == dedup_value:
                    return cls._from_wire(space, item)
            raise DRMemoryNotFoundError(
                f"No session with {routing.dedup_field}={dedup_value!r} in space {space.id}",
                status_code=404,
            )

        raise ValueError(
            "get() requires either id=<uuid> or the subclass DRDeduplicationKey "
            "field as a keyword argument."
        )

    @classmethod
    async def list(
        cls,
        space: DRMemorySpace,
        *,
        participant: str | None = None,
        **kwargs: Any,
    ) -> builtins.list[DRSession]:
        """List sessions matching a range-key prefix and/or participant filter.

        Range-key kwargs must form a **contiguous leading prefix** of the
        declared ``DRRangeKey`` fields (e.g. for fields ``[tenant, topic]`` you
        may filter on ``tenant=`` alone or on ``tenant=`` + ``topic=``, but not
        on ``topic=`` alone).

        Parameters
        ----------
        space : DRMemorySpace
            Memory space to query.
        participant : str | None
            Filter to sessions that include this ObjectId in ``participants``.
        **kwargs
            Leading range-key field values for a prefix query.

        Returns
        -------
        list[DRSession]
            All matching sessions (auto-paginated).

        Raises
        ------
        ValueError
            If range-key kwargs are not a contiguous leading prefix.

        Examples
        --------
        .. code-block:: python

            # All sessions for tenant "acme"
            sessions = await ChatSession.list(space, tenant="acme")

            # Scoped to a user
            sessions = await ChatSession.list(space, participant=user_oid)

            # Combined: user + range prefix
            sessions = await ChatSession.list(
                space, participant=user_oid, tenant="acme", topic="billing"
            )
        """
        routing = cls._get_routing()
        cls._validate_list_kwargs(routing, kwargs)

        # Build query params
        query_params: dict[str, Any] = {}

        # Build description filter from range key kwargs
        if kwargs and routing.range_fields:
            range_values: list[str] = []
            for fname in routing.range_fields:
                if fname in kwargs:
                    value = kwargs[fname]
                    validate_range_key(fname, value)
                    range_values.append(str(value))
            if range_values:
                query_params["description"] = build_query_description(cls._prefix(), range_values)

        if participant is not None:
            query_params["participants"] = participant

        return await cls._list_all(space, query_params)

    @classmethod
    def _validate_list_kwargs(
        cls,
        routing: SessionRoutingTable,
        kwargs: dict[str, Any],
    ) -> None:
        """Raise ``ValueError`` if kwargs are not a contiguous leading prefix."""
        if not kwargs:
            return
        range_fields = routing.range_fields
        # Find which range fields are in kwargs
        provided = [fname for fname in range_fields if fname in kwargs]
        unexpected = [k for k in kwargs if k not in range_fields]
        if unexpected:
            raise ValueError(
                f"list() received unexpected keyword arguments: {unexpected!r}. "
                "Only DRRangeKey field names are accepted as range filters."
            )
        if not provided:
            return
        # They must be a contiguous leading prefix
        expected_prefix = range_fields[: len(provided)]
        if provided != expected_prefix:
            raise ValueError(
                f"list() range-key arguments must form a contiguous leading prefix of "
                f"{range_fields!r}; got {provided!r}."
            )

    @classmethod
    async def _list_all(
        cls,
        space: DRMemorySpace,
        query_params: dict[str, Any],
    ) -> builtins.list[DRSession]:
        """Auto-paginate through all matching sessions."""
        results: builtins.list[DRSession] = []
        offset = 0
        limit = 100
        while True:
            params = {**query_params, "offset": offset, "limit": limit}
            resp = await space._client.request("GET", f"{space.id}/sessions/", params=params)
            data = resp.json()
            items = data.get("items", [])
            for item in items:
                results.append(cls._from_wire(space, item))
            offset += len(items)
            total = data.get("total")
            # Stop on an empty or short page (the last page), or once a known total is
            # reached. Relying on a short page keeps pagination correct even when the
            # server omits "total" (which previously truncated results after page one).
            if not items or len(items) < limit:
                break
            if total is not None and offset >= total:
                break
        return results

    # ── Instance operations ───────────────────────────────────────────────

    async def patch(self, **kwargs: Any) -> None:
        """Update this session in place.

        Pass any combination of metadata fields and/or ``DRRangeKey`` fields.
        ``participants`` and ``DRDeduplicationKey`` fields cannot be changed.

        Parameters
        ----------
        **kwargs
            Fields to update.

        Raises
        ------
        DRMemoryVersionConflictError
            If the session was updated concurrently (stale ``If-Match``).

        Examples
        --------
        .. code-block:: python

            await session.patch(title="New title")
            await session.patch(topic="support", title="Re: billing")
        """
        routing = type(self)._get_routing()
        self._validate_patch_kwargs(routing, kwargs)

        payload: dict[str, Any] = {}

        # Metadata is stored as a single wire object, so when a metadata field is
        # being changed we resend the full declared set (an explicit None clears a
        # field).  A patch that touches no metadata field leaves metadata untouched
        # rather than clobbering it with a rebuilt-from-current copy.
        if any(fname in kwargs for fname in routing.metadata_fields):
            new_metadata: dict[str, Any] = {}
            for fname in routing.metadata_fields:
                if fname in kwargs:
                    new_metadata[fname] = kwargs[fname]
                else:
                    current_val = getattr(self, fname, None)
                    if current_val is not None:
                        new_metadata[fname] = current_val
            payload["metadata"] = new_metadata

        # The description encodes the range keys; only rebuild and send it when a
        # range key is actually changing.  Rebuilding it on a metadata-only patch
        # would raise when a range field is unset (e.g. the stored description did
        # not decode) or silently re-key the session to a field default.
        if routing.range_fields and any(fname in kwargs for fname in routing.range_fields):
            range_values: list[str] = []
            for fname in routing.range_fields:
                val = kwargs.get(fname, getattr(self, fname, ""))
                validate_range_key(fname, val)
                range_values.append(str(val))
            payload["description"] = build_description(type(self)._prefix(), range_values)

        resp = await self._space._client.request(
            "PATCH",
            f"{self._space.id}/sessions/{self._id}/",
            json=payload,
            extra_headers={"If-Match": str(self._version)},
        )
        self._update_from_wire(resp.json())

    @staticmethod
    def _validate_post_kwargs(
        routing: SessionRoutingTable,
        kwargs: dict[str, Any],
    ) -> None:
        """Raise ``ValueError`` on undeclared or unknown session-create kwargs."""
        valid: set[str] = (
            set(routing.metadata_fields) | set(routing.range_fields) | {"participants"}
        )
        if routing.dedup_field:
            valid.add(routing.dedup_field)
        if routing.concurrency_field:
            valid.add(routing.concurrency_field)
        bad = sorted(set(kwargs) - valid)
        if bad:
            raise ValueError(
                f"post() received unexpected keyword arguments: {bad!r}. "
                "Only declared session fields and 'participants' are accepted."
            )

    @staticmethod
    def _validate_get_kwargs(
        routing: SessionRoutingTable,
        kwargs: dict[str, Any],
    ) -> None:
        """Raise ``ValueError`` on lookup kwargs other than the DRDeduplicationKey field."""
        allowed: set[str] = {routing.dedup_field} if routing.dedup_field else set()
        bad = sorted(set(kwargs) - allowed)
        if bad:
            raise ValueError(
                f"get() received unexpected keyword arguments: {bad!r}. "
                "Look up by id=<uuid> or the DRDeduplicationKey field name."
            )

    @staticmethod
    def _validate_patch_kwargs(
        routing: SessionRoutingTable,
        kwargs: dict[str, Any],
    ) -> None:
        """Raise ``ValueError`` on unsupported or reserved patch kwargs."""
        valid = set(routing.metadata_fields) | set(routing.range_fields)
        bad = set(kwargs) - valid
        if bad:
            raise ValueError(
                f"patch() does not accept: {sorted(bad)!r}. "
                "Only metadata fields and DRRangeKey fields may be updated."
            )

    async def delete(self) -> None:
        """Soft-delete this session.

        After deletion the session is no longer returned by ``get`` or ``list``.
        """
        await self._space._client.request("DELETE", f"{self._space.id}/sessions/{self._id}/")
