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

"""Marker-driven field routing tables for ``DRSession`` and ``DREvent`` subclasses.

Called lazily on the first ORM operation to classify each user-declared field
into its wire mapping:

Session fields
--------------
* ``DRDeduplicationKey`` → ``deduplicationKey``
* ``DRRangeKey`` (ordered) → ``description`` segments
* ``DRConcurrencyField`` → mirrors server ``version``
* *(unmarked)* → ``metadata``

Event fields
------------
* *(all unmarked)* → ``body`` additional properties
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NamedTuple

from datarobot_genai.application_utils.persistence.markers import DRConcurrencyField
from datarobot_genai.application_utils.persistence.markers import DRDeduplicationKey
from datarobot_genai.application_utils.persistence.markers import DRRangeKey

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

# ── Reserved field names ─────────────────────────────────────────────────────

# Fields managed by DRSession base — subclasses must not redeclare them.
_SESSION_RESERVED: frozenset[str] = frozenset(
    {
        "participants",
    }
)

# Fields managed by DREvent base — subclasses must not redeclare them.
_EVENT_RESERVED: frozenset[str] = frozenset(
    {
        "content",
        "emitter_type",
        "emitter_id",
    }
)

# ── Routing table types ───────────────────────────────────────────────────────


class SessionRoutingTable(NamedTuple):
    """Immutable routing table built once per ``DRSession`` subclass."""

    dedup_field: str | None  # field carrying DRDeduplicationKey (or None)
    range_fields: list[str]  # ordered DRRangeKey fields
    concurrency_field: str | None  # field carrying DRConcurrencyField (or None)
    metadata_fields: list[str]  # remaining unmarked non-reserved fields


class EventRoutingTable(NamedTuple):
    """Immutable routing table built once per ``DREvent`` subclass."""

    body_fields: list[str]  # all unmarked non-reserved fields → body


# ── Helpers ───────────────────────────────────────────────────────────────────


def _has_marker(field_info: FieldInfo, marker: type) -> bool:
    """Return ``True`` if any metadata element is the given marker class or an instance of it.

    Accepts both the bare marker class (``Annotated[str, DRRangeKey]`` — the documented form)
    and a marker instance (``Annotated[str, DRRangeKey()]``); the latter would otherwise be
    silently misrouted to ``metadata``.
    """
    return any(m is marker or isinstance(m, marker) for m in field_info.metadata)


# ── Builders ──────────────────────────────────────────────────────────────────


def build_session_routing(
    cls_name: str,
    own_fields: dict[str, FieldInfo],
) -> SessionRoutingTable:
    """Inspect subclass-declared fields and return a ``SessionRoutingTable``.

    Parameters
    ----------
    cls_name : str
        Class name, used in error messages.
    own_fields : dict[str, FieldInfo]
        Fields declared in the subclass only (parent fields already excluded).

    Returns
    -------
    SessionRoutingTable

    Raises
    ------
    ValueError
        If a reserved field name is used.
    TypeError
        If more than one field carries ``DRDeduplicationKey`` or
        ``DRConcurrencyField``.
    """
    dedup_fields: list[str] = []
    range_fields: list[str] = []
    concurrency_fields: list[str] = []
    metadata_fields: list[str] = []

    for name, field_info in own_fields.items():
        if name in _SESSION_RESERVED:
            raise ValueError(
                f"{cls_name}: field name {name!r} is reserved by the ORM base. "
                "Choose a different name."
            )
        markers = [
            marker
            for marker in (DRDeduplicationKey, DRRangeKey, DRConcurrencyField)
            if _has_marker(field_info, marker)
        ]
        if len(markers) > 1:
            raise TypeError(
                f"{cls_name}: field {name!r} carries multiple ORM markers "
                f"{[m.__name__ for m in markers]!r}; a field may carry at most one."
            )
        if DRDeduplicationKey in markers:
            dedup_fields.append(name)
        elif DRRangeKey in markers:
            range_fields.append(name)
        elif DRConcurrencyField in markers:
            concurrency_fields.append(name)
        else:
            metadata_fields.append(name)

    if len(dedup_fields) > 1:
        raise TypeError(
            f"{cls_name}: at most one field may carry DRDeduplicationKey; found: {dedup_fields!r}"
        )
    if len(concurrency_fields) > 1:
        raise TypeError(
            f"{cls_name}: at most one field may carry DRConcurrencyField; "
            f"found: {concurrency_fields!r}"
        )

    return SessionRoutingTable(
        dedup_field=dedup_fields[0] if dedup_fields else None,
        range_fields=range_fields,
        concurrency_field=concurrency_fields[0] if concurrency_fields else None,
        metadata_fields=metadata_fields,
    )


def build_event_routing(
    cls_name: str,
    own_fields: dict[str, FieldInfo],
) -> EventRoutingTable:
    """Inspect subclass-declared fields and return an ``EventRoutingTable``.

    Parameters
    ----------
    cls_name : str
        Class name, used in error messages.
    own_fields : dict[str, FieldInfo]
        Fields declared in the subclass only (parent fields already excluded).

    Returns
    -------
    EventRoutingTable

    Raises
    ------
    ValueError
        If a reserved field name is used.
    """
    body_fields: list[str] = []

    for name in own_fields:
        if name in _EVENT_RESERVED:
            raise ValueError(
                f"{cls_name}: field name {name!r} is reserved by the ORM base. "
                "Choose a different name."
            )
        body_fields.append(name)

    return EventRoutingTable(body_fields=body_fields)
