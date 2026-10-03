# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OTel entity-stats tool: ``otel_entity_stats_get`` (plan §2.7).

A preflight/validation tool, not a discovery tool: ``GET /otel/stats/`` takes
only ``serviceName``, ``userId`` and a time window — there is no name search or
type filter, so it cannot answer "which entity is relevant here". Entity
resolution stays where it already works (``workload_list``,
``deployment_get_info``, the user's own config). What this *is* good at is
answering "does this entity have any OTel data, and can I read it?" before an
agent starts paging traces — hence ``entity_type``/``entity_id`` are required
and the tool is scoped to exactly one entity.

Three things the upstream controller makes possible, all load-bearing here:

1. Rows are per ``(user_id, service_name)``, not per entity — one entity
   returns one row *per user who produced telemetry on it*. The top-level
   counts are summed across rows; returning the first row alone would
   undercount. ``by_user`` keeps the per-user breakdown, capped at
   :data:`ENTITY_STATS_BY_USER_MAX` rows, while ``user_count`` stays exact
   across every row fetched (not just the ones shown).
2. Passing ``serviceName`` explicitly avoids the self-scoping trap: when it is
   omitted and the caller is not an org admin, the controller silently sets
   ``user_id = [caller]`` — "entities *you* generated telemetry for", a much
   narrower question than it looks. Always building and passing it (as
   ``f"{entity_type}-{entity_id}"``) is what makes "no data" mean *no data*.
3. No feature flag or seat license is required, unlike every trace/log/metrics
   route in this package — so this tool still answers on a cluster where those
   403. That is the one distinction no other tool here can make, which is why
   a 403 *from this endpoint* is reported as a real permissions failure rather
   than folded into "no data".
"""

from typing import Annotated
from typing import Any

from datarobot.errors import ClientError

from datarobot_genai.drmcputils.client_exceptions import raise_tool_error_for_client_error
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot_otel_query import OtelQueryApiClient
from datarobot_genai.drtools.core.utils import require_object_id
from datarobot_genai.drtools.otel.constants import EntityType

# OtelStatsController's own validator ceiling. Requested unconditionally so the
# sums below are computed from the fullest page the API allows, not a
# tool-chosen smaller page that would silently undercount.
ENTITY_STATS_LIMIT = 1_000

# Cap on the 'by_user' breakdown shown in the response. 'user_count' is NOT
# capped by this — it counts every distinct user_id across all rows fetched
# (up to ENTITY_STATS_LIMIT), so it stays exact even when more than this many
# users produced telemetry on the entity.
ENTITY_STATS_BY_USER_MAX = 20

_ROW_COUNT_FIELDS: tuple[str, ...] = ("span_count", "metric_count", "log_count")


@tool_metadata(
    tags={"otel", "datarobot", "stats", "entity", "preflight", "observability"},
    description=(
        "[OTel—entity stats] Preflight check: does this entity have any OTel "
        "data at all, and can the caller read it? Not a discovery tool — "
        "entity_type and entity_id are required, and there is no name search "
        "or type filter; resolve the entity first with workload_list, "
        "deployment_get_info, or similar.\n\n"
        "Builds service_name internally as f'{entity_type}-{entity_id}' and "
        "always passes it — omitting it would silently scope the query to "
        "only the calling user's own telemetry, which is why 'no data' here "
        "actually means no data. Rows are per (user_id, service_name): "
        "span_count/metric_count/log_count are summed across every matching "
        f"row, and by_user keeps the breakdown (capped at "
        f"{ENTITY_STATS_BY_USER_MAX} rows; user_count stays exact across every "
        f"row fetched, not just the rows shown). Requests limit="
        f"{ENTITY_STATS_LIMIT} (the API's ceiling) and adds a note if the true "
        "total exceeds it, so the sums are never silently partial.\n\n"
        "Unlike every trace/log/metrics tool in this set, this one needs no "
        "feature flag or seat license — it still answers when those tools "
        "would 403, so you can tell 'this entity has data but I can't read "
        "it' apart from 'there is nothing here'. A 403 from this tool means "
        "the caller cannot read this entity (a real permissions failure) and "
        "raises rather than returning an empty result. One inherent "
        "limitation: if OTel storage was never provisioned for this org at "
        "all, that also comes back as has_otel_data=false — indistinguishable "
        "from genuinely no data.\n\n"
        "Example: otel_entity_stats_get(entity_type='deployment', entity_id='...')"
    ),
    display_name="OTel — Get entity stats",
    description_ui=(
        "Check whether an entity has any OTel data and whether the caller can "
        "read it, before paging traces or logs."
    ),
)
async def otel_entity_stats_get(
    *,
    entity_type: Annotated[EntityType, "Type of entity the OTel data belongs to."],
    entity_id: Annotated[str, "24-character hex ID of the entity."],
    start_time: Annotated[
        str | None, "RFC3339 start of the window (e.g. '2026-08-24T00:00:00Z')."
    ] = None,
    end_time: Annotated[str | None, "RFC3339 end of the window."] = None,
) -> dict[str, Any]:
    eid = require_object_id(entity_id, "entity_id")
    service_name = f"{entity_type}-{eid}"

    try:
        result = OtelQueryApiClient().get_entity_stats(
            service_name, start=start_time, end=end_time, limit=ENTITY_STATS_LIMIT
        )
    except ClientError as exc:
        if getattr(exc, "status_code", None) == 403:
            raise ToolError(
                f"Permission denied reading OTel stats for {entity_type} "
                f"'{eid}' (service_name={service_name!r}). This is a "
                "permissions failure, not an empty result — the caller cannot "
                "read this entity.",
                kind=ToolErrorKind.AUTHENTICATION,
            ) from exc
        raise_tool_error_for_client_error(exc)

    rows = result.get("data", []) or []
    user_ids = {row.get("user_id") for row in rows if row.get("user_id") is not None}

    response: dict[str, Any] = {
        "entity_type": entity_type,
        "entity_id": eid,
        "service_name": service_name,
        "has_otel_data": bool(rows),
        **{field: _sum_field(rows, field) for field in _ROW_COUNT_FIELDS},
        "user_count": len(user_ids),
        "by_user": [_by_user_row(row) for row in rows[:ENTITY_STATS_BY_USER_MAX]],
    }

    total_count = _extract_total_count(result)
    if total_count is not None and total_count > ENTITY_STATS_LIMIT:
        response["note"] = (
            f"total_count ({total_count}) exceeds the request limit "
            f"({ENTITY_STATS_LIMIT}); the counts above may be partial."
        )

    return response


def _sum_field(rows: list[dict[str, Any]], field: str) -> int:
    """Sum one count field across every per-(user_id, service_name) row."""
    return sum(int(row.get(field) or 0) for row in rows)


def _by_user_row(row: dict[str, Any]) -> dict[str, Any]:
    """Project one raw stats row to the by_user breakdown shape.

    Normalizes a present-but-``None`` count the same way :func:`_sum_field`
    does (``int(value or 0)``), so a row's per-user breakdown never disagrees
    with the top-level sum it contributed to — e.g. a row with
    ``span_count: None`` must not show up as ``None`` here while counting as
    ``0`` in the summed total.
    """
    counts = {field: int(row.get(field) or 0) for field in _ROW_COUNT_FIELDS}
    return {"user_id": row.get("user_id"), **counts}


def _extract_total_count(result: dict[str, Any]) -> int | None:
    """Read the server's true row count under whichever key it used.

    Returns ``None`` (treated as "unknown, no note") rather than raising when
    the value present is non-numeric — a malformed upstream field should not
    crash an otherwise-successful call.
    """
    for key in ("total_count", "totalCount", "total"):
        value = result.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None
