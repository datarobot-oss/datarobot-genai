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

"""OTel log tool: ``otel_logs_list`` (plan §2.4).

Near-1:1 wrapper around ``GET .../logs/`` — linear ~127 tokens/line, so a capped
default limit and a per-line character cap are the only truncation this module
needs; nothing here composes :mod:`datarobot_genai.drtools.otel.truncation`'s
budget/dedup machinery, which exists for the trace tools' much larger payloads.

Two traps this module exists to avoid:

* ``level`` is a *minimum*, not an exact match, and the full accepted set is
  ``debug|info|warn|warning|error|critical``. ``drmcputils.constants.LOG_LEVELS``
  is only ``("debug", "info", "warn", "error")`` — reusing it would silently
  reject ``warning`` and ``critical``, which this API accepts. Use
  :data:`datarobot_genai.drtools.otel.constants.OTEL_LOG_LEVELS` instead.
* ``max_line_chars`` truncates *characters*, not bytes — one measured attribute
  in the trace tools was multi-byte-heavy, and byte-slicing UTF-8 breaks
  multibyte sequences. ``message`` and ``stacktrace`` are capped independently,
  each with its own ``…[truncated, N more chars]`` marker, so a long stacktrace
  never eats into the message's own budget.
"""

from typing import Annotated
from typing import Any
from typing import Literal

from datarobot.errors import ClientError

from datarobot_genai.drmcputils.client_exceptions import raise_tool_error_for_client_error
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot_otel_query import OtelQueryApiClient
from datarobot_genai.drtools.core.utils import require_object_id
from datarobot_genai.drtools.otel.constants import OTEL_LOG_LEVELS
from datarobot_genai.drtools.otel.constants import EntityType
from datarobot_genai.drtools.otel.truncation import cap_value
from datarobot_genai.drtools.pagination import clamp_limit
from datarobot_genai.drtools.pagination import merge_pagination_metadata

# Cap on each of a log line's 'message'/'stacktrace' fields, independently, in
# characters. Confirmed by plan §9 step 9's run against real deployments, where it
# bit on 3/50 and 11/50 error lines (embedded tracebacks) and left the rest intact —
# i.e. it trims the outliers without flattening ordinary log output. Nothing else in
# this module repeats the literal.
DEFAULT_MAX_LINE_CHARS = 2_000

# Fields capped independently by max_line_chars, each with its own marker.
_CAPPED_LINE_FIELDS: tuple[str, ...] = ("message", "stacktrace")

# Plan §2's shared conventions: GENAI_EXPERIMENTATION is required by *every*
# /otel route, this one included — worth surfacing in the description so a 403
# reads as configuration, not "no access to this entity's logs".
_OTEL_403_HINT = (
    "A 403 here usually means configuration, not missing data: the "
    "GENAI_EXPERIMENTATION feature flag is required for every /otel route."
)


@tool_metadata(
    tags={"otel", "datarobot", "logs", "list", "observability", "debug"},
    description=(
        "[OTel—logs] List OTel log lines for an entity: linear ~127 tokens/line "
        "(measured 977 / 12,229 / 127,307 tokens at limit 10 / 100 / 1000).\n\n"
        "'level' is a MINIMUM level, not an exact match — the full accepted set "
        "is 'debug' | 'info' | 'warn' | 'warning' | 'error' | 'critical'.\n\n"
        "trace_id and span_id are the log<->trace correlation path — the single "
        "most useful filter here for debugging one failing request, and easy to "
        "miss since nothing else names it.\n\n"
        "Each line's 'message' and 'stacktrace' are capped at max_line_chars "
        "independently (0 disables), each with its own "
        "'…[truncated, N more chars]' marker when cut. This endpoint has no "
        "total_count — only 'next'/'previous' for pagination.\n\n"
        f"{_OTEL_403_HINT}\n\n"
        "Example: otel_logs_list(entity_type='deployment', entity_id='...', level='error')\n"
        "Example: otel_logs_list(entity_type='deployment', entity_id='...', "
        "trace_id='...', includes=['ERROR'])"
    ),
    display_name="OTel — List logs",
    description_ui=(
        "List OTel log lines for an entity, filtered by level, time window, body "
        "text, and span or trace ID."
    ),
)
async def otel_logs_list(
    *,
    entity_type: Annotated[EntityType, "Type of entity the OTel data belongs to."],
    entity_id: Annotated[str, "24-character hex ID of the entity."],
    limit: Annotated[int, "Max log lines to return (1-100, clamped). Default 50."] = 50,
    offset: Annotated[int, "Lines to skip for pagination. Default 0."] = 0,
    level: Annotated[
        Literal["debug", "info", "warn", "warning", "error", "critical"],
        "Minimum log level to return (not an exact match). Default 'debug'.",
    ] = "debug",
    start_time: Annotated[
        str | None, "RFC3339 start of the window (e.g. '2026-08-24T00:00:00Z')."
    ] = None,
    end_time: Annotated[str | None, "RFC3339 end of the window."] = None,
    includes: Annotated[
        list[str] | None,
        "Body text substrings that must all be present (AND logic, up to 10).",
    ] = None,
    excludes: Annotated[
        list[str] | None, "Body text substrings that must be absent (up to 10)."
    ] = None,
    span_id: Annotated[str | None, "Filter to a specific OTel span ID."] = None,
    trace_id: Annotated[str | None, "Filter to a specific OTel trace ID."] = None,
    max_line_chars: Annotated[
        int,
        "Cap each of 'message'/'stacktrace' independently, in characters. "
        f"Default {DEFAULT_MAX_LINE_CHARS}. 0 disables.",
    ] = DEFAULT_MAX_LINE_CHARS,
) -> dict[str, Any]:
    eid = require_object_id(entity_id, "entity_id")
    if offset < 0:
        raise ToolError(
            "Argument validation error: 'offset' must be >= 0.",
            kind=ToolErrorKind.VALIDATION,
        )
    if level not in OTEL_LOG_LEVELS:
        raise ToolError(
            f"Argument validation error: 'level' must be one of {OTEL_LOG_LEVELS}.",
            kind=ToolErrorKind.VALIDATION,
        )

    clamped_limit, note = clamp_limit(limit)
    try:
        result = OtelQueryApiClient().list_logs(
            entity_type,
            eid,
            limit=clamped_limit,
            offset=offset,
            level=level,
            start_time=start_time,
            end_time=end_time,
            includes=includes,
            excludes=excludes,
            span_id=span_id,
            trace_id=trace_id,
        )
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)

    data = result.get("data", []) or []
    logs = [_capped_log_line(entry, max_line_chars) for entry in data]
    response = merge_pagination_metadata(
        {"logs": logs, "count": len(logs)},
        result,
        note,
        offset=offset,
        limit=clamped_limit,
    )
    # merge_pagination_metadata generically copies total_count/totalCount/total
    # from the raw response when present. The logs/ endpoint is
    # PaginatedWithoutTotalResponseValidator and never sends one in practice,
    # but that's a property of today's upstream shape, not of this code — strip
    # it explicitly so §2.4's "no total_count" invariant holds even if the
    # upstream response ever grows one of those keys.
    response.pop("total_count", None)
    return response


def _capped_log_line(entry: dict[str, Any], max_line_chars: int) -> dict[str, Any]:
    """Cap 'message' and 'stacktrace' independently, each with its own marker.

    A copy of ``entry`` with every other field passed through unchanged.
    """
    capped = dict(entry)
    for field in _CAPPED_LINE_FIELDS:
        value = capped.get(field)
        if isinstance(value, str):
            capped[field] = _capped_field(value, max_line_chars)
    return capped


def _capped_field(value: str, max_chars: int) -> str:
    """Window one field and append a '…[truncated, N more chars]' marker when cut."""
    window, info = cap_value(value, max_chars)
    if info is None:
        return window
    return f"{window}…[truncated, {info['total_chars'] - info['returned_chars']} more chars]"
