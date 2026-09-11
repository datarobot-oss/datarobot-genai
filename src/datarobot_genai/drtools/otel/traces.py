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

"""OTel trace tools: list traces, retrieve one trace, and drill into a span's payload.

Three tools, in the order an agent is meant to reach for them (plan §2.1-§2.3):

* ``otel_traces_list`` — 1:1 wrapper around ``GET .../traces/``. No truncation:
  the endpoint hard-truncates ``completion`` server-side and carries no span
  attributes, so there is nothing here to project.
* ``otel_trace_get`` — the tool this ticket exists for. Raw passthrough of one
  trace can be 1,560,241 tokens (measured max); ``view="summary"`` (the default)
  reduces that to a flat ~133 tokens/span span-tree projection with per-span
  payload accounting (``payload_chars``/``payload_fields``) so an agent can choose
  where to drill in deliberately. ``view="payloads"`` instead applies canonical
  semconv selection and a hard character budget across the whole span page.
* ``otel_span_payload_get`` — the deliberate escape hatch: one span's payload,
  windowed by ``field_offset`` so even a 740,000-char field is reachable, with
  every dropped duplicate/semconv field named rather than silently discarded.

All truncation logic lives in :mod:`datarobot_genai.drtools.otel.truncation`; this
module composes those helpers with pagination, id validation and error mapping —
it does not reimplement any projection logic of its own.
"""

import json
import logging
from typing import Annotated
from typing import Any
from typing import Literal

from datarobot.errors import ClientError

from datarobot_genai.drmcputils.client_exceptions import raise_tool_error_for_client_error
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot_otel_query import OtelQueryApiClient
from datarobot_genai.drtools.core.utils import require_id
from datarobot_genai.drtools.core.utils import require_object_id
from datarobot_genai.drtools.core.utils import require_trace_id
from datarobot_genai.drtools.otel.constants import RESPONSE_PAYLOAD_FIELDS
from datarobot_genai.drtools.otel.constants import EntityType
from datarobot_genai.drtools.otel.truncation import apply_char_budget
from datarobot_genai.drtools.otel.truncation import canonical_attributes
from datarobot_genai.drtools.otel.truncation import cap_payload_value
from datarobot_genai.drtools.otel.truncation import cap_value
from datarobot_genai.drtools.otel.truncation import summarize_spans
from datarobot_genai.drtools.pagination import clamp_limit
from datarobot_genai.drtools.pagination import merge_pagination_metadata

logger = logging.getLogger(__name__)

# otel_trace_get's defaults, in one place. Plan §9 step 9 — a manual run against a
# real agent trace — is deliberately NOT part of this round, so these three
# numbers are provisional: measured on a proxy (§1's 3.1 chars/token), not on
# Claude's own tokenizer, and not yet exercised against a real trace. Nothing else
# in this package repeats these literals, so correcting them later (per step 9) is
# a one-line change here rather than a hunt through the module.
DEFAULT_TRACE_SPAN_LIMIT = 100
DEFAULT_TRACE_MAX_FIELD_CHARS = 2_000
DEFAULT_TRACE_MAX_TOTAL_CHARS = 60_000

# TracingListQueryValidator.post_validate rejects offset + limit > 10000 server
# side ("adjust endTime instead of using large offsets"). Pre-empted here so the
# failure is a readable VALIDATION error instead of a raw 400.
_TRACES_LIST_MAX_OFFSET_PLUS_LIMIT = 10_000

_TRACE_403_HINT = (
    "A 403 here usually means configuration, not missing data: the "
    "GENAI_EXPERIMENTATION feature flag and the AGENTIC_PREDICTIVE_GOVERNANCE_BUILDER "
    "seat license are required for every /otel trace route, and for 'deployment' or "
    "'custom_application' entities the "
    "MMM_DISABLE_OTEL_TRACING_VIEWING_FOR_DEPLOYMENT_AND_CUSTOM_APPS eng config can "
    "disable trace viewing entirely."
)


# ------------------------------------------------------------------ #
# otel_traces_list                                                     #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"otel", "datarobot", "traces", "list", "observability"},
    description=(
        "[OTel—traces] List OpenTelemetry traces for an entity: flat ~185 "
        "tokens/trace regardless of underlying trace size (the endpoint hard-"
        "truncates completion to 512 chars and carries no span attributes). Use "
        "this as the default entry point before otel_trace_get. Supports "
        "filtering by status, root span name, tool name, duration/cost bounds, "
        "and sorting.\n\n"
        f"{_TRACE_403_HINT}\n\n"
        "Example: otel_traces_list(entity_type='deployment', entity_id='...')\n"
        "Example: otel_traces_list(entity_type='deployment', entity_id='...', "
        "status='error', sort_by='duration', sort_direction='desc')"
    ),
    display_name="OTel — List traces",
    description_ui=(
        "List OpenTelemetry traces for an entity, with filtering by status, root "
        "span name, tool name, duration, and cost."
    ),
)
async def otel_traces_list(
    *,
    entity_type: Annotated[EntityType, "Type of entity the OTel data belongs to."],
    entity_id: Annotated[str, "24-character hex ID of the entity."],
    limit: Annotated[int, "Max traces to return (1-100, clamped). Default 20."] = 20,
    offset: Annotated[int, "Traces to skip for pagination. Default 0."] = 0,
    start_time: Annotated[
        str | None, "RFC3339 start of the window (e.g. '2026-08-24T00:00:00Z')."
    ] = None,
    end_time: Annotated[str | None, "RFC3339 end of the window."] = None,
    status: Annotated[
        Literal["error", "ok"] | None,
        "'error' returns only traces with at least one error span; 'ok' the opposite.",
    ] = None,
    root_span_name: Annotated[
        list[str] | None, "Filter by exact root span name (up to 50 names)."
    ] = None,
    tools: Annotated[
        list[str] | None,
        "Filter by gen_ai.tool.name (up to 50). Pass a list, not a comma-joined string.",
    ] = None,
    trace_type: Annotated[
        Literal["gen_ai"] | None, "Filter by trace type. Currently only 'gen_ai'."
    ] = None,
    min_trace_duration_ns: Annotated[int | None, "Minimum trace duration, in nanoseconds."] = None,
    min_span_duration_ns: Annotated[int | None, "Minimum span duration, in nanoseconds."] = None,
    max_span_duration_ns: Annotated[int | None, "Maximum span duration, in nanoseconds."] = None,
    min_trace_cost: Annotated[
        float | None, "Minimum trace cost. Fractional amounts are valid (e.g. 0.01)."
    ] = None,
    max_trace_cost: Annotated[
        float | None, "Maximum trace cost. Fractional amounts are valid (e.g. 0.01)."
    ] = None,
    sort_by: Annotated[
        Literal["timestamp", "duration", "cost"] | None,
        "Field to sort by. Defaults to 'timestamp' when sort_direction is set.",
    ] = None,
    sort_direction: Annotated[Literal["asc", "desc"] | None, "Sort direction."] = None,
) -> dict[str, Any]:
    eid = require_object_id(entity_id, "entity_id")
    if offset < 0:
        raise ToolError(
            "Argument validation error: 'offset' must be >= 0.",
            kind=ToolErrorKind.VALIDATION,
        )
    if tools and any(isinstance(name, str) and "," in name for name in tools):
        raise ToolError(
            "Argument validation error: 'tools' must be a list of tool names, not a "
            "comma-joined string. Pass a list, not a comma-joined string.",
            kind=ToolErrorKind.VALIDATION,
        )
    if sort_direction is not None and sort_by is None:
        sort_by = "timestamp"

    clamped_limit, note = clamp_limit(limit)
    if offset + clamped_limit > _TRACES_LIST_MAX_OFFSET_PLUS_LIMIT:
        raise ToolError(
            "Argument validation error: 'offset' + 'limit' must be <= "
            f"{_TRACES_LIST_MAX_OFFSET_PLUS_LIMIT}. Adjust start_time/end_time instead "
            "of paging with a large offset.",
            kind=ToolErrorKind.VALIDATION,
        )

    try:
        result = OtelQueryApiClient().list_traces(
            entity_type,
            eid,
            limit=clamped_limit,
            offset=offset,
            start_time=start_time,
            end_time=end_time,
            status=status,
            root_span_name=root_span_name,
            tools=tools,
            trace_type=trace_type,
            min_trace_duration_ns=min_trace_duration_ns,
            min_span_duration_ns=min_span_duration_ns,
            max_span_duration_ns=max_span_duration_ns,
            min_trace_cost=min_trace_cost,
            max_trace_cost=max_trace_cost,
            sort_by=sort_by,
            sort_direction=sort_direction,
        )
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)

    data = result.get("data", []) or []
    return merge_pagination_metadata(
        {"traces": data, "count": len(data)},
        result,
        note,
        offset=offset,
        limit=clamped_limit,
    )


# ------------------------------------------------------------------ #
# otel_trace_get                                                       #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"otel", "datarobot", "trace", "get", "observability", "debug"},
    description=(
        "[OTel—trace] Retrieve one trace's span tree. Raw passthrough is unusable "
        "on large traces (measured p95 = 809,639 tokens, max = 1,560,241) — span "
        "count does not predict size, so this tool always projects.\n\n"
        "view='summary' (default): the span tree with zero payloads — "
        "~133 tokens/span, flat regardless of payload size. Each span carries "
        "'payload_chars' and 'payload_fields' so you can see *where the mass is* "
        "before drilling in with otel_span_payload_get(trace_id=..., span_id=...).\n\n"
        "view='payloads': canonical semconv attributes only (duplicates and "
        "derived fields dropped and named), each field capped at max_field_chars, "
        "under a hard max_total_chars budget across the whole span page. Spans "
        "are emitted in order until the budget is spent, then dropped — "
        "'truncation' reports spans_returned/spans_dropped. Even this view can "
        "still overshoot a 20k-token target on the worst traces, so prefer "
        "summary first and drill into specific spans instead.\n\n"
        "span_limit/span_offset page the underlying spans server-side (default "
        f"page size {DEFAULT_TRACE_SPAN_LIMIT}, no server-side upper bound); span "
        "ordering under pagination is not guaranteed.\n\n"
        f"{_TRACE_403_HINT}\n\n"
        "Example: otel_trace_get(entity_type='deployment', entity_id='...', "
        "trace_id='...')\n"
        "Example: otel_trace_get(entity_type='deployment', entity_id='...', "
        "trace_id='...', view='payloads', max_total_chars=20000)"
    ),
    display_name="OTel — Get trace",
    description_ui=(
        "Retrieve one trace's span tree as a bounded summary, or its canonical "
        "payload attributes under a hard character budget."
    ),
)
async def otel_trace_get(
    *,
    entity_type: Annotated[EntityType, "Type of entity the OTel data belongs to."],
    entity_id: Annotated[str, "24-character hex ID of the entity."],
    trace_id: Annotated[str, "32-character hex OTel trace ID."],
    view: Annotated[
        Literal["summary", "payloads"],
        "'summary' (default): span tree, no payloads. 'payloads': canonical "
        "attributes, capped and budgeted.",
    ] = "summary",
    span_limit: Annotated[
        int, f"Server-side span page size. Default {DEFAULT_TRACE_SPAN_LIMIT}, no upper bound."
    ] = DEFAULT_TRACE_SPAN_LIMIT,
    span_offset: Annotated[int, "Spans to skip for server-side pagination."] = 0,
    max_field_chars: Annotated[
        int, "view='payloads' only: cap per attribute value, in characters."
    ] = DEFAULT_TRACE_MAX_FIELD_CHARS,
    max_total_chars: Annotated[
        int, "view='payloads' only: hard budget across the whole span page, in characters."
    ] = DEFAULT_TRACE_MAX_TOTAL_CHARS,
) -> dict[str, Any]:
    eid = require_object_id(entity_id, "entity_id")
    tid = require_trace_id(trace_id)
    if span_offset < 0:
        raise ToolError(
            "Argument validation error: 'span_offset' must be >= 0.",
            kind=ToolErrorKind.VALIDATION,
        )
    if span_limit < 1:
        raise ToolError(
            "Argument validation error: 'span_limit' must be >= 1.",
            kind=ToolErrorKind.VALIDATION,
        )

    try:
        trace = OtelQueryApiClient().get_trace(
            entity_type, eid, tid, limit=span_limit, offset=span_offset
        )
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)

    spans = trace.get("spans") or []

    truncation: dict[str, Any]
    if view == "summary":
        summaries, stats = summarize_spans(spans)
        result_spans: list[dict[str, Any]] = summaries
        truncation = {
            "mode": "summary",
            "payloads_omitted": True,
            "total_payload_chars": stats["total_payload_chars"],
            "hint": (
                "Fetch one span's payload with otel_span_payload_get(trace_id=..., span_id=...)."
            ),
        }
    else:
        items = [_span_payload_view(span, max_field_chars) for span in spans]
        emitted, budget_stats = apply_char_budget(items, max_total_chars)
        result_spans = emitted
        truncation = {
            "mode": "payloads",
            "spans_returned": budget_stats["spans_returned"],
            "spans_dropped": budget_stats["spans_dropped"],
            "chars_used": budget_stats["chars_used"],
            "max_total_chars": budget_stats["max_total_chars"],
        }

    result: dict[str, Any] = {
        "trace_id": trace.get("trace_id", tid),
        "span_count": trace.get("span_count"),
        "duration": trace.get("duration"),
        "root_span_name": trace.get("root_span_name"),
        "root_service_name": trace.get("root_service_name"),
    }
    if "metrics" in trace:
        result["metrics"] = trace["metrics"]
    result["spans"] = result_spans
    result["truncation"] = truncation
    result["count"] = len(result_spans)

    return merge_pagination_metadata(result, trace, offset=span_offset, limit=span_limit)


# ------------------------------------------------------------------ #
# otel_span_payload_get                                                #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"otel", "datarobot", "span", "payload", "observability", "debug"},
    description=(
        "[OTel—span payload] Drill-down escape hatch: fetch one span's payload "
        "text in full detail, after otel_trace_get(view='summary') has told you "
        "which span to look at.\n\n"
        "By default returns the canonical set (rank-1 prompt/completion plus the "
        "highest-precedence surviving semconv attributes); pass 'fields' with "
        "exact attribute names to fetch specific fields directly, including ones "
        "reported as dropped_as_duplicate or dropped_semconv by a previous call — "
        "the text is still there under its own name, just not duplicated.\n\n"
        "Each field is windowed at max_field_chars. When a field is larger than "
        "that, 'truncation.fields' names it with 'next_offset' — pass that back "
        "as field_offset together with fields=[that one name] (field_offset "
        "requires fields, so an offset meant for one long field cannot blank "
        "every short field beside it); include 'status_message' in fields to "
        "continue a truncated status message. One measured field was 740,000 "
        "chars.\n\n"
        "Known cost: there is no per-span endpoint, so this refetches the trace "
        "page by page and filters client-side until the span is found.\n\n"
        f"{_TRACE_403_HINT}\n\n"
        "Example: otel_span_payload_get(entity_type='deployment', entity_id='...', "
        "trace_id='...', span_id='...')\n"
        "Example (continue a field): otel_span_payload_get(entity_type='deployment', "
        "entity_id='...', trace_id='...', span_id='...', "
        "fields=['gen_ai.task.output'], field_offset=8000)"
    ),
    display_name="OTel — Get span payload",
    description_ui=(
        "Fetch one span's full payload text, windowed field by field, including "
        "fields dropped as duplicates by the trace summary."
    ),
)
async def otel_span_payload_get(
    *,
    entity_type: Annotated[EntityType, "Type of entity the OTel data belongs to."],
    entity_id: Annotated[str, "24-character hex ID of the entity."],
    trace_id: Annotated[str, "32-character hex OTel trace ID."],
    span_id: Annotated[str, "Id of the span within the trace."],
    fields: Annotated[
        list[str] | None,
        "Exact attribute names to fetch, bypassing dedup/semconv selection. "
        "None (default) returns the canonical set.",
    ] = None,
    max_field_chars: Annotated[int, "Cap per field value, in characters."] = 8_000,
    field_offset: Annotated[
        int,
        "Character offset to continue a previously truncated field from. "
        "Requires 'fields' naming the field(s) to continue.",
    ] = 0,
) -> dict[str, Any]:
    eid = require_object_id(entity_id, "entity_id")
    tid = require_trace_id(trace_id)
    sid = require_id(span_id, "span_id")
    if field_offset < 0:
        raise ToolError(
            "Argument validation error: 'field_offset' must be >= 0.",
            kind=ToolErrorKind.VALIDATION,
        )
    if field_offset > 0 and not fields:
        raise ToolError(
            "Argument validation error: 'field_offset' requires 'fields' naming the "
            "field(s) to continue (e.g. fields=['gen_ai.task.output']). Without it, "
            "every field shorter than the offset would come back as an empty window.",
            kind=ToolErrorKind.VALIDATION,
        )

    # The server emits lowercase hex span ids; match case-insensitively so a
    # hand-pasted uppercase id still finds its span instead of a NOT_FOUND.
    sid_lower = sid.lower()

    # No span_limit parameter on this tool (§2.3's signature has none) — the fetch
    # page size is tied to the same DEFAULT_TRACE_SPAN_LIMIT otel_trace_get uses.
    # There is no per-span endpoint either, so page through the trace's spans until
    # the requested one is found: otel_trace_get(span_offset=...) legitimately
    # surfaces span ids past the first page, and stopping at page one made every
    # one of those permanently unreachable from this tool.
    span: dict[str, Any] | None = None
    spans_checked = 0
    fetched_bytes = 0
    total_count: int | None = None
    offset = 0
    try:
        client = OtelQueryApiClient()
        while True:
            trace = client.get_trace(
                entity_type, eid, tid, limit=DEFAULT_TRACE_SPAN_LIMIT, offset=offset
            )
            # Evidence for §10's server-side field-projection ask: whole trace pages
            # are fetched to return one span's payload. ensure_ascii=False matches
            # truncation.py's own char/byte accounting so this approximates what was
            # actually received on the wire, not an escape-expanded blowup of it.
            fetched_bytes += len(json.dumps(trace, default=str, ensure_ascii=False).encode("utf-8"))
            spans = trace.get("spans") or []
            spans_checked += len(spans)
            raw_total = trace.get("total_count")
            if isinstance(raw_total, int):
                total_count = raw_total
            span = next(
                (s for s in spans if str(s.get("span_id") or "").lower() == sid_lower), None
            )
            if span is not None:
                break
            if len(spans) < DEFAULT_TRACE_SPAN_LIMIT:
                break  # short (or empty) page: the server has no more spans
            if total_count is not None and spans_checked >= total_count:
                break
            offset += len(spans)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)

    logger.info(
        "otel_span_payload_get: fetched %d bytes for trace %s to return span %s",
        fetched_bytes,
        tid,
        sid,
    )

    if span is None:
        raise ToolError(
            f"Span '{sid}' not found in trace '{tid}' (checked {spans_checked} of "
            f"{total_count if total_count is not None else spans_checked} spans).",
            kind=ToolErrorKind.NOT_FOUND,
        )

    merged = _merged_payload(span)
    missing: list[str] = []
    dropped_duplicate: list[str] = []
    dropped_semconv: list[str] = []

    status_message_requested = False
    if fields is not None:
        # 'status_message' is structural, not a merged-payload attribute: naming it
        # in 'fields' requests offset continuation of a truncated status message
        # rather than an attribute lookup (which would report it as not found).
        status_message_requested = "status_message" in fields
        requested = [name for name in fields if name != "status_message"]
        missing = [name for name in requested if name not in merged]
        selected = {name: merged[name] for name in requested if name in merged}
    else:
        selected, dropped = canonical_attributes(merged)
        dropped_duplicate = dropped["duplicate"]
        dropped_semconv = dropped["semconv"]

    response_fields: dict[str, Any] = {}
    attributes: dict[str, Any] = {}
    field_windows: dict[str, Any] = {}
    for name, value in selected.items():
        window, info = cap_payload_value(value, max_field_chars, field_offset)
        if info is not None:
            field_windows[name] = info
        if name in RESPONSE_PAYLOAD_FIELDS:
            response_fields[name] = window
        else:
            attributes[name] = window

    # status_message is the one structural field that can carry an unbounded
    # traceback (SpanViewValidator has no server-side max_length on it — see
    # truncation.STATUS_MESSAGE_MAX_CHARS's docstring), so it gets the same
    # max_field_chars window as every other field this tool emits. field_offset
    # applies to it only when 'fields' names it — a continuation offset meant for
    # one long attribute must not blank the (usually short) status message.
    status_message = span.get("status_message")
    if isinstance(status_message, str):
        status_offset = field_offset if status_message_requested else 0
        status_message, status_message_info = cap_value(
            status_message, max_field_chars, status_offset
        )
        if status_message_info is not None:
            field_windows["status_message"] = status_message_info

    truncation: dict[str, Any] = {
        "fields": field_windows,
        "dropped_as_duplicate": dropped_duplicate,
        "dropped_semconv": dropped_semconv,
    }
    if missing:
        truncation["fields_not_found"] = missing

    return {
        "span_id": span.get("span_id") or sid,
        "name": span.get("name"),
        "status_code": span.get("status_code"),
        "status_message": status_message,
        **response_fields,
        "attributes": attributes,
        "truncation": truncation,
    }


# ------------------------------------------------------------------ #
# helpers                                                              #
# ------------------------------------------------------------------ #


def _merged_payload(span: dict[str, Any]) -> dict[str, Any]:
    """Merge a span's rank-1 response fields with its attributes, response wins.

    Response fields are precedence rank 1 (never dropped, win every byte-identity
    tie) — see ``RESPONSE_PAYLOAD_FIELDS`` and ``SEMCONV_PRECEDENCE`` in
    ``constants.py``. ``attributes`` is a dynamic dict on the wire and can carry a
    key literally named ``prompt``/``completion`` beside the span's own response
    field (the same wire shape ``truncation._payload_sizes`` documents and
    accounts for). Overlaying the response fields *after* ``attributes`` — rather
    than merging attributes on top of them — is what makes "never dropped, wins
    every tie" actually true: the response's own normalized text is what
    :func:`canonical_attributes` sees under that name, not whatever an
    instrumentation happened to also write there.
    """
    merged = dict(span.get("attributes") or {})
    merged.update({name: span[name] for name in RESPONSE_PAYLOAD_FIELDS if name in span})
    return merged


def _span_payload_view(span: dict[str, Any], max_field_chars: int) -> dict[str, Any]:
    """Project one span to its canonical, per-field-capped payload (view='payloads')."""
    kept, dropped = canonical_attributes(_merged_payload(span))
    attributes: dict[str, Any] = {}
    field_windows: dict[str, Any] = {}
    for name, value in kept.items():
        window, info = cap_payload_value(value, max_field_chars)
        attributes[name] = window
        if info is not None:
            field_windows[name] = info
    return {
        "span_id": span.get("span_id"),
        "name": span.get("name"),
        "status_code": span.get("status_code"),
        "attributes": attributes,
        "truncation": {
            "fields": field_windows,
            "dropped_as_duplicate": dropped["duplicate"],
            "dropped_semconv": dropped["semconv"],
        },
    }
