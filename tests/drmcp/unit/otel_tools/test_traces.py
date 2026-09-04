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

"""Unit tests for the three OTel trace tools.

Style: GIVEN preconditions / WHEN behavior under test / THEN expected outcomes.

Truncation itself is exercised in isolation by test_truncation.py; these tests
cover what the tools add on top: id validation, pagination, error mapping, and
that the tools actually compose the truncation helpers the way the plan
specifies. Numeric budgets that otel_trace_get owns (span_limit, max_field_chars,
max_total_chars) are always read off ``traces.DEFAULT_TRACE_*`` rather than
retyped here, so a later fixup to those defaults (plan §9 step 9) cannot silently
desync the tests from the tool.
"""

import json
import logging
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from datarobot.errors import ClientError

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.otel import traces
from tests.drmcp.unit.otel_tools.trace_factory import OVERSIZED_SPAN_COUNT
from tests.drmcp.unit.otel_tools.trace_factory import OVERSIZED_TOTAL_PAYLOAD_CHARS
from tests.drmcp.unit.otel_tools.trace_factory import OVERSIZED_TRACE_ID
from tests.drmcp.unit.otel_tools.trace_factory import SMALL_TRACE_ID

_ENTITY_ID = "a" * 24


@pytest.fixture
def mock_rest_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def patched_dr_client(mock_rest_client: MagicMock) -> Iterator[MagicMock]:
    with patch(
        "datarobot_genai.drtools.core.clients.datarobot_otel_query.request_user_dr_client"
    ) as mock_cm:
        mock_cm.return_value.__enter__.return_value = mock_rest_client
        mock_cm.return_value.__exit__.return_value = False
        yield mock_rest_client


def _stub_json(client: MagicMock, payload: dict[str, Any]) -> None:
    client.get.return_value = MagicMock(json=lambda: payload)


def _single_span_trace(
    *,
    trace_id: str,
    span_id: str,
    attributes: dict[str, Any],
    completion: str | None = None,
    status_message: str | None = None,
) -> dict[str, Any]:
    """Build a minimal one-span trace envelope for a specific merge/window edge case."""
    span: dict[str, Any] = {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": None,
        "name": "llm.chat",
        "status_code": "ERROR" if status_message else "OK",
        "status_message": status_message,
        "kind": "CLIENT",
        "service_name": "svc",
        "duration": 1.0,
        "start_time": 0.0,
        "attributes": attributes,
        "events": [],
        "links": [],
    }
    if completion is not None:
        span["completion"] = completion
    return {
        "trace_id": trace_id,
        "span_count": 1,
        "duration": 1.0,
        "root_span_name": "llm.chat",
        "root_service_name": "svc",
        "spans": [span],
        "count": 1,
        "offset": 0,
        "limit": 100,
        "total_count": 1,
        "next": None,
        "previous": None,
    }


# ------------------------------------------------------------------ #
# otel_traces_list                                                     #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_otel_traces_list_maps_data_to_traces_and_merges_pagination(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN a raw traces/ response shaped like TracingListResponseValidator
    _stub_json(
        patched_dr_client,
        {
            "data": [{"trace_id": "t1"}, {"trace_id": "t2"}],
            "count": 2,
            "total_count": 40,
            "next": "https://example/traces/?offset=20",
            "previous": None,
        },
    )

    # WHEN otel_traces_list is called with only the required entity args
    result = await traces.otel_traces_list(entity_type="deployment", entity_id=_ENTITY_ID)

    # THEN 'data' becomes 'traces', and pagination metadata is merged in
    patched_dr_client.get.assert_called_once_with(
        f"otel/deployment/{_ENTITY_ID}/traces/", params={"limit": 20, "offset": 0}
    )
    assert result["traces"] == [{"trace_id": "t1"}, {"trace_id": "t2"}]
    assert result["count"] == 2
    assert result["offset"] == 0
    assert result["limit"] == 20
    assert result["total_count"] == 40
    assert result["next"] == "https://example/traces/?offset=20"
    assert "previous" not in result  # merge_pagination_metadata skips None values


@pytest.mark.asyncio
async def test_otel_traces_list_strips_and_validates_entity_id(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN a malformed entity_id
    # WHEN / THEN otel_traces_list rejects it before making any HTTP call
    with pytest.raises(ToolError) as exc_info:
        await traces.otel_traces_list(entity_type="deployment", entity_id="not-hex")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION
    patched_dr_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_otel_traces_list_rejects_negative_offset() -> None:
    # GIVEN a negative offset
    # WHEN / THEN otel_traces_list raises a VALIDATION error
    with pytest.raises(ToolError) as exc_info:
        await traces.otel_traces_list(entity_type="deployment", entity_id=_ENTITY_ID, offset=-1)
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_otel_traces_list_rejects_a_comma_joined_tools_string() -> None:
    # GIVEN a single comma-joined string passed where a list of tool names belongs
    # WHEN / THEN otel_traces_list raises a VALIDATION error naming the fix
    with pytest.raises(ToolError) as exc_info:
        await traces.otel_traces_list(
            entity_type="deployment", entity_id=_ENTITY_ID, tools=["search,browse"]
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION
    assert "comma" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_otel_traces_list_defaults_sort_by_to_timestamp_when_only_direction_given(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN sort_direction with no sort_by (the server 400s this combination)
    _stub_json(patched_dr_client, {"data": []})

    # WHEN otel_traces_list is called
    await traces.otel_traces_list(
        entity_type="deployment", entity_id=_ENTITY_ID, sort_direction="desc"
    )

    # THEN sort_by is defaulted to 'timestamp' before the call is made
    patched_dr_client.get.assert_called_once_with(
        f"otel/deployment/{_ENTITY_ID}/traces/",
        params={"limit": 20, "offset": 0, "sortBy": "timestamp", "sortDirection": "desc"},
    )


@pytest.mark.asyncio
async def test_otel_traces_list_rejects_offset_plus_limit_over_the_ceiling() -> None:
    # GIVEN offset + limit that would exceed the server's 10,000 ceiling
    # WHEN / THEN otel_traces_list pre-empts the server's 400 with a VALIDATION error
    with pytest.raises(ToolError) as exc_info:
        await traces.otel_traces_list(
            entity_type="deployment", entity_id=_ENTITY_ID, offset=9_950, limit=100
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION
    assert "start_time" in str(exc_info.value) or "end_time" in str(exc_info.value)


@pytest.mark.asyncio
async def test_otel_traces_list_clamps_an_oversized_limit_and_reports_a_note(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN a limit above the 100 ceiling
    _stub_json(patched_dr_client, {"data": []})

    # WHEN otel_traces_list is called
    result = await traces.otel_traces_list(
        entity_type="deployment", entity_id=_ENTITY_ID, limit=500
    )

    # THEN the clamped value is what's actually sent, and a note explains it
    patched_dr_client.get.assert_called_once_with(
        f"otel/deployment/{_ENTITY_ID}/traces/", params={"limit": 100, "offset": 0}
    )
    assert result["limit"] == 100
    assert "note" in result


@pytest.mark.asyncio
async def test_otel_traces_list_wraps_a_client_error_as_a_tool_error(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN the upstream call fails
    patched_dr_client.get.side_effect = ClientError("500", status_code=500, json={})

    # WHEN / THEN otel_traces_list raises an UPSTREAM ToolError, not the raw SDK error
    with pytest.raises(ToolError) as exc_info:
        await traces.otel_traces_list(entity_type="deployment", entity_id=_ENTITY_ID)
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


@pytest.mark.asyncio
async def test_otel_traces_list_forwards_fractional_cost_bounds(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN sub-dollar cost bounds — trace cost is a fractional currency amount
    _stub_json(patched_dr_client, {"data": []})

    # WHEN otel_traces_list is called with them
    await traces.otel_traces_list(
        entity_type="deployment", entity_id=_ENTITY_ID, min_trace_cost=0.001, max_trace_cost=0.01
    )

    # THEN they reach the wire unmangled — the int-typed signature rejected every
    # realistic sub-dollar bound at schema validation before the tool body ran
    patched_dr_client.get.assert_called_once_with(
        f"otel/deployment/{_ENTITY_ID}/traces/",
        params={"limit": 20, "offset": 0, "minTraceCost": 0.001, "maxTraceCost": 0.01},
    )


# ------------------------------------------------------------------ #
# otel_trace_get                                                       #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_otel_trace_get_view_summary_default_projects_the_oversized_trace(
    patched_dr_client: MagicMock, oversized_agent_trace: dict[str, Any]
) -> None:
    # GIVEN the oversized trace: 12 spans, 1,022,000 payload chars
    _stub_json(patched_dr_client, oversized_agent_trace)

    # WHEN otel_trace_get is called with the default view
    result = await traces.otel_trace_get(
        entity_type="deployment", entity_id=_ENTITY_ID, trace_id=OVERSIZED_TRACE_ID
    )

    # THEN the trace-level structural fields, metrics, and truncation hint are
    # present, and no span carries a payload
    assert result["trace_id"] == OVERSIZED_TRACE_ID
    assert result["span_count"] == OVERSIZED_SPAN_COUNT
    assert result["root_span_name"] == "agent.run"
    assert result["metrics"] == oversized_agent_trace["metrics"]
    assert len(result["spans"]) == OVERSIZED_SPAN_COUNT
    assert result["count"] == OVERSIZED_SPAN_COUNT
    for span in result["spans"]:
        assert "attributes" not in span
        assert "prompt" not in span
        assert "completion" not in span
    assert result["truncation"] == {
        "mode": "summary",
        "payloads_omitted": True,
        "total_payload_chars": OVERSIZED_TOTAL_PAYLOAD_CHARS,
        "hint": ("Fetch one span's payload with otel_span_payload_get(trace_id=..., span_id=...)."),
    }
    # THEN pagination metadata is merged from the raw response plus the request
    assert result["offset"] == 0
    assert result["limit"] == traces.DEFAULT_TRACE_SPAN_LIMIT
    assert result["total_count"] == oversized_agent_trace["total_count"]
    assert "next" not in result  # the fixture's next/previous are both None


@pytest.mark.asyncio
async def test_otel_trace_get_view_summary_stays_under_the_measured_ceiling(
    patched_dr_client: MagicMock, oversized_agent_trace: dict[str, Any]
) -> None:
    """The regression guard from §7, exercised through the tool itself.

    test_truncation.py already pins summarize_spans' own output to a per-span
    ceiling; this asserts the same property survives composition into the full
    tool response (trace envelope + pagination + truncation block), so a future
    change that re-attaches raw span data at the tool layer would still be caught
    even though it wouldn't touch summarize_spans at all.
    """
    # GIVEN the oversized trace
    _stub_json(patched_dr_client, oversized_agent_trace)

    # WHEN the full tool response is serialized as it would be returned
    result = await traces.otel_trace_get(
        entity_type="deployment", entity_id=_ENTITY_ID, trace_id=OVERSIZED_TRACE_ID
    )
    serialized_chars = len(json.dumps(result))

    # THEN the whole response is still a rounding error against the payload it
    # describes, not merely the span summaries in isolation
    assert serialized_chars < OVERSIZED_TOTAL_PAYLOAD_CHARS / 50, (
        f"otel_trace_get(view='summary') response grew to {serialized_chars} chars "
        f"against {OVERSIZED_TOTAL_PAYLOAD_CHARS} payload chars it describes"
    )


@pytest.mark.asyncio
async def test_otel_trace_get_uses_the_shared_default_span_pagination(
    patched_dr_client: MagicMock, small_trace: dict[str, Any]
) -> None:
    # GIVEN no explicit span_limit/span_offset
    _stub_json(patched_dr_client, small_trace)

    # WHEN otel_trace_get is called
    await traces.otel_trace_get(
        entity_type="workload", entity_id=_ENTITY_ID, trace_id=SMALL_TRACE_ID
    )

    # THEN the client is called with the single named default (traces.DEFAULT_TRACE_SPAN_LIMIT)
    patched_dr_client.get.assert_called_once_with(
        f"otel/workload/{_ENTITY_ID}/traces/{SMALL_TRACE_ID}/",
        params={"limit": traces.DEFAULT_TRACE_SPAN_LIMIT, "offset": 0},
    )


@pytest.mark.asyncio
async def test_otel_trace_get_forwards_explicit_span_pagination(
    patched_dr_client: MagicMock, small_trace: dict[str, Any]
) -> None:
    # GIVEN explicit span_limit/span_offset
    _stub_json(patched_dr_client, small_trace)

    # WHEN otel_trace_get is called with overrides
    await traces.otel_trace_get(
        entity_type="workload",
        entity_id=_ENTITY_ID,
        trace_id=SMALL_TRACE_ID,
        span_limit=5,
        span_offset=10,
    )

    # THEN the overrides, not the defaults, are sent
    patched_dr_client.get.assert_called_once_with(
        f"otel/workload/{_ENTITY_ID}/traces/{SMALL_TRACE_ID}/",
        params={"limit": 5, "offset": 10},
    )


@pytest.mark.asyncio
async def test_otel_trace_get_view_payloads_fits_the_default_budget_untouched(
    patched_dr_client: MagicMock, oversized_agent_trace: dict[str, Any]
) -> None:
    # GIVEN the oversized trace and otel_trace_get's own defaults
    # (max_field_chars=2,000 keeps this particular fixture's canonical payload
    # small enough that the 60,000-char budget never has to drop anything --
    # unlike the real 4.76 MB trace the plan measured, whose canonical payload
    # alone was ~65k tokens even at an 8,000-char field cap)
    _stub_json(patched_dr_client, oversized_agent_trace)

    # WHEN otel_trace_get is called with view='payloads' and no overrides
    result = await traces.otel_trace_get(
        entity_type="deployment",
        entity_id=_ENTITY_ID,
        trace_id=OVERSIZED_TRACE_ID,
        view="payloads",
    )

    # THEN nothing is dropped, and every span carries canonical attributes only
    assert result["truncation"]["mode"] == "payloads"
    assert result["truncation"]["spans_dropped"] == 0
    assert result["truncation"]["spans_returned"] == OVERSIZED_SPAN_COUNT
    assert result["count"] == OVERSIZED_SPAN_COUNT
    assert result["truncation"]["chars_used"] <= traces.DEFAULT_TRACE_MAX_TOTAL_CHARS
    for span in result["spans"]:
        assert "attributes" in span
        assert "truncation" in span


@pytest.mark.asyncio
async def test_otel_trace_get_view_payloads_drops_spans_when_the_budget_is_tight(
    patched_dr_client: MagicMock, oversized_agent_trace: dict[str, Any]
) -> None:
    # GIVEN a wider per-field cap (8,000 chars) that pushes the canonical payload
    # over the default 60,000-char total budget -- the same combination
    # test_truncation.py exercises directly against apply_char_budget
    _stub_json(patched_dr_client, oversized_agent_trace)

    # WHEN otel_trace_get is called with that wider field cap
    result = await traces.otel_trace_get(
        entity_type="deployment",
        entity_id=_ENTITY_ID,
        trace_id=OVERSIZED_TRACE_ID,
        view="payloads",
        max_field_chars=8_000,
    )

    # THEN the hard stop holds and the loss is reported, never silent
    truncation = result["truncation"]
    assert truncation["spans_dropped"] > 0
    assert truncation["spans_returned"] + truncation["spans_dropped"] == OVERSIZED_SPAN_COUNT
    assert truncation["chars_used"] <= traces.DEFAULT_TRACE_MAX_TOTAL_CHARS
    assert result["count"] == truncation["spans_returned"]
    assert truncation["max_total_chars"] == traces.DEFAULT_TRACE_MAX_TOTAL_CHARS


@pytest.mark.asyncio
async def test_otel_trace_get_view_payloads_caps_container_attributes_within_the_budget(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN a two-span trace whose first span carries a 100k-char message list —
    # uncapped, its serialized cost alone blows the whole 60,000-char budget and
    # drops it plus every span after it
    trace_id = "e" * 32
    messages = [{"role": "user", "content": "x" * 100_000}]
    trace = _single_span_trace(
        trace_id=trace_id, span_id="f" * 16, attributes={"gen_ai.input.messages": messages}
    )
    second = dict(trace["spans"][0])
    second["span_id"] = "a" * 16
    second["attributes"] = {"gen_ai.tool.name": "search"}
    trace["spans"] = [trace["spans"][0], second]
    trace["span_count"] = trace["count"] = trace["total_count"] = 2
    _stub_json(patched_dr_client, trace)

    # WHEN view='payloads' is requested with the defaults
    result = await traces.otel_trace_get(
        entity_type="deployment", entity_id=_ENTITY_ID, trace_id=trace_id, view="payloads"
    )

    # THEN both spans are returned — the container was windowed at max_field_chars
    # like any string payload — and the window is marked as serialized JSON
    assert result["truncation"]["spans_dropped"] == 0
    assert result["count"] == 2
    first_span = result["spans"][0]
    window = first_span["attributes"]["gen_ai.input.messages"]
    assert isinstance(window, str)
    assert len(window) == traces.DEFAULT_TRACE_MAX_FIELD_CHARS
    assert first_span["truncation"]["fields"]["gen_ai.input.messages"]["serialized"] is True


@pytest.mark.asyncio
async def test_otel_trace_get_small_trace_survives_both_views_untouched(
    patched_dr_client: MagicMock, small_trace: dict[str, Any]
) -> None:
    # GIVEN the small non-agentic trace, which carries no metrics key at all
    _stub_json(patched_dr_client, small_trace)

    # WHEN both views are requested
    summary = await traces.otel_trace_get(
        entity_type="workload", entity_id=_ENTITY_ID, trace_id=SMALL_TRACE_ID
    )
    _stub_json(patched_dr_client, small_trace)
    payloads = await traces.otel_trace_get(
        entity_type="workload",
        entity_id=_ENTITY_ID,
        trace_id=SMALL_TRACE_ID,
        view="payloads",
    )

    # THEN nothing is truncated or dropped, and the absent 'metrics' key is not
    # fabricated
    assert "metrics" not in summary
    assert "metrics" not in payloads
    assert summary["count"] == len(small_trace["spans"])
    assert payloads["count"] == len(small_trace["spans"])
    assert payloads["truncation"]["spans_dropped"] == 0
    assert [s["name"] for s in summary["spans"]] == ["GET /health", "db.query", "cache.get"]


@pytest.mark.asyncio
async def test_otel_trace_get_rejects_a_malformed_trace_id() -> None:
    # GIVEN a trace_id that is not 32 hex characters
    # WHEN / THEN otel_trace_get raises a VALIDATION error
    with pytest.raises(ToolError) as exc_info:
        await traces.otel_trace_get(
            entity_type="deployment", entity_id=_ENTITY_ID, trace_id="too-short"
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_otel_trace_get_rejects_negative_span_offset() -> None:
    # GIVEN a negative span_offset
    # WHEN / THEN otel_trace_get raises a VALIDATION error
    with pytest.raises(ToolError) as exc_info:
        await traces.otel_trace_get(
            entity_type="deployment",
            entity_id=_ENTITY_ID,
            trace_id=OVERSIZED_TRACE_ID,
            span_offset=-1,
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_otel_trace_get_rejects_a_non_positive_span_limit() -> None:
    # GIVEN a span_limit of zero (the server requires > 0)
    # WHEN / THEN otel_trace_get raises a VALIDATION error before calling the server
    with pytest.raises(ToolError) as exc_info:
        await traces.otel_trace_get(
            entity_type="deployment",
            entity_id=_ENTITY_ID,
            trace_id=OVERSIZED_TRACE_ID,
            span_limit=0,
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_otel_trace_get_wraps_a_404_as_not_found(patched_dr_client: MagicMock) -> None:
    # GIVEN the upstream call 404s
    patched_dr_client.get.side_effect = ClientError("404", status_code=404, json={})

    # WHEN / THEN otel_trace_get raises a NOT_FOUND ToolError
    with pytest.raises(ToolError) as exc_info:
        await traces.otel_trace_get(
            entity_type="deployment", entity_id=_ENTITY_ID, trace_id=OVERSIZED_TRACE_ID
        )
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


# ------------------------------------------------------------------ #
# otel_span_payload_get                                                #
# ------------------------------------------------------------------ #

# Span index 3 ("tool.execute") carries a single 160,000-char value written four
# times over by four instrumentations; its canonical set collapses to
# {"completion", "gen_ai.tool.name", "gen_ai.tool.call.id"} with the other three
# names reported as duplicates -- verified against the actual fixture output
# before writing these assertions, not derived from the docstrings alone.
_GIANT_SPAN_INDEX = 3


@pytest.mark.asyncio
async def test_otel_span_payload_get_default_canonical_set_windows_the_giant_field(
    patched_dr_client: MagicMock, oversized_agent_trace: dict[str, Any]
) -> None:
    # GIVEN the tool.execute span whose completion is 160,000 chars
    span = oversized_agent_trace["spans"][_GIANT_SPAN_INDEX]
    _stub_json(patched_dr_client, oversized_agent_trace)

    # WHEN otel_span_payload_get is called with no fields override
    result = await traces.otel_span_payload_get(
        entity_type="deployment",
        entity_id=_ENTITY_ID,
        trace_id=OVERSIZED_TRACE_ID,
        span_id=span["span_id"],
    )

    # THEN the canonical field is windowed at the 8,000-char default and its
    # duplicates are named, not silently dropped
    assert result["span_id"] == span["span_id"]
    assert result["name"] == "tool.execute"
    assert result["completion"] == span["completion"][:8_000]
    assert result["attributes"] == {
        "gen_ai.tool.name": "crm_account_export",
        "gen_ai.tool.call.id": "call_7Kq1",
    }
    assert result["truncation"]["fields"] == {
        "completion": {"returned_chars": 8_000, "total_chars": 160_000, "next_offset": 8_000}
    }
    assert set(result["truncation"]["dropped_as_duplicate"]) == {
        "gen_ai.task.output",
        "input.value",
        "traceloop.entity.output",
    }
    assert result["truncation"]["dropped_semconv"] == []
    assert "fields_not_found" not in result["truncation"]


@pytest.mark.asyncio
async def test_otel_span_payload_get_fields_param_fetches_a_dropped_duplicate_by_name(
    patched_dr_client: MagicMock, oversized_agent_trace: dict[str, Any]
) -> None:
    # GIVEN the same span, and a name the default call above reports as a
    # dropped duplicate
    span = oversized_agent_trace["spans"][_GIANT_SPAN_INDEX]
    _stub_json(patched_dr_client, oversized_agent_trace)

    # WHEN otel_span_payload_get is called with that exact name in 'fields'
    result = await traces.otel_span_payload_get(
        entity_type="deployment",
        entity_id=_ENTITY_ID,
        trace_id=OVERSIZED_TRACE_ID,
        span_id=span["span_id"],
        fields=["traceloop.entity.output"],
    )

    # THEN the escape hatch bypasses dedup entirely: the field comes back
    # directly, and nothing is reported as dropped in this mode
    assert result["attributes"] == {
        "traceloop.entity.output": span["attributes"]["traceloop.entity.output"][:8_000]
    }
    assert "completion" not in result
    assert result["truncation"]["dropped_as_duplicate"] == []
    assert result["truncation"]["dropped_semconv"] == []


@pytest.mark.asyncio
async def test_otel_span_payload_get_field_offset_continues_a_truncated_field(
    patched_dr_client: MagicMock, oversized_agent_trace: dict[str, Any]
) -> None:
    # GIVEN the first call's next_offset for the giant completion field
    span = oversized_agent_trace["spans"][_GIANT_SPAN_INDEX]
    _stub_json(patched_dr_client, oversized_agent_trace)

    # WHEN a follow-up call continues from that offset
    result = await traces.otel_span_payload_get(
        entity_type="deployment",
        entity_id=_ENTITY_ID,
        trace_id=OVERSIZED_TRACE_ID,
        span_id=span["span_id"],
        fields=["completion"],
        field_offset=8_000,
    )

    # THEN the window picks up exactly where the previous one left off
    assert result["completion"] == span["completion"][8_000:16_000]
    assert result["truncation"]["fields"]["completion"] == {
        "returned_chars": 8_000,
        "total_chars": 160_000,
        "next_offset": 16_000,
    }


@pytest.mark.asyncio
async def test_otel_span_payload_get_reports_fields_not_found(
    patched_dr_client: MagicMock, oversized_agent_trace: dict[str, Any]
) -> None:
    # GIVEN a request for a field name that does not exist on the span
    span = oversized_agent_trace["spans"][_GIANT_SPAN_INDEX]
    _stub_json(patched_dr_client, oversized_agent_trace)

    # WHEN otel_span_payload_get is called with a mix of a real and a bogus name
    result = await traces.otel_span_payload_get(
        entity_type="deployment",
        entity_id=_ENTITY_ID,
        trace_id=OVERSIZED_TRACE_ID,
        span_id=span["span_id"],
        fields=["gen_ai.tool.name", "no.such.field"],
    )

    # THEN the real field is returned and the missing one is named, not silent
    assert result["attributes"] == {"gen_ai.tool.name": "crm_account_export"}
    assert result["truncation"]["fields_not_found"] == ["no.such.field"]


@pytest.mark.asyncio
async def test_otel_span_payload_get_response_field_wins_over_a_colliding_attribute_name(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN a span whose 'attributes' dict happens to carry a literal
    # "completion" key beside the span's own rank-1 response field -- the exact
    # wire shape truncation._payload_sizes already documents and accounts for
    trace_id = "a" * 32
    span_id = "b" * 16
    real_completion = "REAL response-level completion text. " * 10
    fake_attribute_completion = "FAKE attribute-level completion text. " * 10
    trace = _single_span_trace(
        trace_id=trace_id,
        span_id=span_id,
        completion=real_completion,
        attributes={"completion": fake_attribute_completion},
    )
    _stub_json(patched_dr_client, trace)

    # WHEN the default canonical set is requested
    result = await traces.otel_span_payload_get(
        entity_type="deployment", entity_id=_ENTITY_ID, trace_id=trace_id, span_id=span_id
    )

    # THEN the true, platform-normalized response text wins -- never the
    # colliding attribute value -- and nothing reports it as dropped, because
    # nothing about the response field was dropped
    assert result["completion"] == real_completion
    assert fake_attribute_completion not in json.dumps(result)
    assert result["truncation"]["dropped_as_duplicate"] == []
    assert result["truncation"]["dropped_semconv"] == []

    # AND the explicit 'fields' escape hatch -- meant to be the guaranteed
    # ground-truth path -- resolves through the same guarantee
    _stub_json(patched_dr_client, trace)
    result_fields = await traces.otel_span_payload_get(
        entity_type="deployment",
        entity_id=_ENTITY_ID,
        trace_id=trace_id,
        span_id=span_id,
        fields=["completion"],
    )
    assert result_fields["completion"] == real_completion


@pytest.mark.asyncio
async def test_otel_trace_get_view_payloads_response_field_wins_over_a_colliding_attribute_name(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN the same colliding wire shape, exercised through otel_trace_get's
    # other call site onto the same _merged_payload helper
    trace_id = "e" * 32
    span_id = "f" * 16
    real_completion = "REAL response-level completion text. " * 10
    fake_attribute_completion = "FAKE attribute-level completion text. " * 10
    trace = _single_span_trace(
        trace_id=trace_id,
        span_id=span_id,
        completion=real_completion,
        attributes={"completion": fake_attribute_completion},
    )
    _stub_json(patched_dr_client, trace)

    # WHEN view='payloads' is requested
    result = await traces.otel_trace_get(
        entity_type="deployment", entity_id=_ENTITY_ID, trace_id=trace_id, view="payloads"
    )

    # THEN the response field's own text survives under 'completion', not the
    # colliding attribute value
    assert result["spans"][0]["attributes"]["completion"] == real_completion
    assert fake_attribute_completion not in json.dumps(result)


@pytest.mark.asyncio
async def test_otel_span_payload_get_windows_a_long_status_message(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN an ERROR span whose status_message is far larger than max_field_chars
    # -- SpanViewValidator.status_message has no server-side max_length, so a
    # traceback can arrive verbatim (see truncation.STATUS_MESSAGE_MAX_CHARS)
    trace_id = "c" * 32
    span_id = "d" * 16
    long_status_message = "Traceback (most recent call last):\n" + ("frame line. " * 2_000)
    trace = _single_span_trace(
        trace_id=trace_id,
        span_id=span_id,
        attributes={},
        status_message=long_status_message,
    )
    _stub_json(patched_dr_client, trace)

    # WHEN otel_span_payload_get is called with a small max_field_chars
    result = await traces.otel_span_payload_get(
        entity_type="deployment",
        entity_id=_ENTITY_ID,
        trace_id=trace_id,
        span_id=span_id,
        max_field_chars=100,
    )

    # THEN status_message is windowed exactly like any other field this tool
    # emits, not returned raw and unbounded
    assert result["status_message"] == long_status_message[:100]
    assert result["truncation"]["fields"]["status_message"] == {
        "returned_chars": 100,
        "total_chars": len(long_status_message),
        "next_offset": 100,
    }


@pytest.mark.asyncio
async def test_otel_span_payload_get_raises_not_found_for_an_absent_span_id(
    patched_dr_client: MagicMock, oversized_agent_trace: dict[str, Any]
) -> None:
    # GIVEN a span_id that is not in the fetched trace
    _stub_json(patched_dr_client, oversized_agent_trace)

    # WHEN / THEN otel_span_payload_get raises NOT_FOUND, not a KeyError
    with pytest.raises(ToolError) as exc_info:
        await traces.otel_span_payload_get(
            entity_type="deployment",
            entity_id=_ENTITY_ID,
            trace_id=OVERSIZED_TRACE_ID,
            span_id="0" * 16,
        )
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


@pytest.mark.asyncio
async def test_otel_span_payload_get_logs_the_fetched_byte_count(
    patched_dr_client: MagicMock,
    oversized_agent_trace: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # GIVEN the oversized trace (there is no per-span endpoint, so the whole
    # trace is fetched to answer a one-span request -- §2.3's known cost)
    span = oversized_agent_trace["spans"][_GIANT_SPAN_INDEX]
    _stub_json(patched_dr_client, oversized_agent_trace)

    # WHEN otel_span_payload_get is called
    with caplog.at_level(logging.INFO, logger="datarobot_genai.drtools.otel.traces"):
        await traces.otel_span_payload_get(
            entity_type="deployment",
            entity_id=_ENTITY_ID,
            trace_id=OVERSIZED_TRACE_ID,
            span_id=span["span_id"],
        )

    # THEN the fetched byte count is logged -- the evidence for §10's server-side
    # field-projection ask -- and it is not a trivially small number
    messages = [record.message for record in caplog.records if "fetched" in record.message]
    assert len(messages) == 1
    assert str(OVERSIZED_TRACE_ID) in messages[0]
    assert any(char.isdigit() for char in messages[0])


@pytest.mark.asyncio
async def test_otel_span_payload_get_fetches_with_the_shared_default_span_limit(
    patched_dr_client: MagicMock, oversized_agent_trace: dict[str, Any]
) -> None:
    # GIVEN no per-span endpoint exists, so the tool must page the trace itself
    span = oversized_agent_trace["spans"][0]
    _stub_json(patched_dr_client, oversized_agent_trace)

    # WHEN otel_span_payload_get is called
    await traces.otel_span_payload_get(
        entity_type="deployment",
        entity_id=_ENTITY_ID,
        trace_id=OVERSIZED_TRACE_ID,
        span_id=span["span_id"],
    )

    # THEN it fetches using the same named default otel_trace_get uses, not an
    # independently hardcoded number
    patched_dr_client.get.assert_called_once_with(
        f"otel/deployment/{_ENTITY_ID}/traces/{OVERSIZED_TRACE_ID}/",
        params={"limit": traces.DEFAULT_TRACE_SPAN_LIMIT, "offset": 0},
    )


@pytest.mark.asyncio
async def test_otel_span_payload_get_rejects_a_malformed_entity_id() -> None:
    # GIVEN a malformed entity_id
    # WHEN / THEN otel_span_payload_get raises a VALIDATION error
    with pytest.raises(ToolError) as exc_info:
        await traces.otel_span_payload_get(
            entity_type="deployment",
            entity_id="short",
            trace_id=OVERSIZED_TRACE_ID,
            span_id="anything",
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_otel_span_payload_get_rejects_an_empty_span_id() -> None:
    # GIVEN a blank span_id
    # WHEN / THEN otel_span_payload_get raises a VALIDATION error
    with pytest.raises(ToolError) as exc_info:
        await traces.otel_span_payload_get(
            entity_type="deployment",
            entity_id=_ENTITY_ID,
            trace_id=OVERSIZED_TRACE_ID,
            span_id="   ",
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_otel_span_payload_get_rejects_a_negative_field_offset() -> None:
    # GIVEN a negative field_offset
    # WHEN / THEN otel_span_payload_get raises a VALIDATION error
    with pytest.raises(ToolError) as exc_info:
        await traces.otel_span_payload_get(
            entity_type="deployment",
            entity_id=_ENTITY_ID,
            trace_id=OVERSIZED_TRACE_ID,
            span_id="anything",
            field_offset=-1,
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_otel_span_payload_get_wraps_a_client_error_as_a_tool_error(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN the upstream call fails with a non-404 error
    patched_dr_client.get.side_effect = ClientError("500", status_code=500, json={})

    # WHEN / THEN otel_span_payload_get raises an UPSTREAM ToolError
    with pytest.raises(ToolError) as exc_info:
        await traces.otel_span_payload_get(
            entity_type="deployment",
            entity_id=_ENTITY_ID,
            trace_id=OVERSIZED_TRACE_ID,
            span_id="anything",
        )
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


@pytest.mark.asyncio
async def test_otel_span_payload_get_rejects_field_offset_without_fields(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN a continuation offset with no fields naming what to continue — every
    # field shorter than the offset would come back as an empty window
    # WHEN / THEN the tool raises a VALIDATION error before any HTTP call
    with pytest.raises(ToolError) as exc_info:
        await traces.otel_span_payload_get(
            entity_type="deployment",
            entity_id=_ENTITY_ID,
            trace_id=OVERSIZED_TRACE_ID,
            span_id="anything",
            field_offset=8_000,
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION
    assert "fields" in str(exc_info.value)
    patched_dr_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_otel_span_payload_get_field_offset_does_not_blank_the_status_message(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN an ERROR span with a short status message and one long attribute
    trace_id = "c" * 32
    span_id = "d" * 16
    long_output = "o" * 20_000
    trace = _single_span_trace(
        trace_id=trace_id,
        span_id=span_id,
        attributes={"gen_ai.task.output": long_output},
        status_message="RateLimitError: 429",
    )
    _stub_json(patched_dr_client, trace)

    # WHEN a continuation call pages the long attribute past the message's length
    result = await traces.otel_span_payload_get(
        entity_type="deployment",
        entity_id=_ENTITY_ID,
        trace_id=trace_id,
        span_id=span_id,
        fields=["gen_ai.task.output"],
        field_offset=8_000,
    )

    # THEN the attribute window continues while the short status message comes
    # back whole — an offset meant for one field no longer blanks the rest
    assert result["attributes"]["gen_ai.task.output"] == long_output[8_000:16_000]
    assert result["status_message"] == "RateLimitError: 429"
    assert "status_message" not in result["truncation"]["fields"]


@pytest.mark.asyncio
async def test_otel_span_payload_get_continues_a_status_message_named_in_fields(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN an ERROR span whose long status_message a first call truncated
    trace_id = "c" * 32
    span_id = "d" * 16
    long_status_message = "Traceback (most recent call last):\n" + ("frame line. " * 2_000)
    trace = _single_span_trace(
        trace_id=trace_id, span_id=span_id, attributes={}, status_message=long_status_message
    )
    _stub_json(patched_dr_client, trace)

    # WHEN the continuation names 'status_message' in fields with an offset
    result = await traces.otel_span_payload_get(
        entity_type="deployment",
        entity_id=_ENTITY_ID,
        trace_id=trace_id,
        span_id=span_id,
        fields=["status_message"],
        field_offset=100,
        max_field_chars=100,
    )

    # THEN the status message window continues from the offset, and the name is
    # not reported as fields_not_found — it is structural, not an attribute
    assert result["status_message"] == long_status_message[100:200]
    assert result["truncation"]["fields"]["status_message"]["next_offset"] == 200
    assert "fields_not_found" not in result["truncation"]


@pytest.mark.asyncio
async def test_otel_span_payload_get_windows_an_oversized_container_attribute(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN a span whose gen_ai.prompt is a JSON-decoded message list far larger
    # than max_field_chars — the OTLP array shape that used to come back whole,
    # uncapped and with no truncation record
    trace_id = "a" * 32
    span_id = "b" * 16
    messages = [{"role": "user", "content": "x" * 100_000}]
    serialized = json.dumps(messages, default=str, ensure_ascii=False)
    trace = _single_span_trace(
        trace_id=trace_id, span_id=span_id, attributes={"gen_ai.prompt": messages}
    )
    _stub_json(patched_dr_client, trace)

    # WHEN the span payload is fetched with the default 8,000-char field cap
    result = await traces.otel_span_payload_get(
        entity_type="deployment",
        entity_id=_ENTITY_ID,
        trace_id=trace_id,
        span_id=span_id,
        fields=["gen_ai.prompt"],
    )

    # THEN the value is windowed over its serialized JSON with a truncation
    # record, instead of being returned whole
    assert result["attributes"]["gen_ai.prompt"] == serialized[:8_000]
    assert result["truncation"]["fields"]["gen_ai.prompt"] == {
        "returned_chars": 8_000,
        "total_chars": len(serialized),
        "next_offset": 8_000,
        "serialized": True,
    }


@pytest.mark.asyncio
async def test_otel_span_payload_get_matches_ids_case_insensitively(
    patched_dr_client: MagicMock, oversized_agent_trace: dict[str, Any]
) -> None:
    # GIVEN a trace id and span id pasted in uppercase — the server emits
    # lowercase hex, and an exact == match raised NOT_FOUND for a span that exists
    span = oversized_agent_trace["spans"][_GIANT_SPAN_INDEX]
    _stub_json(patched_dr_client, oversized_agent_trace)

    # WHEN the tool is called with uppercase ids
    result = await traces.otel_span_payload_get(
        entity_type="deployment",
        entity_id=_ENTITY_ID,
        trace_id=OVERSIZED_TRACE_ID.upper(),
        span_id=span["span_id"].upper(),
    )

    # THEN the span is found, the URL carries the lowercased trace id, and the
    # response reports the server's own span id casing
    assert result["span_id"] == span["span_id"]
    call_path = patched_dr_client.get.call_args[0][0]
    assert f"/traces/{OVERSIZED_TRACE_ID}/" in call_path


def _minimal_span(index: int, span_id: str | None = None) -> dict[str, Any]:
    """Build the smallest span _span_payload_view and the id match can read."""
    return {
        "span_id": span_id or f"{index:016x}",
        "name": f"span-{index}",
        "status_code": "OK",
        "status_message": None,
        "attributes": {},
    }


def _span_page(
    trace_id: str, spans: list[dict[str, Any]], offset: int, total_count: int
) -> dict[str, Any]:
    """Wrap one server-side span page in the trace-detail envelope."""
    return {
        "trace_id": trace_id,
        "spans": spans,
        "count": len(spans),
        "offset": offset,
        "limit": traces.DEFAULT_TRACE_SPAN_LIMIT,
        "total_count": total_count,
        "next": None,
        "previous": None,
    }


@pytest.mark.asyncio
async def test_otel_span_payload_get_pages_past_the_first_span_page(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN a 150-span trace whose target span sits on the second server page —
    # otel_trace_get(span_offset=100) legitimately surfaces such span ids
    trace_id = "a" * 32
    target_span_id = "beef000000000120"
    first_page = _span_page(trace_id, [_minimal_span(i) for i in range(100)], 0, 150)
    second_page = _span_page(
        trace_id,
        [_minimal_span(100 + i) for i in range(20)]
        + [_minimal_span(120, target_span_id)]
        + [_minimal_span(121 + i) for i in range(29)],
        100,
        150,
    )
    patched_dr_client.get.side_effect = [
        MagicMock(json=lambda: first_page),
        MagicMock(json=lambda: second_page),
    ]

    # WHEN the drill-down tool is asked for that span
    result = await traces.otel_span_payload_get(
        entity_type="deployment",
        entity_id=_ENTITY_ID,
        trace_id=trace_id,
        span_id=target_span_id,
    )

    # THEN the tool followed the pagination instead of raising NOT_FOUND
    assert result["span_id"] == target_span_id
    assert patched_dr_client.get.call_count == 2
    patched_dr_client.get.assert_any_call(
        f"otel/deployment/{_ENTITY_ID}/traces/{trace_id}/",
        params={"limit": traces.DEFAULT_TRACE_SPAN_LIMIT, "offset": 100},
    )


@pytest.mark.asyncio
async def test_otel_span_payload_get_raises_not_found_after_checking_every_page(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN the same 150-span trace and a span id that is on none of its pages
    trace_id = "a" * 32
    first_page = _span_page(trace_id, [_minimal_span(i) for i in range(100)], 0, 150)
    second_page = _span_page(trace_id, [_minimal_span(100 + i) for i in range(50)], 100, 150)
    patched_dr_client.get.side_effect = [
        MagicMock(json=lambda: first_page),
        MagicMock(json=lambda: second_page),
    ]

    # WHEN / THEN the tool raises NOT_FOUND only after checking every page,
    # and the error reports the full count it checked ("f" * 16 collides with
    # no generated id — _minimal_span ids are zero-padded hex indexes)
    with pytest.raises(ToolError) as exc_info:
        await traces.otel_span_payload_get(
            entity_type="deployment",
            entity_id=_ENTITY_ID,
            trace_id=trace_id,
            span_id="f" * 16,
        )
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND
    assert "150" in str(exc_info.value)
    assert patched_dr_client.get.call_count == 2
