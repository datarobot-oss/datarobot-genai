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

"""Unit tests for OtelQueryApiClient.

Style: GIVEN preconditions / WHEN behavior under test / THEN expected outcomes.

The client is pure transport (camelCase param assembly + path construction),
so that assembly is the whole point of these tests. No tools exist yet — the
client is exercised directly, the same way tests/drmcp/unit/workload_tools/
exercises WorkloadApiClient's HTTP calls through request_user_dr_client.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from datarobot_genai.drtools.core.clients.datarobot_otel_query import OtelQueryApiClient


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


@pytest.fixture
def client() -> OtelQueryApiClient:
    return OtelQueryApiClient()


# ------------------------------------------------------------------ #
# list_traces                                                          #
# ------------------------------------------------------------------ #


def test_list_traces_builds_path_from_entity_type_and_id_with_default_params(
    client: OtelQueryApiClient, patched_dr_client: MagicMock
) -> None:
    # GIVEN no optional filters
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"data": []})

    # WHEN list_traces is called with only the required entity args
    result = client.list_traces("deployment", "abc123")

    # THEN the path is built from entity_type/entity_id and only limit/offset are sent
    patched_dr_client.get.assert_called_once_with(
        "otel/deployment/abc123/traces/", params={"limit": 20, "offset": 0}
    )
    assert result == {"data": []}


def test_list_traces_camelizes_every_optional_filter(
    client: OtelQueryApiClient, patched_dr_client: MagicMock
) -> None:
    # GIVEN every optional filter set, including the *_ns duration params
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"data": []})

    # WHEN list_traces is called with all filters
    client.list_traces(
        "use_case",
        "xyz789",
        limit=50,
        offset=10,
        start_time="2026-08-01T00:00:00Z",
        end_time="2026-08-02T00:00:00Z",
        status="error",
        root_span_name=["agent.run"],
        tools=["search"],
        trace_type="gen_ai",
        min_trace_duration_ns=1_000,
        min_span_duration_ns=2_000,
        max_span_duration_ns=3_000,
        min_trace_cost=1,
        max_trace_cost=100,
        sort_by="duration",
        sort_direction="desc",
    )

    # THEN every filter key is camelCase on the wire, and snake_case values
    # (status, trace_type) are passed through unchanged per §2's convention
    patched_dr_client.get.assert_called_once_with(
        "otel/use_case/xyz789/traces/",
        params={
            "limit": 50,
            "offset": 10,
            "startTime": "2026-08-01T00:00:00Z",
            "endTime": "2026-08-02T00:00:00Z",
            "status": "error",
            "rootSpanName": ["agent.run"],
            "tools": ["search"],
            "traceType": "gen_ai",
            "minTraceDuration": 1_000,
            "minSpanDuration": 2_000,
            "maxSpanDuration": 3_000,
            "minTraceCost": 1,
            "maxTraceCost": 100,
            "sortBy": "duration",
            "sortDirection": "desc",
        },
    )


def test_list_traces_omits_unset_optional_filters(
    client: OtelQueryApiClient, patched_dr_client: MagicMock
) -> None:
    # GIVEN optional list filters passed as empty lists (falsy)
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"data": []})

    # WHEN list_traces is called with empty-list filters and no other optionals
    client.list_traces("deployment", "abc123", root_span_name=[], tools=[])

    # THEN the empty lists are not sent as query params
    patched_dr_client.get.assert_called_once_with(
        "otel/deployment/abc123/traces/", params={"limit": 20, "offset": 0}
    )


def test_list_traces_sends_zero_valued_duration_and_cost_filters(
    client: OtelQueryApiClient, patched_dr_client: MagicMock
) -> None:
    # GIVEN duration/cost filters explicitly set to 0 (a valid value, not "unset")
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"data": []})

    # WHEN list_traces is called with those filters at 0
    client.list_traces(
        "deployment",
        "abc123",
        min_trace_duration_ns=0,
        min_trace_cost=0,
    )

    # THEN 0 is sent, not treated as absent (checked via "is not None", not truthiness)
    patched_dr_client.get.assert_called_once_with(
        "otel/deployment/abc123/traces/",
        params={"limit": 20, "offset": 0, "minTraceDuration": 0, "minTraceCost": 0},
    )


def test_list_traces_forwards_fractional_cost_bounds(
    client: OtelQueryApiClient, patched_dr_client: MagicMock
) -> None:
    # GIVEN sub-dollar cost bounds — trace cost is a fractional currency amount
    # (the SDK types it Float; the stub trace's own cost is 0.002)
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"data": []})

    # WHEN list_traces is called with fractional bounds
    client.list_traces("deployment", "abc123", min_trace_cost=0.001, max_trace_cost=0.01)

    # THEN they are forwarded unmangled — an int-typed signature rejected every
    # realistic sub-dollar bound before the request was ever made
    patched_dr_client.get.assert_called_once_with(
        "otel/deployment/abc123/traces/",
        params={"limit": 20, "offset": 0, "minTraceCost": 0.001, "maxTraceCost": 0.01},
    )


# ------------------------------------------------------------------ #
# get_trace                                                            #
# ------------------------------------------------------------------ #


def test_get_trace_builds_path_with_trace_id_and_default_span_pagination(
    client: OtelQueryApiClient, patched_dr_client: MagicMock
) -> None:
    # GIVEN a trace_id
    trace_id = "a" * 32
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"spans": []})

    # WHEN get_trace is called with no explicit pagination
    result = client.get_trace("deployment", "abc123", trace_id)

    # THEN the path includes the trace id and default span limit/offset are sent
    patched_dr_client.get.assert_called_once_with(
        f"otel/deployment/abc123/traces/{trace_id}/", params={"limit": 100, "offset": 0}
    )
    assert result == {"spans": []}


def test_get_trace_forwards_explicit_span_pagination(
    client: OtelQueryApiClient, patched_dr_client: MagicMock
) -> None:
    # GIVEN explicit span_limit/span_offset-equivalent pagination
    trace_id = "b" * 32
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"spans": []})

    # WHEN get_trace is called with limit/offset overrides
    client.get_trace("workload", "wkld-1", trace_id, limit=10, offset=5)

    # THEN limit/offset page the spans, sent unchanged (already lowercase on the wire)
    patched_dr_client.get.assert_called_once_with(
        f"otel/workload/wkld-1/traces/{trace_id}/", params={"limit": 10, "offset": 5}
    )


# ------------------------------------------------------------------ #
# list_logs                                                            #
# ------------------------------------------------------------------ #


def test_list_logs_builds_path_and_defaults_level_to_debug(
    client: OtelQueryApiClient, patched_dr_client: MagicMock
) -> None:
    # GIVEN no optional filters
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"data": []})

    # WHEN list_logs is called with only the required entity args
    client.list_logs("custom_application", "app-1")

    # THEN limit/offset/level default and no other keys are sent
    patched_dr_client.get.assert_called_once_with(
        "otel/custom_application/app-1/logs/",
        params={"limit": 50, "offset": 0, "level": "debug"},
    )


def test_list_logs_camelizes_span_and_trace_id_and_forwards_body_filters(
    client: OtelQueryApiClient, patched_dr_client: MagicMock
) -> None:
    # GIVEN every optional filter set, including the six-level minimum ('warning')
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"data": []})

    # WHEN list_logs is called with all filters
    client.list_logs(
        "deployment",
        "abc123",
        limit=25,
        offset=5,
        level="warning",
        start_time="2026-08-01T00:00:00Z",
        end_time="2026-08-02T00:00:00Z",
        includes=["timeout"],
        excludes=["retry"],
        span_id="span-1",
        trace_id="c" * 32,
    )

    # THEN spanId/traceId are camelCase and includes/excludes pass through as lists
    patched_dr_client.get.assert_called_once_with(
        "otel/deployment/abc123/logs/",
        params={
            "limit": 25,
            "offset": 5,
            "level": "warning",
            "startTime": "2026-08-01T00:00:00Z",
            "endTime": "2026-08-02T00:00:00Z",
            "includes": ["timeout"],
            "excludes": ["retry"],
            "spanId": "span-1",
            "traceId": "c" * 32,
        },
    )


# ------------------------------------------------------------------ #
# list_metrics_summary                                                 #
# ------------------------------------------------------------------ #


def test_list_metrics_summary_sends_no_params_when_unfiltered(
    client: OtelQueryApiClient, patched_dr_client: MagicMock
) -> None:
    # GIVEN no search/metric_type
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"metrics": []})

    # WHEN list_metrics_summary is called with only entity args
    client.list_metrics_summary("deployment", "abc123")

    # THEN no query params are sent at all
    patched_dr_client.get.assert_called_once_with(
        "otel/deployment/abc123/metrics/summary/", params={}
    )


def test_list_metrics_summary_camelizes_metric_type(
    client: OtelQueryApiClient, patched_dr_client: MagicMock
) -> None:
    # GIVEN search and metric_type filters
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"metrics": []})

    # WHEN list_metrics_summary is called with both
    client.list_metrics_summary("deployment", "abc123", search="latency", metric_type="counter")

    # THEN metric_type becomes metricType on the wire
    patched_dr_client.get.assert_called_once_with(
        "otel/deployment/abc123/metrics/summary/",
        params={"search": "latency", "metricType": "counter"},
    )


# ------------------------------------------------------------------ #
# get_metrics_values / get_autocollected_metrics_values                #
# ------------------------------------------------------------------ #


def test_get_metrics_values_always_sends_histogram_buckets(
    client: OtelQueryApiClient, patched_dr_client: MagicMock
) -> None:
    # GIVEN the default (unset) histogram_buckets
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"metric_aggregations": []})

    # WHEN get_metrics_values is called with no other args
    client.get_metrics_values("deployment", "abc123")

    # THEN histogramBuckets is sent even at its False default (not Optional)
    patched_dr_client.get.assert_called_once_with(
        "otel/deployment/abc123/metrics/values/", params={"histogramBuckets": False}
    )


def test_get_metrics_values_forwards_window_and_histogram_flag(
    client: OtelQueryApiClient, patched_dr_client: MagicMock
) -> None:
    # GIVEN a time window and histogram_buckets=True
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"metric_aggregations": []})

    # WHEN get_metrics_values is called with start/end and the flag set
    client.get_metrics_values(
        "deployment",
        "abc123",
        histogram_buckets=True,
        start="2026-08-01T00:00:00Z",
        end="2026-08-02T00:00:00Z",
    )

    # THEN start/end become startTime/endTime and the flag is forwarded
    patched_dr_client.get.assert_called_once_with(
        "otel/deployment/abc123/metrics/values/",
        params={
            "histogramBuckets": True,
            "startTime": "2026-08-01T00:00:00Z",
            "endTime": "2026-08-02T00:00:00Z",
        },
    )


def test_get_autocollected_metrics_values_has_no_histogram_param(
    client: OtelQueryApiClient, patched_dr_client: MagicMock
) -> None:
    # GIVEN a time window (autocollected has no histogram_buckets concept)
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"metrics": []})

    # WHEN get_autocollected_metrics_values is called
    client.get_autocollected_metrics_values(
        "deployment", "abc123", start="2026-08-01T00:00:00Z", end="2026-08-02T00:00:00Z"
    )

    # THEN only startTime/endTime are sent, on the autocollectedValues path
    patched_dr_client.get.assert_called_once_with(
        "otel/deployment/abc123/metrics/autocollectedValues/",
        params={"startTime": "2026-08-01T00:00:00Z", "endTime": "2026-08-02T00:00:00Z"},
    )


def test_get_autocollected_metrics_values_sends_no_params_when_unfiltered(
    client: OtelQueryApiClient, patched_dr_client: MagicMock
) -> None:
    # GIVEN no time window
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"metrics": []})

    # WHEN get_autocollected_metrics_values is called with only entity args
    client.get_autocollected_metrics_values("deployment", "abc123")

    # THEN no query params are sent
    patched_dr_client.get.assert_called_once_with(
        "otel/deployment/abc123/metrics/autocollectedValues/", params={}
    )


# ------------------------------------------------------------------ #
# get_entity_stats                                                     #
# ------------------------------------------------------------------ #


def test_get_entity_stats_hits_the_flat_stats_path_with_service_name_and_default_limit(
    client: OtelQueryApiClient, patched_dr_client: MagicMock
) -> None:
    # GIVEN a pre-built service_name (the caller's job, not this client's)
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"data": []})

    # WHEN get_entity_stats is called with only service_name
    result = client.get_entity_stats("deployment-abc123")

    # THEN the path is the flat /otel/stats/ endpoint, and limit defaults to the
    # validator's ceiling of 1000 so per-user sums are never silently partial
    patched_dr_client.get.assert_called_once_with(
        "otel/stats/", params={"serviceName": "deployment-abc123", "limit": 1000}
    )
    assert result == {"data": []}


def test_get_entity_stats_forwards_time_window(
    client: OtelQueryApiClient, patched_dr_client: MagicMock
) -> None:
    # GIVEN a time window and an explicit limit
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"data": []})

    # WHEN get_entity_stats is called with start/end/limit
    client.get_entity_stats(
        "workload-wkld-1", start="2026-08-01T00:00:00Z", end="2026-08-02T00:00:00Z", limit=500
    )

    # THEN start/end become startTime/endTime and limit is forwarded
    patched_dr_client.get.assert_called_once_with(
        "otel/stats/",
        params={
            "serviceName": "workload-wkld-1",
            "limit": 500,
            "startTime": "2026-08-01T00:00:00Z",
            "endTime": "2026-08-02T00:00:00Z",
        },
    )


# ------------------------------------------------------------------ #
# request_user_dr_client usage                                        #
# ------------------------------------------------------------------ #


def test_each_method_scopes_the_rest_client_to_a_single_with_block(
    client: OtelQueryApiClient, patched_dr_client: MagicMock
) -> None:
    # GIVEN a client call
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"data": []})

    # WHEN list_traces is called
    client.list_traces("deployment", "abc123")

    # THEN exactly one GET is issued through the per-request credentialed client
    assert patched_dr_client.get.call_count == 1
