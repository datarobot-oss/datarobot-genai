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

"""OTel query API methods backed by the per-request DataRobot REST client.

Named ``datarobot_otel_query.py``, not ``datarobot_otel.py``, to stay visually
distinct from ``core/telemetry/datarobot_otel.py`` (which *emits* telemetry).
Same reason the class below is ``OtelQueryApiClient``, not ``OtelApiClient`` —
this package reads OTel data; it does not produce any.

Direct mirror of :class:`WorkloadApiClient`: one thin method per
``GET /otel/*``, each wrapped in :func:`request_user_dr_client` so credentials
come from the requesting user's headers. Transport only — camelCase query-param
assembly on the way out, snake_case key normalization on the way back
(:func:`_normalize_keys`), nothing else. No truncation, no validation: callers
(the ``otel`` tools) are responsible for both.
"""

import logging
from typing import Any
from typing import cast

from datarobot.utils import underscorize

from datarobot_genai.drmcputils.clients.datarobot import request_user_dr_client

logger = logging.getLogger(__name__)

# Keys whose *contents* are OTel data, not DataRobot API fields, and so must survive
# verbatim. An attribute map is keyed by semantic-convention names ('gen_ai.task.output',
# 'traceloop.entity.output'), which ``otel/constants.py`` matches literally and which the
# agent passes straight back as otel_span_payload_get(fields=...); underscorize() would
# rewrite any of them carrying a capital letter and break both. The key itself is already
# snake_case, so only recursion into the value is suppressed.
_OPAQUE_KEYS: frozenset[str] = frozenset(
    {
        "attributes",
        "resource",
        "scope",
        "events",
        "links",
        # The trace envelope's guard maps are keyed by user-configured guard names
        # ('PII Detection'), not API fields; underscorize() would mangle any name
        # carrying a capital or a space ('pii _detection'). Both spellings appear
        # because the raw wire key is camelCase while stub/hand-written fixtures
        # are already snake_case.
        "promptGuards",
        "responseGuards",
        "prompt_guards",
        "response_guards",
    }
)


def _normalize_keys(value: Any) -> Any:
    """Recursively rewrite the API's camelCase response keys to snake_case.

    drflask camelizes response keys on the way out ('traceId', 'spansCount',
    'genAiUsageInputTokens') while every tool in :mod:`datarobot_genai.drtools.otel` reads
    snake_case, so the translation has to happen somewhere. Doing it here — once, in the
    transport — keeps the tools, their fixtures and the integration stubs all speaking one
    dialect.

    Uses the DataRobot SDK's own :func:`underscorize`, so this agrees with ``from_api``.
    It differs from ``from_api`` in two ways that matter here: null-valued keys are kept
    (``from_api`` drops them, which would make a present-but-null payload field
    indistinguishable from an absent one), and :data:`_OPAQUE_KEYS` are not descended into.

    Idempotent on already-snake_case input, so a stubbed or hand-written fixture passes
    through unchanged.
    """
    if isinstance(value, list):
        return [_normalize_keys(item) for item in value]
    if isinstance(value, dict):
        return {
            underscorize(key): item if key in _OPAQUE_KEYS else _normalize_keys(item)
            for key, item in value.items()
        }
    return value


def _get_normalized(path: str, params: dict[str, Any]) -> dict[str, Any]:
    """``GET`` one ``/otel`` route and return its body with keys normalized."""
    with request_user_dr_client() as client:
        return cast(dict[str, Any], _normalize_keys(client.get(path, params=params).json()))


class OtelQueryApiClient:
    """OTel query API methods backed by the per-request DataRobot REST client.

    Each call runs inside :func:`request_user_dr_client` so credentials come from
    the requesting user's headers::

        client = OtelQueryApiClient()
        result = client.list_traces(entity_type="deployment", entity_id="...")
    """

    # ------------------------------------------------------------------ #
    # Traces                                                               #
    # ------------------------------------------------------------------ #

    def list_traces(
        self,
        entity_type: str,
        entity_id: str,
        *,
        limit: int = 20,
        offset: int = 0,
        start_time: str | None = None,
        end_time: str | None = None,
        status: str | None = None,
        root_span_name: list[str] | None = None,
        tools: list[str] | None = None,
        trace_type: str | None = None,
        min_trace_duration_ns: int | None = None,
        min_span_duration_ns: int | None = None,
        max_span_duration_ns: int | None = None,
        min_trace_cost: float | None = None,
        max_trace_cost: float | None = None,
        sort_by: str | None = None,
        sort_direction: str | None = None,
    ) -> dict[str, Any]:
        """GET /otel/{entityType}/{entityId}/traces/ — paginated trace list."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        if status:
            params["status"] = status
        if root_span_name:
            params["rootSpanName"] = root_span_name
        if tools:
            params["tools"] = tools
        if trace_type:
            params["traceType"] = trace_type
        if min_trace_duration_ns is not None:
            params["minTraceDuration"] = min_trace_duration_ns
        if min_span_duration_ns is not None:
            params["minSpanDuration"] = min_span_duration_ns
        if max_span_duration_ns is not None:
            params["maxSpanDuration"] = max_span_duration_ns
        if min_trace_cost is not None:
            params["minTraceCost"] = min_trace_cost
        if max_trace_cost is not None:
            params["maxTraceCost"] = max_trace_cost
        if sort_by:
            params["sortBy"] = sort_by
        if sort_direction:
            params["sortDirection"] = sort_direction
        return _get_normalized(f"otel/{entity_type}/{entity_id}/traces/", params)

    def get_trace(
        self,
        entity_type: str,
        entity_id: str,
        trace_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """GET /otel/{entityType}/{entityId}/traces/{traceId}/ — one trace, spans paginated."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return _get_normalized(f"otel/{entity_type}/{entity_id}/traces/{trace_id}/", params)

    # ------------------------------------------------------------------ #
    # Logs                                                                 #
    # ------------------------------------------------------------------ #

    def list_logs(
        self,
        entity_type: str,
        entity_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
        level: str = "debug",
        start_time: str | None = None,
        end_time: str | None = None,
        includes: list[str] | None = None,
        excludes: list[str] | None = None,
        span_id: str | None = None,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """GET /otel/{entityType}/{entityId}/logs/ — OTel log lines for an entity."""
        params: dict[str, Any] = {"limit": limit, "offset": offset, "level": level}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        if includes:
            params["includes"] = includes
        if excludes:
            params["excludes"] = excludes
        if span_id:
            params["spanId"] = span_id
        if trace_id:
            params["traceId"] = trace_id
        return _get_normalized(f"otel/{entity_type}/{entity_id}/logs/", params)

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #

    def list_metrics_summary(
        self,
        entity_type: str,
        entity_id: str,
        *,
        search: str | None = None,
        metric_type: str | None = None,
    ) -> dict[str, Any]:
        """GET /otel/{entityType}/{entityId}/metrics/summary/ — catalog of emitted metrics."""
        params: dict[str, Any] = {}
        if search:
            params["search"] = search
        if metric_type:
            params["metricType"] = metric_type
        return _get_normalized(f"otel/{entity_type}/{entity_id}/metrics/summary/", params)

    def get_metrics_values(
        self,
        entity_type: str,
        entity_id: str,
        *,
        histogram_buckets: bool = False,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        """GET /otel/{entityType}/{entityId}/metrics/values/ — configured metric values."""
        params: dict[str, Any] = {"histogramBuckets": histogram_buckets}
        if start:
            params["startTime"] = start
        if end:
            params["endTime"] = end
        return _get_normalized(f"otel/{entity_type}/{entity_id}/metrics/values/", params)

    def get_autocollected_metrics_values(
        self,
        entity_type: str,
        entity_id: str,
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        """GET /otel/{entityType}/{entityId}/metrics/autocollectedValues/ — platform metrics."""
        params: dict[str, Any] = {}
        if start:
            params["startTime"] = start
        if end:
            params["endTime"] = end
        return _get_normalized(
            f"otel/{entity_type}/{entity_id}/metrics/autocollectedValues/", params
        )

    # ------------------------------------------------------------------ #
    # Stats                                                                #
    # ------------------------------------------------------------------ #

    def get_entity_stats(
        self,
        service_name: str,
        *,
        start: str | None = None,
        end: str | None = None,
        limit: int = 1000,
    ) -> dict[str, Any]:
        """GET /otel/stats/ — per-(user, service) OTel volume for one entity.

        ``service_name`` is ``f"{entity_type}-{entity_id}"``, built by the
        caller — this client never constructs it.
        """
        params: dict[str, Any] = {"serviceName": service_name, "limit": limit}
        if start:
            params["startTime"] = start
        if end:
            params["endTime"] = end
        return _get_normalized("otel/stats/", params)
