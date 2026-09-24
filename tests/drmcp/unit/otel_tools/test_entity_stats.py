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

"""Unit tests for ``otel_entity_stats_get``.

Style: GIVEN preconditions / WHEN behavior under test / THEN expected outcomes.

Plan §2.7's three outcomes, each pinned to a distinct result:

1. Entity has data            -> 200, rows        -> counts, has_otel_data=True
2. Entity readable, no data   -> 200, data: []     -> zeros, has_otel_data=False
3. Entity not readable (403)  -> ToolError naming the entity as a permissions
                                  failure, never an empty result

Plus the traps: service_name is always built internally as
f"{entity_type}-{entity_id}" and always sent; rows are per (user_id,
service_name) so top-level counts are SUMMED across rows, not read off the
first one; by_user is capped at 20 but user_count stays exact; limit=1000 is
always requested and a note is added if total_count exceeds it.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from datarobot.errors import ClientError

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.otel import entity_stats

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


def _row(user_id: str, *, span_count: int = 0, metric_count: int = 0, log_count: int = 0) -> dict:
    return {
        "user_id": user_id,
        "service_name": "deployment-x",
        "span_count": span_count,
        "metric_count": metric_count,
        "log_count": log_count,
    }


# ------------------------------------------------------------------ #
# outcome 1 — entity has data                                          #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_otel_entity_stats_get_sums_counts_across_per_user_rows(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN three rows for the same entity, one per user who produced telemetry
    _stub_json(
        patched_dr_client,
        {
            "data": [
                _row("user-1", span_count=4100, metric_count=0, log_count=88000),
                _row("user-2", span_count=100, metric_count=0, log_count=291),
                _row("user-3", span_count=13, metric_count=0, log_count=0),
            ]
        },
    )

    # WHEN the entity is queried
    result = await entity_stats.otel_entity_stats_get(
        entity_type="deployment", entity_id=_ENTITY_ID
    )

    # THEN service_name is built internally and always sent, and the top-level
    # counts are the SUM across rows, not the first row alone
    patched_dr_client.get.assert_called_once_with(
        "otel/stats/",
        params={"serviceName": f"deployment-{_ENTITY_ID}", "limit": 1000},
    )
    assert result["entity_type"] == "deployment"
    assert result["entity_id"] == _ENTITY_ID
    assert result["service_name"] == f"deployment-{_ENTITY_ID}"
    assert result["has_otel_data"] is True
    assert result["span_count"] == 4213
    assert result["metric_count"] == 0
    assert result["log_count"] == 88291
    assert result["user_count"] == 3


@pytest.mark.asyncio
async def test_otel_entity_stats_get_by_user_capped_at_20_but_user_count_exact(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN 25 distinct users' rows for one entity
    rows = [_row(f"user-{i}", span_count=1) for i in range(25)]
    _stub_json(patched_dr_client, {"data": rows})

    # WHEN the entity is queried
    result = await entity_stats.otel_entity_stats_get(
        entity_type="deployment", entity_id=_ENTITY_ID
    )

    # THEN by_user is capped at 20 rows for display, but user_count and the
    # summed span_count both reflect the full set of 25 rows, not just the 20 shown
    assert len(result["by_user"]) == 20
    assert result["user_count"] == 25
    assert result["span_count"] == 25


@pytest.mark.asyncio
async def test_otel_entity_stats_get_reports_when_total_count_exceeds_limit(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN a response whose true total exceeds the 1000-row request limit
    _stub_json(
        patched_dr_client,
        {"data": [_row("user-1", span_count=10)], "total_count": 1500},
    )

    # WHEN the entity is queried
    result = await entity_stats.otel_entity_stats_get(
        entity_type="deployment", entity_id=_ENTITY_ID
    )

    # THEN a note flags that the sums may be partial, rather than silently
    # presenting a truncated sum as complete
    assert "note" in result
    assert "1500" in result["note"]
    assert "1000" in result["note"]


@pytest.mark.asyncio
async def test_otel_entity_stats_get_no_note_when_total_count_within_limit(
    patched_dr_client: MagicMock,
) -> None:
    _stub_json(
        patched_dr_client,
        {"data": [_row("user-1", span_count=10)], "total_count": 1},
    )

    result = await entity_stats.otel_entity_stats_get(
        entity_type="deployment", entity_id=_ENTITY_ID
    )

    assert "note" not in result


@pytest.mark.asyncio
async def test_otel_entity_stats_get_non_numeric_total_count_does_not_crash(
    patched_dr_client: MagicMock,
) -> None:
    """A malformed upstream total_count must not surface as an unhandled ValueError.

    Every other error path in this module (403/404/500) is a controlled
    ToolError; a garbage value in an otherwise-200 response should degrade to
    "unknown, no note" rather than crashing the call.
    """
    # GIVEN an upstream response whose total_count is not a number
    _stub_json(
        patched_dr_client,
        {"data": [_row("user-1", span_count=10)], "total_count": "not-a-number"},
    )

    # WHEN the entity is queried
    result = await entity_stats.otel_entity_stats_get(
        entity_type="deployment", entity_id=_ENTITY_ID
    )

    # THEN the call succeeds, treating the malformed field as unknown rather
    # than raising, and adds no (necessarily wrong) note about it
    assert result["span_count"] == 10
    assert "note" not in result


# ------------------------------------------------------------------ #
# outcome 2 — entity readable, no telemetry                            #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_otel_entity_stats_get_no_telemetry_is_zeros_not_missing(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN the entity is readable but has produced no telemetry at all
    _stub_json(patched_dr_client, {"data": []})

    # WHEN the entity is queried
    result = await entity_stats.otel_entity_stats_get(
        entity_type="deployment", entity_id=_ENTITY_ID
    )

    # THEN it's a normal 200 with explicit zeros and has_otel_data=False —
    # never an exception
    assert result["has_otel_data"] is False
    assert result["span_count"] == 0
    assert result["metric_count"] == 0
    assert result["log_count"] == 0
    assert result["user_count"] == 0
    assert result["by_user"] == []


# ------------------------------------------------------------------ #
# outcome 3 — entity not readable (403)                                #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_otel_entity_stats_get_403_raises_permission_error_naming_entity(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN the caller cannot read this entity's telemetry (check_read_permission
    # -> PermissionDenied -> 403)
    patched_dr_client.get.side_effect = ClientError("403", status_code=403, json={})

    # WHEN the entity is queried
    with pytest.raises(ToolError) as exc_info:
        await entity_stats.otel_entity_stats_get(entity_type="deployment", entity_id=_ENTITY_ID)

    # THEN it raises rather than returning an empty result, the message names
    # the entity, and says this is a permissions issue
    message = str(exc_info.value).lower()
    assert "permission" in message
    assert _ENTITY_ID in str(exc_info.value)
    assert "deployment" in message


@pytest.mark.asyncio
async def test_otel_entity_stats_get_404_maps_to_not_found(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("404", status_code=404, json={})
    with pytest.raises(ToolError) as exc_info:
        await entity_stats.otel_entity_stats_get(entity_type="deployment", entity_id=_ENTITY_ID)
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


@pytest.mark.asyncio
async def test_otel_entity_stats_get_500_maps_to_upstream(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("500", status_code=500, json={})
    with pytest.raises(ToolError) as exc_info:
        await entity_stats.otel_entity_stats_get(entity_type="deployment", entity_id=_ENTITY_ID)
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


# ------------------------------------------------------------------ #
# validation / plumbing                                                #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_otel_entity_stats_get_strips_and_validates_entity_id(
    patched_dr_client: MagicMock,
) -> None:
    _stub_json(patched_dr_client, {"data": []})

    await entity_stats.otel_entity_stats_get(entity_type="workload", entity_id=f"  {_ENTITY_ID}  ")

    patched_dr_client.get.assert_called_once_with(
        "otel/stats/",
        params={"serviceName": f"workload-{_ENTITY_ID}", "limit": 1000},
    )


@pytest.mark.asyncio
async def test_otel_entity_stats_get_malformed_entity_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await entity_stats.otel_entity_stats_get(entity_type="deployment", entity_id="not-hex")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_otel_entity_stats_get_passes_time_window(patched_dr_client: MagicMock) -> None:
    _stub_json(patched_dr_client, {"data": []})

    await entity_stats.otel_entity_stats_get(
        entity_type="deployment",
        entity_id=_ENTITY_ID,
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-02T00:00:00Z",
    )

    patched_dr_client.get.assert_called_once_with(
        "otel/stats/",
        params={
            "serviceName": f"deployment-{_ENTITY_ID}",
            "limit": 1000,
            "startTime": "2026-01-01T00:00:00Z",
            "endTime": "2026-01-02T00:00:00Z",
        },
    )


@pytest.mark.asyncio
async def test_otel_entity_stats_get_by_user_row_shape(patched_dr_client: MagicMock) -> None:
    _stub_json(
        patched_dr_client,
        {"data": [_row("user-1", span_count=5, metric_count=1, log_count=9)]},
    )

    result = await entity_stats.otel_entity_stats_get(
        entity_type="deployment", entity_id=_ENTITY_ID
    )

    assert result["by_user"] == [
        {"user_id": "user-1", "span_count": 5, "metric_count": 1, "log_count": 9}
    ]


@pytest.mark.asyncio
async def test_otel_entity_stats_get_by_user_row_agrees_with_the_summed_total_on_null_counts(
    patched_dr_client: MagicMock,
) -> None:
    """A row with an explicit ``None`` count must not disagree with its own sum.

    ``_sum_field`` normalizes ``None`` to 0 when summing; ``by_user`` must
    report the same row the same way, or the two halves of one response tell
    an agent two different stories about the same data.
    """
    # GIVEN a row where span_count is present but explicitly null
    _stub_json(
        patched_dr_client,
        {"data": [{"user_id": "user-1", "span_count": None, "metric_count": 1, "log_count": 2}]},
    )

    # WHEN the entity is queried
    result = await entity_stats.otel_entity_stats_get(
        entity_type="deployment", entity_id=_ENTITY_ID
    )

    # THEN the top-level sum and the by_user breakdown agree: both 0, not one
    # 0 and the other None
    assert result["span_count"] == 0
    assert result["by_user"][0]["span_count"] == 0
