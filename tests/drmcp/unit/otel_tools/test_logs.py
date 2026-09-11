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

"""Unit tests for ``otel_logs_list``.

Style: GIVEN preconditions / WHEN behavior under test / THEN expected outcomes.

Two traps pinned explicitly, per plan §2.4/§7:

* ``level`` accepts all six of ``debug|info|warn|warning|error|critical`` — in
  particular ``warning`` and ``critical``, which ``drmcputils.constants.LOG_LEVELS``
  does not carry.
* ``max_line_chars`` truncates ``message`` and ``stacktrace`` independently, in
  characters, each with its own ``…[truncated, N more chars]`` marker.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from datarobot.errors import ClientError

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.core import get_registered_tools
from datarobot_genai.drtools.otel import logs
from datarobot_genai.drtools.otel.constants import OTEL_LOG_LEVELS

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


# ------------------------------------------------------------------ #
# success / pagination / wire params                                   #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_otel_logs_list_maps_data_and_merges_pagination(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN a raw logs/ response with no total_count (PaginatedWithoutTotalResponseValidator)
    _stub_json(
        patched_dr_client,
        {
            "data": [
                {"timestamp": 1.0, "level": "info", "message": "started"},
                {"timestamp": 2.0, "level": "error", "message": "boom"},
            ],
            "count": 2,
            "next": "https://example/logs/?offset=50",
            "previous": None,
        },
    )

    # WHEN otel_logs_list is called with only the required entity args
    result = await logs.otel_logs_list(entity_type="deployment", entity_id=_ENTITY_ID)

    # THEN 'data' becomes 'logs', pagination metadata is merged in, and there is
    # no total_count anywhere in the response
    patched_dr_client.get.assert_called_once_with(
        f"otel/deployment/{_ENTITY_ID}/logs/",
        params={"limit": 50, "offset": 0, "level": "debug"},
    )
    assert [line["message"] for line in result["logs"]] == ["started", "boom"]
    assert result["count"] == 2
    assert result["offset"] == 0
    assert result["limit"] == 50
    assert result["next"] == "https://example/logs/?offset=50"
    assert "previous" not in result
    assert "total_count" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("upstream_key", ["total_count", "totalCount", "total"])
async def test_otel_logs_list_never_leaks_a_total_count_even_if_upstream_sends_one(
    upstream_key: str, patched_dr_client: MagicMock
) -> None:
    """§2.4's 'no total_count' invariant must hold regardless of upstream shape.

    merge_pagination_metadata generically copies total_count/totalCount/total
    from any api_response that carries one; this endpoint's real response
    never does today, but the invariant should not rest on that being true
    forever, so otel_logs_list must strip it explicitly.
    """
    # GIVEN an (unrealistic, but not impossible) upstream response that DOES
    # carry a total-shaped key
    _stub_json(patched_dr_client, {"data": [], "count": 0, upstream_key: 999_999})

    # WHEN otel_logs_list is called
    result = await logs.otel_logs_list(entity_type="deployment", entity_id=_ENTITY_ID)

    # THEN total_count never appears in the response
    assert "total_count" not in result


@pytest.mark.asyncio
async def test_otel_logs_list_strips_entity_id(patched_dr_client: MagicMock) -> None:
    _stub_json(patched_dr_client, {"data": [], "count": 0})

    await logs.otel_logs_list(entity_type="deployment", entity_id=f"  {_ENTITY_ID}  ")

    patched_dr_client.get.assert_called_once_with(
        f"otel/deployment/{_ENTITY_ID}/logs/",
        params={"limit": 50, "offset": 0, "level": "debug"},
    )


@pytest.mark.asyncio
async def test_otel_logs_list_with_filters(patched_dr_client: MagicMock) -> None:
    _stub_json(patched_dr_client, {"data": [], "count": 0})

    await logs.otel_logs_list(
        entity_type="deployment",
        entity_id=_ENTITY_ID,
        level="error",
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-02T00:00:00Z",
        includes=["ERROR", "FATAL"],
        excludes=["healthcheck"],
        span_id="span-1",
        trace_id="trace-1",
    )

    patched_dr_client.get.assert_called_once_with(
        f"otel/deployment/{_ENTITY_ID}/logs/",
        params={
            "limit": 50,
            "offset": 0,
            "level": "error",
            "startTime": "2026-01-01T00:00:00Z",
            "endTime": "2026-01-02T00:00:00Z",
            "includes": ["ERROR", "FATAL"],
            "excludes": ["healthcheck"],
            "spanId": "span-1",
            "traceId": "trace-1",
        },
    )


@pytest.mark.asyncio
async def test_otel_logs_list_clamps_limit(patched_dr_client: MagicMock) -> None:
    _stub_json(patched_dr_client, {"data": [], "count": 0})

    result = await logs.otel_logs_list(entity_type="deployment", entity_id=_ENTITY_ID, limit=500)

    patched_dr_client.get.assert_called_once_with(
        f"otel/deployment/{_ENTITY_ID}/logs/",
        params={"limit": 100, "offset": 0, "level": "debug"},
    )
    assert result["limit"] == 100
    assert "note" in result


@pytest.mark.asyncio
async def test_otel_logs_list_negative_offset_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await logs.otel_logs_list(entity_type="deployment", entity_id=_ENTITY_ID, offset=-1)
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_otel_logs_list_client_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("500", status_code=500, json={})
    with pytest.raises(ToolError) as exc_info:
        await logs.otel_logs_list(entity_type="deployment", entity_id=_ENTITY_ID)
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


@pytest.mark.asyncio
async def test_otel_logs_list_not_found(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("404", status_code=404, json={})
    with pytest.raises(ToolError) as exc_info:
        await logs.otel_logs_list(entity_type="deployment", entity_id=_ENTITY_ID)
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


# ------------------------------------------------------------------ #
# level — the OTEL_LOG_LEVELS trap (plan §2.4, pinned by §7)           #
# ------------------------------------------------------------------ #


def test_otel_log_levels_constant_carries_all_six_levels() -> None:
    # GIVEN/THEN: the module-level constant this tool validates against must be
    # the full six-level set, not drmcputils.constants.LOG_LEVELS's four.
    assert set(OTEL_LOG_LEVELS) == {"debug", "info", "warn", "warning", "error", "critical"}


@pytest.mark.asyncio
@pytest.mark.parametrize("level", list(OTEL_LOG_LEVELS))
async def test_otel_logs_list_accepts_every_otel_log_level(
    level: str, patched_dr_client: MagicMock
) -> None:
    # GIVEN a stubbed response
    _stub_json(patched_dr_client, {"data": [], "count": 0})

    # WHEN called with each of the six accepted levels, including 'warning' and
    # 'critical' which drmcputils.constants.LOG_LEVELS would reject
    await logs.otel_logs_list(entity_type="deployment", entity_id=_ENTITY_ID, level=level)  # type: ignore[arg-type]

    # THEN the level is forwarded to the wire unchanged, never rejected
    patched_dr_client.get.assert_called_once_with(
        f"otel/deployment/{_ENTITY_ID}/logs/",
        params={"limit": 50, "offset": 0, "level": level},
    )


@pytest.mark.asyncio
async def test_otel_logs_list_warning_level_accepted(patched_dr_client: MagicMock) -> None:
    """Regression pinned explicitly by plan §7: 'warning' must not be rejected."""
    _stub_json(patched_dr_client, {"data": [], "count": 0})

    await logs.otel_logs_list(entity_type="deployment", entity_id=_ENTITY_ID, level="warning")

    patched_dr_client.get.assert_called_once_with(
        f"otel/deployment/{_ENTITY_ID}/logs/",
        params={"limit": 50, "offset": 0, "level": "warning"},
    )


@pytest.mark.asyncio
async def test_otel_logs_list_critical_level_accepted(patched_dr_client: MagicMock) -> None:
    """Regression pinned explicitly by plan §7: 'critical' must not be rejected."""
    _stub_json(patched_dr_client, {"data": [], "count": 0})

    await logs.otel_logs_list(entity_type="deployment", entity_id=_ENTITY_ID, level="critical")

    patched_dr_client.get.assert_called_once_with(
        f"otel/deployment/{_ENTITY_ID}/logs/",
        params={"limit": 50, "offset": 0, "level": "critical"},
    )


@pytest.mark.asyncio
async def test_otel_logs_list_invalid_level_raises(patched_dr_client: MagicMock) -> None:
    # GIVEN a level outside the OTel API's own accepted set (bypassing the MCP
    # schema layer, as a direct unit-test call does)
    with pytest.raises(ToolError) as exc_info:
        await logs.otel_logs_list(entity_type="deployment", entity_id=_ENTITY_ID, level="verbose")  # type: ignore[arg-type]
    assert exc_info.value.kind is ToolErrorKind.VALIDATION
    patched_dr_client.get.assert_not_called()


# ------------------------------------------------------------------ #
# max_line_chars — independent message/stacktrace windowing             #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_otel_logs_list_truncates_long_message_with_marker(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN a log line whose message is longer than max_line_chars
    long_message = "x" * 5_000
    _stub_json(patched_dr_client, {"data": [{"message": long_message}], "count": 1})

    # WHEN read with a small max_line_chars
    result = await logs.otel_logs_list(
        entity_type="deployment", entity_id=_ENTITY_ID, max_line_chars=100
    )

    # THEN the message is windowed to 100 chars plus an explicit marker naming
    # exactly how many characters were dropped
    message = result["logs"][0]["message"]
    assert message.startswith("x" * 100)
    assert message == "x" * 100 + "…[truncated, 4900 more chars]"


@pytest.mark.asyncio
async def test_otel_logs_list_truncates_message_and_stacktrace_independently(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN one line with a short message and a long stacktrace
    short_message = "connection reset"
    long_stacktrace = "y" * 3_000
    _stub_json(
        patched_dr_client,
        {"data": [{"message": short_message, "stacktrace": long_stacktrace}], "count": 1},
    )

    # WHEN read with a max_line_chars smaller than the stacktrace but larger
    # than the message
    result = await logs.otel_logs_list(
        entity_type="deployment", entity_id=_ENTITY_ID, max_line_chars=200
    )

    # THEN the short message survives untouched while the stacktrace is
    # windowed and marked on its own — one field's cap does not eat the other's
    line = result["logs"][0]
    assert line["message"] == short_message
    assert line["stacktrace"] == "y" * 200 + "…[truncated, 2800 more chars]"


@pytest.mark.asyncio
async def test_otel_logs_list_max_line_chars_zero_disables_truncation(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN a very long message
    long_message = "z" * 10_000
    _stub_json(patched_dr_client, {"data": [{"message": long_message}], "count": 1})

    # WHEN max_line_chars=0
    result = await logs.otel_logs_list(
        entity_type="deployment", entity_id=_ENTITY_ID, max_line_chars=0
    )

    # THEN the message is returned in full, untruncated
    assert result["logs"][0]["message"] == long_message


@pytest.mark.asyncio
async def test_otel_logs_list_passes_through_non_capped_fields_unchanged(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN a line carrying span_id/trace_id/level/timestamp alongside message
    _stub_json(
        patched_dr_client,
        {
            "data": [
                {
                    "timestamp": 123.456,
                    "level": "error",
                    "message": "boom",
                    "span_id": "span-abc",
                    "trace_id": "trace-abc",
                }
            ],
            "count": 1,
        },
    )

    # WHEN read with default truncation
    result = await logs.otel_logs_list(entity_type="deployment", entity_id=_ENTITY_ID)

    # THEN every field other than message/stacktrace is passed through as-is
    line = result["logs"][0]
    assert line["timestamp"] == 123.456
    assert line["level"] == "error"
    assert line["span_id"] == "span-abc"
    assert line["trace_id"] == "trace-abc"


# ------------------------------------------------------------------ #
# description text — the GENAI_EXPERIMENTATION 403 hint (plan §2)      #
# ------------------------------------------------------------------ #


def test_otel_logs_list_description_documents_the_genai_experimentation_403_cause() -> None:
    # GIVEN the tool's registered metadata
    by_name = {(md.get("name") or fn.__name__): md for fn, md in get_registered_tools()}
    description = by_name["otel_logs_list"]["description"]

    # THEN a 403 is documented as a feature-flag/configuration cause, not
    # left indistinguishable from "no access to this entity's logs"
    assert "GENAI_EXPERIMENTATION" in description
    assert "403" in description
