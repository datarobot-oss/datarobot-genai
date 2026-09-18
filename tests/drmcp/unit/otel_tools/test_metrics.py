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

"""Unit tests for ``otel_metrics_catalog_list`` and ``otel_metrics_values_get``.

Style: GIVEN preconditions / WHEN behavior under test / THEN expected outcomes.

Plan §2.6's trap pinned explicitly: ``source='autocollected'`` is the default,
and an empty ``metric_aggregations`` for ``source='configured'`` must be
returned as a normal, unremarkable empty list — never rewritten into an error
or a synthetic "no data" flag. That distinction lives in the tool's
description text (asserted below), not in extra response-shape logic.
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
from datarobot_genai.drtools.otel import metrics

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
# otel_metrics_catalog_list                                            #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_otel_metrics_catalog_list_maps_data_to_metrics(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN a raw metrics/summary/ response
    _stub_json(
        patched_dr_client,
        {
            "data": [
                {"otel_name": "gen_ai.usage.input_tokens", "metric_type": "counter"},
                {"otel_name": "gen_ai.usage.output_tokens", "metric_type": "counter"},
            ]
        },
    )

    # WHEN otel_metrics_catalog_list is called with only the required entity args
    result = await metrics.otel_metrics_catalog_list(entity_type="deployment", entity_id=_ENTITY_ID)

    # THEN 'data' becomes 'metrics' and 'count' reflects the returned length,
    # with no params sent since search/metric_type are both unset
    patched_dr_client.get.assert_called_once_with(
        f"otel/deployment/{_ENTITY_ID}/metrics/summary/", params={}
    )
    assert result["count"] == 2
    assert result["metrics"][0]["otel_name"] == "gen_ai.usage.input_tokens"
    assert "note" not in result


@pytest.mark.asyncio
async def test_otel_metrics_catalog_list_with_search_and_type(
    patched_dr_client: MagicMock,
) -> None:
    _stub_json(patched_dr_client, {"data": []})

    await metrics.otel_metrics_catalog_list(
        entity_type="deployment",
        entity_id=_ENTITY_ID,
        search="latency",
        metric_type="histogram",
    )

    patched_dr_client.get.assert_called_once_with(
        f"otel/deployment/{_ENTITY_ID}/metrics/summary/",
        params={"search": "latency", "metricType": "histogram"},
    )


@pytest.mark.asyncio
async def test_otel_metrics_catalog_list_clamps_to_100_with_note(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN a server response at its own max_length=1000 ceiling
    _stub_json(
        patched_dr_client,
        {"data": [{"otel_name": f"metric.{i}"} for i in range(150)]},
    )

    # WHEN read without narrowing via 'search'
    result = await metrics.otel_metrics_catalog_list(entity_type="deployment", entity_id=_ENTITY_ID)

    # THEN the client-side cap of 100 applies and is reported, not silent
    assert result["count"] == 100
    assert len(result["metrics"]) == 100
    assert "note" in result
    assert "100" in result["note"]


@pytest.mark.asyncio
async def test_otel_metrics_catalog_list_empty_is_not_flagged(
    patched_dr_client: MagicMock,
) -> None:
    _stub_json(patched_dr_client, {"data": []})

    result = await metrics.otel_metrics_catalog_list(entity_type="deployment", entity_id=_ENTITY_ID)

    assert result == {"metrics": [], "count": 0}


@pytest.mark.asyncio
async def test_otel_metrics_catalog_list_client_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("500", status_code=500, json={})
    with pytest.raises(ToolError) as exc_info:
        await metrics.otel_metrics_catalog_list(entity_type="deployment", entity_id=_ENTITY_ID)
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


@pytest.mark.asyncio
async def test_otel_metrics_catalog_list_not_found(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("404", status_code=404, json={})
    with pytest.raises(ToolError) as exc_info:
        await metrics.otel_metrics_catalog_list(entity_type="deployment", entity_id=_ENTITY_ID)
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


# ------------------------------------------------------------------ #
# otel_metrics_values_get — source='autocollected' (the default)      #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_otel_metrics_values_get_default_source_is_autocollected(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN a raw autocollectedValues/ response
    _stub_json(
        patched_dr_client,
        {"data": [{"otel_name": "cpu.usage", "aggregated_value": 0.42}]},
    )

    # WHEN otel_metrics_values_get is called with no 'source' argument at all
    result = await metrics.otel_metrics_values_get(entity_type="deployment", entity_id=_ENTITY_ID)

    # THEN the autocollectedValues endpoint was hit (never metrics/values/)
    patched_dr_client.get.assert_called_once_with(
        f"otel/deployment/{_ENTITY_ID}/metrics/autocollectedValues/",
        params={},
    )
    assert result == {"metrics": [{"otel_name": "cpu.usage", "aggregated_value": 0.42}], "count": 1}


@pytest.mark.asyncio
async def test_otel_metrics_values_get_autocollected_explicit_with_window(
    patched_dr_client: MagicMock,
) -> None:
    _stub_json(patched_dr_client, {"data": []})

    await metrics.otel_metrics_values_get(
        entity_type="deployment",
        entity_id=_ENTITY_ID,
        source="autocollected",
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-02T00:00:00Z",
    )

    patched_dr_client.get.assert_called_once_with(
        f"otel/deployment/{_ENTITY_ID}/metrics/autocollectedValues/",
        params={"startTime": "2026-01-01T00:00:00Z", "endTime": "2026-01-02T00:00:00Z"},
    )


# ------------------------------------------------------------------ #
# otel_metrics_values_get — source='configured'                        #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_otel_metrics_values_get_configured_maps_the_three_fields(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN a raw metrics/values/ response
    _stub_json(
        patched_dr_client,
        {
            "start_time": "2026-01-01T00:00:00Z",
            "end_time": "2026-01-02T00:00:00Z",
            "metric_aggregations": [{"otel_name": "custom.metric", "aggregated_value": 1.0}],
        },
    )

    # WHEN read with source='configured'
    result = await metrics.otel_metrics_values_get(
        entity_type="deployment", entity_id=_ENTITY_ID, source="configured"
    )

    # THEN histogramBuckets defaults to False and the exact three documented
    # fields are returned
    patched_dr_client.get.assert_called_once_with(
        f"otel/deployment/{_ENTITY_ID}/metrics/values/",
        params={"histogramBuckets": False},
    )
    assert result == {
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-02T00:00:00Z",
        "metric_aggregations": [{"otel_name": "custom.metric", "aggregated_value": 1.0}],
    }


@pytest.mark.asyncio
async def test_otel_metrics_values_get_configured_empty_is_not_flagged_as_error(
    patched_dr_client: MagicMock,
) -> None:
    """The common case: no custom metrics configured. Must NOT raise or warn."""
    # GIVEN the server's own real-world common case: no custom metrics at all
    _stub_json(
        patched_dr_client,
        {"start_time": None, "end_time": None, "metric_aggregations": []},
    )

    # WHEN read with source='configured'
    result = await metrics.otel_metrics_values_get(
        entity_type="deployment", entity_id=_ENTITY_ID, source="configured"
    )

    # THEN the empty list is returned plainly — no exception, no synthetic
    # 'no data' marker bolted onto the response
    assert result["metric_aggregations"] == []


@pytest.mark.asyncio
async def test_otel_metrics_values_get_configured_missing_key_defaults_to_empty_list(
    patched_dr_client: MagicMock,
) -> None:
    # GIVEN a response that omits metric_aggregations entirely
    _stub_json(patched_dr_client, {"start_time": None, "end_time": None})

    result = await metrics.otel_metrics_values_get(
        entity_type="deployment", entity_id=_ENTITY_ID, source="configured"
    )

    assert result["metric_aggregations"] == []


@pytest.mark.asyncio
async def test_otel_metrics_values_get_configured_with_histogram_buckets(
    patched_dr_client: MagicMock,
) -> None:
    _stub_json(
        patched_dr_client,
        {"start_time": None, "end_time": None, "metric_aggregations": []},
    )

    await metrics.otel_metrics_values_get(
        entity_type="deployment",
        entity_id=_ENTITY_ID,
        source="configured",
        histogram_buckets=True,
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-02T00:00:00Z",
    )

    patched_dr_client.get.assert_called_once_with(
        f"otel/deployment/{_ENTITY_ID}/metrics/values/",
        params={
            "histogramBuckets": True,
            "startTime": "2026-01-01T00:00:00Z",
            "endTime": "2026-01-02T00:00:00Z",
        },
    )


@pytest.mark.asyncio
async def test_otel_metrics_values_get_client_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("500", status_code=500, json={})
    with pytest.raises(ToolError) as exc_info:
        await metrics.otel_metrics_values_get(entity_type="deployment", entity_id=_ENTITY_ID)
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


# ------------------------------------------------------------------ #
# description text — the "empty means unconfigured, not no data" note  #
# ------------------------------------------------------------------ #


def test_otel_metrics_values_get_description_explains_empty_configured_result() -> None:
    # GIVEN the tool's registered metadata
    by_name = {(md.get("name") or fn.__name__): md for fn, md in get_registered_tools()}
    description = by_name["otel_metrics_values_get"]["description"].lower()

    # THEN the description spells out that an empty metric_aggregations means
    # "not configured", not "no data" — so an agent does not read it as failure
    assert "not 'no data'" in description
    assert "no custom metrics configured" in description


def test_otel_metrics_values_get_description_documents_the_genai_experimentation_403_cause() -> (
    None
):
    by_name = {(md.get("name") or fn.__name__): md for fn, md in get_registered_tools()}
    description = by_name["otel_metrics_values_get"]["description"]

    # THEN a 403 is documented as a feature-flag/configuration cause, not
    # left indistinguishable from "no access to this entity's metrics"
    assert "GENAI_EXPERIMENTATION" in description
    assert "403" in description


def test_otel_metrics_catalog_list_description_documents_the_genai_experimentation_403_cause() -> (
    None
):
    by_name = {(md.get("name") or fn.__name__): md for fn, md in get_registered_tools()}
    description = by_name["otel_metrics_catalog_list"]["description"]

    assert "GENAI_EXPERIMENTATION" in description
    assert "403" in description
