# Copyright 2025 DataRobot, Inc.
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

import base64
from unittest.mock import MagicMock
from unittest.mock import patch

import datarobot as dr
import polars as pl
import pytest
from datarobot.errors import ClientError
from datarobot.models.data_store import DataStoreParameters

from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind
from datarobot_genai.drtools.predictive import data


def test_merge_pagination_metadata_adds_offset_limit_next_total() -> None:
    """merge_pagination_metadata echoes offset/limit and normalizes list API pagination fields."""
    base: dict = {"datastores": []}
    body = {
        "next": "https://example/api?offset=10",
        "previous": None,
        "total": 25,
    }
    out = data.merge_pagination_metadata(base, body, offset=0, limit=10)
    assert out is base
    assert out["offset"] == 0
    assert out["limit"] == 10
    assert out["next"] == "https://example/api?offset=10"
    assert "previous" not in out
    assert out["total_count"] == 25


def test_merge_pagination_metadata_total_count_wins_over_total() -> None:
    """If both total_count and total are present, total_count is used (first in iteration)."""
    base: dict = {"x": 1}
    body = {"total_count": 7, "total": 99}
    out = data.merge_pagination_metadata(base, body)
    assert out["total_count"] == 7


def test_merge_pagination_metadata_list_body_ignored() -> None:
    """Non-dict body does not add next/previous/total (browse/query edge cases)."""
    base: dict = {"k": 1}
    empty_list: list = []
    out = data.merge_pagination_metadata(base, empty_list, offset=1, limit=2)
    assert out["offset"] == 1
    assert out["limit"] == 2
    assert "next" not in out


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_upload_dataset_success_from_bytes() -> None:
    raw = b"a,b\n1,2\n"
    b64 = base64.b64encode(raw).decode("ascii")
    mock_catalog_item = MagicMock()
    mock_catalog_item.id = "12345"
    mock_catalog_item.status = "completed"
    mock_catalog_item.version_id = None
    mock_catalog_item.name = "somefile.csv"
    with patch.object(
        dr.Dataset, "create_from_file", return_value=mock_catalog_item
    ) as mock_create:
        result = await data.catalog_upload_dataset(
            file_content_base64=b64, dataset_filename="somefile.csv"
        )
        mock_create.assert_called_once()
        kwargs = mock_create.call_args.kwargs
        assert kwargs["filelike"].name == "somefile.csv"
        assert kwargs["filelike"].getvalue() == raw
        assert isinstance(result, dict)
        assert result == {
            "dataset_id": "12345",
            "dataset_version_id": None,
            "dataset_name": "somefile.csv",
        }


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_upload_dataset_success_with_url() -> None:
    mock_catalog_item = MagicMock()
    mock_catalog_item.id = "12345"
    mock_catalog_item.version_id = None
    mock_catalog_item.name = "somefile.csv"
    with patch.object(dr.Dataset, "create_from_url", return_value=mock_catalog_item) as mock_create:
        result = await data.catalog_upload_dataset(file_url="https://example.com/somefile.csv")
        mock_create.assert_called_once_with("https://example.com/somefile.csv")
        assert isinstance(result, dict)
        assert result == {
            "dataset_id": "12345",
            "dataset_version_id": None,
            "dataset_name": "somefile.csv",
        }


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_upload_dataset_error_with_url() -> None:
    mock_catalog_item = MagicMock()
    mock_catalog_item.id = "12345"
    mock_catalog_item.version_id = None
    mock_catalog_item.name = "somefile.csv"
    with patch.object(dr.Dataset, "create_from_url", return_value=mock_catalog_item):
        with pytest.raises(
            ToolError,
            match="Invalid file URL: https:notavalidurl/somefile.csv",
        ):
            await data.catalog_upload_dataset(file_url="https:notavalidurl/somefile.csv")


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_upload_dataset_error_no_content_or_url() -> None:
    with pytest.raises(
        ToolError,
        match="Either file_content_base64 or file_url must be provided.",
    ):
        await data.catalog_upload_dataset()


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_upload_dataset_error_both_base64_and_url() -> None:
    with pytest.raises(
        ToolError,
        match="Please provide either file_content_base64 or file_url, not both.",
    ):
        await data.catalog_upload_dataset(
            file_content_base64=base64.b64encode(b"x").decode("ascii"),
            file_url="https://example.com/somefile.csv",
        )


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_upload_dataset_invalid_base64() -> None:
    with pytest.raises(ToolError, match="Invalid base64"):
        await data.catalog_upload_dataset(file_content_base64="not-valid-base64!!!")


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_upload_dataset_base64_whitespace_only() -> None:
    with pytest.raises(ToolError, match="file_content_base64"):
        await data.catalog_upload_dataset(file_content_base64="   ")


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_upload_dataset_error() -> None:
    with patch.object(dr.Dataset, "create_from_file", side_effect=Exception("fail")):
        with pytest.raises(Exception) as exc_info:
            await data.catalog_upload_dataset(
                file_content_base64=base64.b64encode(b"x").decode("ascii")
            )
        assert "fail" in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_list_datasets_success() -> None:
    mock_ds1 = MagicMock()
    mock_ds1.id = "1"
    mock_ds1.name = "ds1"
    mock_ds2 = MagicMock()
    mock_ds2.id = "2"
    mock_ds2.name = "ds2"
    with patch.object(dr.Dataset, "iterate", return_value=iter([mock_ds1, mock_ds2])) as mock_iter:
        result = await data.catalog_list_datasets()
        assert result["count"] == 2
        # datasets is now a dict mapping id to name
        assert result["datasets"]["1"] == "ds1"
        assert result["datasets"]["2"] == "ds2"
        assert result["limit"] == 100
        assert result["may_have_more"] is False
        mock_iter.assert_called_once_with(offset=0, limit=100)


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_list_datasets_pagination_respects_limit() -> None:
    """``limit`` caps items per call even if the iterator would yield more (SDK pages)."""
    row_mocks = []
    for i in range(5):
        m = MagicMock()
        m.id = str(i)
        m.name = f"ds{i}"
        row_mocks.append(m)
    # Simulate a generator that would keep going (e.g. multiple server pages)
    with patch.object(dr.Dataset, "iterate", return_value=iter(row_mocks[1:])) as mock_iter:
        result = await data.catalog_list_datasets(offset=1, limit=3)

        assert result["count"] == 3
        assert set(result["datasets"].keys()) == {"1", "2", "3"}
        assert result["offset"] == 1
        assert result["limit"] == 3
        assert result["may_have_more"] is True
        mock_iter.assert_called_once_with(offset=1, limit=3)


@pytest.mark.asyncio
async def test_catalog_list_datasets_rejects_negative_offset() -> None:
    with pytest.raises(ToolError, match="offset must be non-negative"):
        await data.catalog_list_datasets(offset=-1)


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_list_datasets_clamps_invalid_limit_below_one() -> None:
    """limit<1 is clamped to 100; response includes a note; iterate uses limit 100."""
    with patch.object(dr.Dataset, "iterate", return_value=iter([])) as mock_iter:
        result = await data.catalog_list_datasets(limit=0)
        assert result["limit"] == 100
        assert "Limit must be at least 1" in (result.get("note") or "")
        mock_iter.assert_called_once_with(offset=0, limit=100)


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_list_datasets_clamps_limit_above_max() -> None:
    with patch.object(dr.Dataset, "iterate", return_value=iter([])) as mock_iter:
        result = await data.catalog_list_datasets(limit=1001)
        assert result["limit"] == 100
        assert "Limit cannot exceed 100" in (result.get("note") or "")
        mock_iter.assert_called_once_with(offset=0, limit=100)


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_list_datasets_uses_default_limit() -> None:
    """Default limit 100 is passed to iterate and echoed in the result when not overridden."""
    mocks = []
    for i in range(3, 6):
        m = MagicMock()
        m.id = str(i)
        m.name = f"ds{i}"
        mocks.append(m)
    with patch.object(dr.Dataset, "iterate", return_value=iter(mocks)) as mock_iter:
        result = await data.catalog_list_datasets(offset=2)
        assert result["count"] == 3
        assert set(result["datasets"].keys()) == {"3", "4", "5"}
        assert result["offset"] == 2
        assert result["limit"] == 100
        assert result["may_have_more"] is False
        mock_iter.assert_called_once_with(offset=2, limit=100)


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_list_datasets_empty() -> None:
    with patch.object(dr.Dataset, "iterate", return_value=iter([])):
        result = await data.catalog_list_datasets()
        assert result["datasets"] == {}
        assert result["count"] == 0
        assert "offset" not in result
        assert result["limit"] == 100


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_list_datasets_empty_paged_includes_count_and_pagination_echo() -> None:
    with patch.object(dr.Dataset, "iterate", return_value=iter([])):
        result = await data.catalog_list_datasets(offset=10, limit=5)
        assert result["datasets"] == {}
        assert result["count"] == 0
        assert result["offset"] == 10
        assert result["limit"] == 5
        assert "may_have_more" not in result


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_list_ai_catalog_may_have_more_false_when_page_not_full() -> None:
    m1, m2 = MagicMock(), MagicMock()
    m1.id, m1.name = "a", "A"
    m2.id, m2.name = "b", "B"
    with patch.object(dr.Dataset, "iterate", return_value=iter([m1, m2])):
        result = await data.catalog_list_datasets(offset=0, limit=5)
        assert result["count"] == 2
        assert result["may_have_more"] is False


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_get_preview_success() -> None:
    mock_dataset = MagicMock()
    mock_dataset.id = "ds1"
    mock_dataset.name = "Test Dataset"
    mock_dataset.created_at = "2025-01-01"
    mock_dataset.row_count = 100

    mock_df = pl.DataFrame({"a": [1, 2], "b": [3, 4]}).to_pandas()
    mock_dataset.get_raw_sample_data.return_value = mock_df
    with patch.object(dr.Dataset, "get", return_value=mock_dataset):
        result = await data.catalog_get_preview(dataset_id="ds1")
        assert isinstance(result, dict)
        assert result["id"] == "ds1"
        assert result["name"] == "Test Dataset"
        assert "columns" in result
        assert "sample" in result
        assert len(result["sample"]) == 2


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_get_preview_respects_sample_rows() -> None:
    """sample_rows caps how many leading rows are returned (head)."""
    mock_dataset = MagicMock()
    mock_dataset.id = "ds1"
    mock_dataset.name = "S"
    mock_dataset.created_at = "2025-01-01"
    mock_dataset.row_count = 5
    mock_df = pl.DataFrame({"a": list(range(5)), "b": list(range(5, 10))}).to_pandas()
    mock_dataset.get_raw_sample_data.return_value = mock_df
    with patch.object(dr.Dataset, "get", return_value=mock_dataset):
        result = await data.catalog_get_preview(dataset_id="ds1", sample_rows=2)
        assert result["sample"] == [
            {"a": 0, "b": 5},
            {"a": 1, "b": 6},
        ]


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_get_preview_sample_rows_zero_returns_no_rows() -> None:
    mock_dataset = MagicMock()
    mock_dataset.id = "ds1"
    mock_dataset.name = "S"
    mock_dataset.created_at = "2025-01-01"
    mock_df = pl.DataFrame({"a": [0]}).to_pandas()
    mock_dataset.get_raw_sample_data.return_value = mock_df
    with patch.object(dr.Dataset, "get", return_value=mock_dataset):
        result = await data.catalog_get_preview(dataset_id="ds1", sample_rows=0)
        assert result["sample"] == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_get_preview_large_sample_rows_returns_full_frame() -> None:
    """head(sample_rows) is bounded by the preview frame; unrelated to catalog pagination max."""
    mock_dataset = MagicMock()
    mock_dataset.id = "ds1"
    mock_dataset.name = "S"
    mock_dataset.created_at = "2025-01-01"
    mock_df = pl.DataFrame({"a": list(range(10))}).to_pandas()
    mock_dataset.get_raw_sample_data.return_value = mock_df
    with patch.object(dr.Dataset, "get", return_value=mock_dataset):
        result = await data.catalog_get_preview(dataset_id="ds1", sample_rows=5000)
        assert len(result["sample"]) == 10


@pytest.mark.asyncio
async def test_catalog_get_preview_missing_id() -> None:
    with pytest.raises(ToolError, match="Dataset ID must be provided"):
        await data.catalog_get_preview()


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_get_preview_no_sample() -> None:
    mock_dataset = MagicMock()
    mock_dataset.id = "ds1"
    mock_dataset.name = "Test"
    mock_dataset.created_at = "2025-01-01"
    mock_dataset.row_count = 50
    with patch.object(dr.Dataset, "get", return_value=mock_dataset):
        result = await data.catalog_get_preview(dataset_id="ds1", include_sample=False)
        assert "sample" not in result


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_list_datastores_success() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {
                "id": "store1",
                "canonicalName": "My Store",
                "creatorId": "user1",
                "params": {"type": "jdbc"},
            }
        ],
        "next": "https://app.datarobot.com/api/v2/externalDataStores/?offset=10&limit=10",
    }
    mock_rest = MagicMock()
    mock_rest.get.return_value = mock_response
    with patch.object(dr.client, "get_client", return_value=mock_rest):
        result = await data.catalog_list_datastores(offset=0, limit=10)
        assert isinstance(result, dict)
        assert result["count"] == 1
        assert result["datastores"][0]["id"] == "store1"
        assert result["offset"] == 0
        assert result["limit"] == 10
        assert "next" in result
        mock_rest.get.assert_called_once_with(
            "externalDataStores/", params={"offset": 0, "limit": 10}
        )


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_list_datastores_serializes_params_from_response() -> None:
    """API `params` may be plain dicts; _serialize_datastore_params keeps output JSON-friendly."""
    params = DataStoreParameters(
        driver_id="pg",
        jdbc_url="jdbc:postgresql://host/db",
        fields=None,
        connector_id=None,
    )
    expected = {
        "driver_id": "pg",
        "jdbc_url": "jdbc:postgresql://host/db",
    }
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {
                "id": "store1",
                "canonicalName": "My Store",
                "creatorId": "user1",
                "params": expected,
            }
        ],
    }
    mock_rest = MagicMock()
    mock_rest.get.return_value = mock_response
    with patch.object(dr.client, "get_client", return_value=mock_rest):
        result = await data.catalog_list_datastores()
        assert data._serialize_datastore_params(params) == expected
        assert result["datastores"][0]["params"] == expected
        assert "offset" not in result
        assert result["limit"] == 100
        mock_rest.get.assert_called_once_with("externalDataStores/", params={"limit": 100})


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_list_datastores_pagination_total_count_from_api() -> None:
    """API may return total_count; merge exposes it as total_count on the tool result."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {
                "id": "a",
                "canonicalName": "A",
                "creatorId": "c1",
                "params": {},
            }
        ],
        "total_count": 42,
        "previous": "https://api/prev",
    }
    mock_rest = MagicMock()
    mock_rest.get.return_value = mock_response
    with patch.object(dr.client, "get_client", return_value=mock_rest):
        result = await data.catalog_list_datastores(offset=5, limit=1)
        assert result["count"] == 1
        assert result["offset"] == 5
        assert result["limit"] == 1
        assert result["total_count"] == 42
        assert result["previous"] == "https://api/prev"
        mock_rest.get.assert_called_once_with(
            "externalDataStores/", params={"offset": 5, "limit": 1}
        )


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_list_datastores_clamps_limit_above_max() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": []}
    mock_rest = MagicMock()
    mock_rest.get.return_value = mock_response
    with patch.object(dr.client, "get_client", return_value=mock_rest):
        result = await data.catalog_list_datastores(limit=1001)
        assert result["limit"] == 100
        assert "Limit cannot exceed 100" in (result.get("note") or "")
        mock_rest.get.assert_called_once_with("externalDataStores/", params={"limit": 100})


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_browse_datastore_success() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": ["table1", "table2"]}
    mock_rest_client = MagicMock()
    mock_rest_client.get.return_value = mock_response
    with patch.object(dr.client, "get_client", return_value=mock_rest_client):
        result = await data.catalog_browse_datastore(datastore_id="store1", path="/schema1")
        assert isinstance(result, dict)
        assert result["count"] == 2
        assert result["datastore_id"] == "store1"
        assert result["offset"] == 0
        assert result["limit"] == 100


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_browse_datastore_pagination_from_api_response() -> None:
    """Browse passes offset/limit to the API and forwards next/total from the response body."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [{"name": "t1"}],
        "next": "https://x/tables?offset=20",
        "total": 100,
    }
    mock_rest = MagicMock()
    mock_rest.get.return_value = mock_response
    with patch.object(dr.client, "get_client", return_value=mock_rest):
        result = await data.catalog_browse_datastore(
            datastore_id="store1",
            path="/s",
            offset=10,
            limit=5,
        )
        assert result["count"] == 1
        assert result["offset"] == 10
        assert result["limit"] == 5
        assert result["next"] == "https://x/tables?offset=20"
        assert result["total_count"] == 100
        mock_rest.get.assert_called_once_with(
            "externalDataDrivers/store1/tables/",
            params={"offset": 10, "limit": 5, "path": "/s"},
        )


@pytest.mark.asyncio
async def test_catalog_browse_datastore_missing_id() -> None:
    with pytest.raises(ToolError, match="Datastore ID must be provided"):
        await data.catalog_browse_datastore()


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_browse_datastore_clamps_limit_above_max() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": []}
    mock_rest = MagicMock()
    mock_rest.get.return_value = mock_response
    with patch.object(dr.client, "get_client", return_value=mock_rest):
        result = await data.catalog_browse_datastore(datastore_id="s1", limit=2000)
        assert result["limit"] == 100
        assert "Limit cannot exceed 100" in (result.get("note") or "")
        mock_rest.get.assert_called_once_with(
            "externalDataDrivers/s1/tables/", params={"offset": 0, "limit": 100}
        )


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_query_datastore_success() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [{"id": 1, "val": "a"}],
        "columns": ["id", "val"],
    }
    mock_rest_client = MagicMock()
    mock_rest_client.post.return_value = mock_response
    with patch.object(dr.client, "get_client", return_value=mock_rest_client):
        result = await data.catalog_query_datastore(datastore_id="store1", sql="SELECT * FROM t")
        assert isinstance(result, dict)
        assert result["row_count"] == 1
        assert result["columns"] == ["id", "val"]
        assert result["offset"] == 0
        assert result["limit"] == 100


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_query_datastore_clamps_limit_above_max() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": [], "columns": []}
    mock_rest = MagicMock()
    mock_rest.post.return_value = mock_response
    with patch.object(dr.client, "get_client", return_value=mock_rest):
        out = await data.catalog_query_datastore(datastore_id="ds1", sql="SELECT 1", limit=2000)
        assert out["limit"] == 100
        assert "Limit cannot exceed 100" in (out.get("note") or "")
        mock_rest.post.assert_called_once_with(
            "externalDataDrivers/ds1/execute/",
            json={"query": "SELECT 1", "offset": 0, "limit": 100},
        )


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_query_datastore_pagination_offset_limit_and_response_metadata() -> None:
    """Query sends offset/limit in the JSON body and merges pagination fields from the response."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [],
        "columns": ["a"],
        "next": "https://x/execute?offset=75",
        "total_count": 200,
    }
    mock_rest = MagicMock()
    mock_rest.post.return_value = mock_response
    with patch.object(dr.client, "get_client", return_value=mock_rest):
        result = await data.catalog_query_datastore(
            datastore_id="ds99",
            sql="SELECT 1",
            offset=50,
            limit=25,
        )
        assert result["row_count"] == 0
        assert result["offset"] == 50
        assert result["limit"] == 25
        assert result["next"] == "https://x/execute?offset=75"
        assert result["total_count"] == 200
        mock_rest.post.assert_called_once_with(
            "externalDataDrivers/ds99/execute/",
            json={"query": "SELECT 1", "offset": 50, "limit": 25},
        )


@pytest.mark.asyncio
async def test_catalog_query_datastore_missing_id() -> None:
    with pytest.raises(ToolError, match="Datastore ID must be provided"):
        await data.catalog_query_datastore(sql="SELECT 1")


@pytest.mark.asyncio
async def test_catalog_query_datastore_missing_sql() -> None:
    with pytest.raises(ToolError, match="SQL query must be provided"):
        await data.catalog_query_datastore(datastore_id="store1")


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_list_datasets_error() -> None:
    with patch.object(dr.Dataset, "iterate", side_effect=Exception("fail")):
        with pytest.raises(Exception) as exc_info:
            await data.catalog_list_datasets()
        assert "fail" in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_catalog_list_datasets_client_error_during_lazy_iteration() -> None:

    def _gen_raises_on_consume():
        raise ClientError(
            "500 client error: {'message': 'Server Error'}",
            status_code=500,
            json={"message": "Server Error"},
        )
        yield  # pragma: no cover — makes this a generator; first next() runs raise first

    with patch.object(dr.Dataset, "iterate", return_value=_gen_raises_on_consume()):
        with pytest.raises(ToolError) as exc_info:
            await data.catalog_list_datasets()
        assert exc_info.value.kind is ToolErrorKind.UPSTREAM
        assert "500" in str(exc_info.value)
