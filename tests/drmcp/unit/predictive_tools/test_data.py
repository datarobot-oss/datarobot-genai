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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import polars as pl
import pytest
from datarobot.models.data_store import DataStoreParameters

from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.predictive import data


@pytest.mark.asyncio
async def test_upload_dataset_to_ai_catalog_success() -> None:
    with (
        patch(
            "datarobot_genai.drtools.predictive.data.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.data.DataRobotClient") as mock_data_robot_client,
        patch("datarobot_genai.drtools.predictive.data.os.path.exists", return_value=True),
    ):
        mock_client = MagicMock()
        mock_catalog_item = MagicMock()
        mock_catalog_item.id = "12345"
        mock_catalog_item.name = "test_dataset"
        mock_catalog_item.status = "completed"
        mock_catalog_item.version_id = None
        mock_catalog_item.name = "somefile.csv"
        mock_client.Dataset.create_from_file.return_value = mock_catalog_item
        mock_data_robot_client.return_value.get_client.return_value = mock_client

        result = await data.upload_dataset_to_ai_catalog(file_path="somefile.csv")
        mock_client.Dataset.create_from_file.assert_called_once_with("somefile.csv")
        assert isinstance(result, dict)
        assert result == {
            "dataset_id": "12345",
            "dataset_version_id": None,
            "dataset_name": "somefile.csv",
        }


@pytest.mark.asyncio
async def test_upload_dataset_to_ai_catalog_success_with_url() -> None:
    with (
        patch(
            "datarobot_genai.drtools.predictive.data.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.data.DataRobotClient") as mock_data_robot_client,
    ):
        mock_client = MagicMock()
        mock_catalog_item = MagicMock()
        mock_catalog_item.id = "12345"
        mock_catalog_item.version_id = None
        mock_catalog_item.name = "somefile.csv"
        mock_client.Dataset.create_from_url.return_value = mock_catalog_item
        mock_data_robot_client.return_value.get_client.return_value = mock_client

        result = await data.upload_dataset_to_ai_catalog(
            file_url="https://example.com/somefile.csv"
        )
        mock_client.Dataset.create_from_url.assert_called_once_with(
            "https://example.com/somefile.csv"
        )
        assert isinstance(result, dict)
        assert result == {
            "dataset_id": "12345",
            "dataset_version_id": None,
            "dataset_name": "somefile.csv",
        }


@pytest.mark.asyncio
async def test_upload_dataset_to_ai_catalog_error_with_url() -> None:
    with (
        patch(
            "datarobot_genai.drtools.predictive.data.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.data.DataRobotClient") as mock_data_robot_client,
    ):
        mock_client = MagicMock()
        mock_catalog_item = MagicMock()
        mock_catalog_item.id = "12345"
        mock_catalog_item.version_id = None
        mock_catalog_item.name = "somefile.csv"
        mock_client.Dataset.create_from_url.return_value = mock_catalog_item
        mock_data_robot_client.return_value.get_client.return_value = mock_client

        with pytest.raises(
            ToolError,
            match="Invalid file URL: https:notavalidurl/somefile.csv",
        ):
            await data.upload_dataset_to_ai_catalog(file_url="https:notavalidurl/somefile.csv")


@pytest.mark.asyncio
async def test_upload_dataset_to_ai_catalog_error_no_file_path_or_url() -> None:
    with (
        patch(
            "datarobot_genai.drtools.predictive.data.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.data.DataRobotClient"),
    ):
        with pytest.raises(
            ToolError,
            match="Either file_path or file_url must be provided.",
        ):
            await data.upload_dataset_to_ai_catalog()


@pytest.mark.asyncio
async def test_upload_dataset_to_ai_catalog_error_both_file_path_and_url() -> None:
    with (
        patch(
            "datarobot_genai.drtools.predictive.data.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.data.DataRobotClient"),
    ):
        with pytest.raises(
            ToolError,
            match="Please provide either file_path or file_url, not both.",
        ):
            await data.upload_dataset_to_ai_catalog(
                file_path="somefile.csv", file_url="https://example.com/somefile.csv"
            )


@pytest.mark.asyncio
async def test_upload_dataset_to_ai_catalog_file_not_found() -> None:
    with (
        patch(
            "datarobot_genai.drtools.predictive.data.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.data.DataRobotClient"),
        patch("datarobot_genai.drtools.predictive.data.os.path.exists", return_value=False),
    ):
        with pytest.raises(
            ToolError,
            match="File not found: nofile.csv",
        ):
            await data.upload_dataset_to_ai_catalog(file_path="nofile.csv")


@pytest.mark.asyncio
async def test_upload_dataset_to_ai_catalog_error() -> None:
    with (
        patch(
            "datarobot_genai.drtools.predictive.data.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.data.DataRobotClient") as mock_data_robot_client,
        patch("datarobot_genai.drtools.predictive.data.os.path.exists", return_value=True),
    ):
        mock_client = MagicMock()
        mock_client.Dataset.create_from_file.side_effect = Exception("fail")
        mock_data_robot_client.return_value.get_client.return_value = mock_client

        with pytest.raises(Exception) as exc_info:
            await data.upload_dataset_to_ai_catalog(file_path="somefile.csv")
        assert "fail" in str(exc_info.value)


@pytest.mark.asyncio
async def test_list_ai_catalog_items_success() -> None:
    with (
        patch(
            "datarobot_genai.drtools.predictive.data.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.data.DataRobotClient") as mock_data_robot_client,
    ):
        mock_client = MagicMock()
        mock_ds1 = MagicMock()
        mock_ds1.id = "1"
        mock_ds1.name = "ds1"
        mock_ds2 = MagicMock()
        mock_ds2.id = "2"
        mock_ds2.name = "ds2"
        mock_client.Dataset.list.return_value = [mock_ds1, mock_ds2]
        mock_data_robot_client.return_value.get_client.return_value = mock_client

        result = await data.list_ai_catalog_items()
        assert result["count"] == 2
        # datasets is now a dict mapping id to name
        assert result["datasets"]["1"] == "ds1"
        assert result["datasets"]["2"] == "ds2"


@pytest.mark.asyncio
async def test_list_ai_catalog_items_empty() -> None:
    with (
        patch(
            "datarobot_genai.drtools.predictive.data.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.data.DataRobotClient") as mock_data_robot_client,
    ):
        mock_client = MagicMock()
        mock_client.Dataset.list.return_value = []
        mock_data_robot_client.return_value.get_client.return_value = mock_client

        result = await data.list_ai_catalog_items()
        assert result["datasets"] == []


@pytest.mark.asyncio
async def test_get_dataset_details_success() -> None:
    with (
        patch(
            "datarobot_genai.drtools.predictive.data.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.data.DataRobotClient") as mock_data_robot_client,
    ):
        mock_client = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.id = "ds1"
        mock_dataset.name = "Test Dataset"
        mock_dataset.created_at = "2025-01-01"
        mock_dataset.row_count = 100

        mock_df = pl.DataFrame({"a": [1, 2], "b": [3, 4]}).to_pandas()
        mock_dataset.get_raw_sample_data.return_value = mock_df
        mock_client.Dataset.get.return_value = mock_dataset
        mock_data_robot_client.return_value.get_client.return_value = mock_client

        result = await data.get_dataset_details(dataset_id="ds1")
        assert isinstance(result, dict)
        assert result["id"] == "ds1"
        assert result["name"] == "Test Dataset"
        assert "columns" in result
        assert "sample" in result


@pytest.mark.asyncio
async def test_get_dataset_details_missing_id() -> None:
    with pytest.raises(ToolError, match="Dataset ID must be provided"):
        await data.get_dataset_details()


@pytest.mark.asyncio
async def test_get_dataset_details_no_sample() -> None:
    with (
        patch(
            "datarobot_genai.drtools.predictive.data.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.data.DataRobotClient") as mock_data_robot_client,
    ):
        mock_client = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.id = "ds1"
        mock_dataset.name = "Test"
        mock_dataset.created_at = "2025-01-01"
        mock_dataset.row_count = 50
        mock_client.Dataset.get.return_value = mock_dataset
        mock_data_robot_client.return_value.get_client.return_value = mock_client

        result = await data.get_dataset_details(dataset_id="ds1", include_sample=False)
        assert "sample" not in result


@pytest.mark.asyncio
async def test_list_datastores_success() -> None:
    with (
        patch(
            "datarobot_genai.drtools.predictive.data.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.data.DataRobotClient") as mock_data_robot_client,
    ):
        mock_client = MagicMock()
        mock_ds = MagicMock()
        mock_ds.id = "store1"
        mock_ds.canonical_name = "My Store"
        mock_ds.creator_id = "user1"
        mock_ds.params = {"type": "jdbc"}
        mock_client.DataStore.list.return_value = [mock_ds]
        mock_data_robot_client.return_value.get_client.return_value = mock_client

        result = await data.list_datastores()
        assert isinstance(result, dict)
        assert result["count"] == 1
        assert result["datastores"][0]["id"] == "store1"


@pytest.mark.asyncio
async def test_list_datastores_serializes_sdk_params() -> None:
    """SDK returns DataStoreParameters objects; tool output must be JSON-serializable."""
    params = DataStoreParameters(
        driver_id="pg",
        jdbc_url="jdbc:postgresql://host/db",
        fields=None,
        connector_id=None,
    )
    with (
        patch(
            "datarobot_genai.drtools.predictive.data.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.data.DataRobotClient") as mock_data_robot_client,
    ):
        mock_client = MagicMock()
        mock_ds = MagicMock()
        mock_ds.id = "store1"
        mock_ds.canonical_name = "My Store"
        mock_ds.creator_id = "user1"
        mock_ds.params = params
        mock_client.DataStore.list.return_value = [mock_ds]
        mock_data_robot_client.return_value.get_client.return_value = mock_client

        result = await data.list_datastores()
        assert result["datastores"][0]["params"] == {
            "driver_id": "pg",
            "jdbc_url": "jdbc:postgresql://host/db",
        }


@pytest.mark.asyncio
async def test_browse_datastore_success() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": ["table1", "table2"]}
    with (
        patch(
            "datarobot_genai.drtools.predictive.data.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.data.DataRobotClient") as mock_data_robot_client,
    ):
        mock_rest_client = MagicMock()
        mock_rest_client.get.return_value = mock_response
        mock_dr_module = MagicMock()
        mock_dr_module.client.get_client.return_value = mock_rest_client
        mock_data_robot_client.return_value.get_client.return_value = mock_dr_module

        result = await data.browse_datastore(datastore_id="store1", path="/schema1")
        assert isinstance(result, dict)
        assert result["count"] == 2
        assert result["datastore_id"] == "store1"


@pytest.mark.asyncio
async def test_browse_datastore_missing_id() -> None:
    with pytest.raises(ToolError, match="Datastore ID must be provided"):
        await data.browse_datastore()


@pytest.mark.asyncio
async def test_query_datastore_success() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [{"id": 1, "val": "a"}],
        "columns": ["id", "val"],
    }
    with (
        patch(
            "datarobot_genai.drtools.predictive.data.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.data.DataRobotClient") as mock_data_robot_client,
    ):
        mock_rest_client = MagicMock()
        mock_rest_client.post.return_value = mock_response
        mock_dr_module = MagicMock()
        mock_dr_module.client.get_client.return_value = mock_rest_client
        mock_data_robot_client.return_value.get_client.return_value = mock_dr_module

        result = await data.query_datastore(datastore_id="store1", sql="SELECT * FROM t")
        assert isinstance(result, dict)
        assert result["row_count"] == 1
        assert result["columns"] == ["id", "val"]


@pytest.mark.asyncio
async def test_query_datastore_missing_id() -> None:
    with pytest.raises(ToolError, match="Datastore ID must be provided"):
        await data.query_datastore(sql="SELECT 1")


@pytest.mark.asyncio
async def test_query_datastore_missing_sql() -> None:
    with pytest.raises(ToolError, match="SQL query must be provided"):
        await data.query_datastore(datastore_id="store1")


@pytest.mark.asyncio
async def test_list_ai_catalog_items_error() -> None:
    with (
        patch(
            "datarobot_genai.drtools.predictive.data.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.data.DataRobotClient") as mock_data_robot_client,
    ):
        mock_client = MagicMock()
        mock_client.Dataset.list.side_effect = Exception("fail")
        mock_data_robot_client.return_value.get_client.return_value = mock_client

        with pytest.raises(Exception) as exc_info:
            await data.list_ai_catalog_items()
        assert "fail" in str(exc_info.value)
