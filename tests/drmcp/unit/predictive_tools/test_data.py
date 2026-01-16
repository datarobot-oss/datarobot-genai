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

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp.tools.predictive import data


@pytest.mark.asyncio
async def test_upload_dataset_to_ai_catalog_success() -> None:
    with (
        patch("datarobot_genai.drmcp.tools.predictive.data.get_sdk_client") as mock_get_client,
        patch("datarobot_genai.drmcp.tools.predictive.data.os.path.exists", return_value=True),
    ):
        mock_client = MagicMock()
        mock_catalog_item = MagicMock()
        mock_catalog_item.id = "12345"
        mock_catalog_item.name = "test_dataset"
        mock_catalog_item.status = "completed"
        mock_catalog_item.version_id = None
        mock_catalog_item.name = "somefile.csv"
        mock_client.Dataset.create_from_file.return_value = mock_catalog_item
        mock_get_client.return_value = mock_client

        result = await data.upload_dataset_to_ai_catalog(file_path="somefile.csv")
        mock_client.Dataset.create_from_file.assert_called_once_with("somefile.csv")
        assert isinstance(result, ToolResult)
        assert result.content[0].text == "Successfully uploaded dataset: 12345"
        assert result.structured_content == {
            "dataset_id": "12345",
            "dataset_version_id": None,
            "dataset_name": "somefile.csv",
        }


@pytest.mark.asyncio
async def test_upload_dataset_to_ai_catalog_success_with_url() -> None:
    with (
        patch("datarobot_genai.drmcp.tools.predictive.data.get_sdk_client") as mock_get_client,
    ):
        mock_client = MagicMock()
        mock_catalog_item = MagicMock()
        mock_catalog_item.id = "12345"
        mock_catalog_item.version_id = None
        mock_catalog_item.name = "somefile.csv"
        mock_client.Dataset.create_from_url.return_value = mock_catalog_item
        mock_get_client.return_value = mock_client

        result = await data.upload_dataset_to_ai_catalog(
            file_url="https://example.com/somefile.csv"
        )
        mock_client.Dataset.create_from_url.assert_called_once_with(
            "https://example.com/somefile.csv"
        )
        assert isinstance(result, ToolResult)
        assert result.content[0].text == "Successfully uploaded dataset: 12345"
        assert result.structured_content == {
            "dataset_id": "12345",
            "dataset_version_id": None,
            "dataset_name": "somefile.csv",
        }


@pytest.mark.asyncio
async def test_upload_dataset_to_ai_catalog_error_with_url() -> None:
    with (
        patch("datarobot_genai.drmcp.tools.predictive.data.get_sdk_client") as mock_get_client,
    ):
        mock_client = MagicMock()
        mock_catalog_item = MagicMock()
        mock_catalog_item.id = "12345"
        mock_catalog_item.version_id = None
        mock_catalog_item.name = "somefile.csv"
        mock_client.Dataset.create_from_url.return_value = mock_catalog_item
        mock_get_client.return_value = mock_client

        result = await data.upload_dataset_to_ai_catalog(file_url="https:notavalidurl/somefile.csv")
        assert isinstance(result, ToolError)
        assert str(result) == "Invalid file URL: https:notavalidurl/somefile.csv"


@pytest.mark.asyncio
async def test_upload_dataset_to_ai_catalog_error_no_file_path_or_url() -> None:
    with (
        patch("datarobot_genai.drmcp.tools.predictive.data.get_sdk_client"),
    ):
        result = await data.upload_dataset_to_ai_catalog()
        assert isinstance(result, ToolError)
        assert str(result) == "Either file_path or file_url must be provided."


@pytest.mark.asyncio
async def test_upload_dataset_to_ai_catalog_error_both_file_path_and_url() -> None:
    with (
        patch("datarobot_genai.drmcp.tools.predictive.data.get_sdk_client"),
    ):
        result = await data.upload_dataset_to_ai_catalog(
            file_path="somefile.csv", file_url="https://example.com/somefile.csv"
        )
        assert isinstance(result, ToolError)
        assert str(result) == "Please provide either file_path or file_url, not both."


@pytest.mark.asyncio
async def test_upload_dataset_to_ai_catalog_file_not_found() -> None:
    with (
        patch("datarobot_genai.drmcp.tools.predictive.data.get_sdk_client"),
        patch("datarobot_genai.drmcp.tools.predictive.data.os.path.exists", return_value=False),
    ):
        result = await data.upload_dataset_to_ai_catalog(file_path="nofile.csv")
        assert isinstance(result, ToolError)
        assert str(result) == "File not found: nofile.csv"


@pytest.mark.asyncio
async def test_upload_dataset_to_ai_catalog_error() -> None:
    with (
        patch("datarobot_genai.drmcp.tools.predictive.data.get_sdk_client") as mock_get_client,
        patch("datarobot_genai.drmcp.tools.predictive.data.os.path.exists", return_value=True),
    ):
        mock_client = MagicMock()
        mock_client.Dataset.create_from_file.side_effect = Exception("fail")
        mock_get_client.return_value = mock_client

        with pytest.raises(Exception) as exc_info:
            await data.upload_dataset_to_ai_catalog(file_path="somefile.csv")
        assert "fail" in str(exc_info.value)


@pytest.mark.asyncio
async def test_list_ai_catalog_items_success() -> None:
    with patch("datarobot_genai.drmcp.tools.predictive.data.get_sdk_client") as mock_get_client:
        mock_client = MagicMock()
        mock_ds1 = MagicMock()
        mock_ds1.id = "1"
        mock_ds1.name = "ds1"
        mock_ds2 = MagicMock()
        mock_ds2.id = "2"
        mock_ds2.name = "ds2"
        mock_client.Dataset.list.return_value = [mock_ds1, mock_ds2]
        mock_get_client.return_value = mock_client

        result = await data.list_ai_catalog_items()
        # Access text from TextContent object
        content_text = result.content[0].text
        assert "Found 2 AI Catalog items" in content_text
        assert result.structured_content["count"] == 2
        # datasets is now a dict mapping id to name
        assert result.structured_content["datasets"]["1"] == "ds1"
        assert result.structured_content["datasets"]["2"] == "ds2"


@pytest.mark.asyncio
async def test_list_ai_catalog_items_empty() -> None:
    with patch("datarobot_genai.drmcp.tools.predictive.data.get_sdk_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.Dataset.list.return_value = []
        mock_get_client.return_value = mock_client

        result = await data.list_ai_catalog_items()
        # Access text from TextContent object
        assert "No AI Catalog items found." in result.content[0].text
        assert result.structured_content["datasets"] == []


@pytest.mark.asyncio
async def test_list_ai_catalog_items_error() -> None:
    with patch("datarobot_genai.drmcp.tools.predictive.data.get_sdk_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.Dataset.list.side_effect = Exception("fail")
        mock_get_client.return_value = mock_client

        with pytest.raises(Exception) as exc_info:
            await data.list_ai_catalog_items()
        assert "fail" in str(exc_info.value)
