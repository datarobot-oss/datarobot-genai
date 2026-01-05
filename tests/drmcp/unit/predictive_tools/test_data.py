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

from datarobot_genai.drmcp.core.exceptions import MCPError
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
        mock_client.Dataset.create_from_file.return_value = mock_catalog_item
        mock_get_client.return_value = mock_client

        result = await data.upload_dataset_to_ai_catalog("somefile.csv")
        mock_client.Dataset.create_from_file.assert_called_once_with("somefile.csv")
        # Access text from TextContent object
        assert "Successfully uploaded dataset: 12345" in result.content[0].text
        assert result.structured_content["id"] == "12345"
        assert result.structured_content["name"] == "test_dataset"
        assert result.structured_content["status"] == "completed"


@pytest.mark.asyncio
async def test_upload_dataset_to_ai_catalog_file_not_found() -> None:
    with (
        patch("datarobot_genai.drmcp.tools.predictive.data.get_sdk_client"),
        patch("datarobot_genai.drmcp.tools.predictive.data.os.path.exists", return_value=False),
    ):
        with pytest.raises(MCPError) as exc_info:
            await data.upload_dataset_to_ai_catalog("nofile.csv")
        assert "File not found: nofile.csv" in str(exc_info.value)


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
            await data.upload_dataset_to_ai_catalog("somefile.csv")
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
        assert "Found 2 AI Catalog items." in content_text
        assert result.structured_content["count"] == 2
        assert len(result.structured_content["datasets"]) == 2
        assert result.structured_content["datasets"][0]["id"] == "1"
        assert result.structured_content["datasets"][0]["name"] == "ds1"
        assert result.structured_content["datasets"][1]["id"] == "2"
        assert result.structured_content["datasets"][1]["name"] == "ds2"


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
