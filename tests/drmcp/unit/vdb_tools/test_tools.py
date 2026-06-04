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

from collections.abc import Iterator
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import datarobot as dr
import pytest
from datarobot.errors import ClientError

from datarobot_genai.drtools.core.clients.datarobot import ThreadSafeDataRobotClient
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind
from datarobot_genai.drtools.vdb import tools


@pytest.fixture
def mock_request_user_client() -> Iterator[Mock]:
    with patch.object(ThreadSafeDataRobotClient, "request_user_client") as mock_func:
        yield mock_func


@pytest.fixture
def mock_get_client_context_with_token_from_request_header(
    mock_request_user_client: Mock,
) -> Mock:
    return mock_request_user_client


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_list_success() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {
                "id": "dep1",
                "label": "VDB 1",
                "status": "active",
                "capabilities": {"supportsVectorDatabaseQuerying": True},
                "model": {"targetType": "VectorDatabase"},
            },
        ]
    }
    mock_rest_client = MagicMock()
    mock_rest_client.get.return_value = mock_response
    with patch.object(dr.client, "get_client", return_value=mock_rest_client):
        result = await tools.vdb_list()
        assert isinstance(result, dict)
        assert result["count"] == 1
        assert result["vector_databases"][0]["deployment_id"] == "dep1"
        mock_rest_client.get.assert_called_once_with(
            "deployments/",
            params={"limit": 100, "modelTargetType": "VectorDatabase"},
        )


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_list_pagination_params() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": [], "next": "url", "total_count": 7}
    mock_rest_client = MagicMock()
    mock_rest_client.get.return_value = mock_response
    with patch.object(dr.client, "get_client", return_value=mock_rest_client):
        result = await tools.vdb_list(offset=10, limit=25)
        mock_rest_client.get.assert_called_once_with(
            "deployments/",
            params={"limit": 25, "modelTargetType": "VectorDatabase", "offset": 10},
        )
        assert result["offset"] == 10
        assert result["limit"] == 25
        assert result["next"] == "url"
        assert result["total_count"] == 7


@pytest.mark.asyncio
async def test_vdb_list_negative_offset_validation() -> None:
    with pytest.raises(ToolError, match="offset must be non-negative"):
        await tools.vdb_list(offset=-1)


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_list_empty() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": []}
    mock_rest_client = MagicMock()
    mock_rest_client.get.return_value = mock_response
    with patch.object(dr.client, "get_client", return_value=mock_rest_client):
        result = await tools.vdb_list()
        assert result["count"] == 0
        assert result["vector_databases"] == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_query_success() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": [{"content": "doc1", "metadata": {}}]}
    mock_rest_client = MagicMock()
    mock_rest_client.post.return_value = mock_response
    with patch.object(dr.client, "get_client", return_value=mock_rest_client):
        result = await tools.vdb_query(deployment_id="dep1", query="test query")
        assert isinstance(result, dict)
        assert result["count"] == 1
        assert result["deployment_id"] == "dep1"


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_list_client_error_404() -> None:
    mock_rest_client = MagicMock()
    mock_rest_client.get.side_effect = ClientError(
        "404 client error: {'message': 'Not Found'}",
        status_code=404,
        json={"message": "Not Found"},
    )
    with patch.object(dr.client, "get_client", return_value=mock_rest_client):
        with pytest.raises(ToolError) as exc_info:
            await tools.vdb_list()
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND
    assert "404" in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_query_client_error_404() -> None:
    mock_rest_client = MagicMock()
    mock_rest_client.post.side_effect = ClientError(
        "404 client error: {'message': 'Not Found'}",
        status_code=404,
        json={"message": "Not Found"},
    )
    with patch.object(dr.client, "get_client", return_value=mock_rest_client):
        with pytest.raises(ToolError) as exc_info:
            await tools.vdb_query(deployment_id="missing-dep", query="test")
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND
    assert "404" in str(exc_info.value)


@pytest.mark.asyncio
async def test_vdb_query_missing_deployment_id() -> None:
    with pytest.raises(ToolError, match="Deployment ID must be provided"):
        await tools.vdb_query(query="test")


@pytest.mark.asyncio
async def test_vdb_query_missing_query() -> None:
    with pytest.raises(ToolError, match="Query must be provided"):
        await tools.vdb_query(deployment_id="dep1")
