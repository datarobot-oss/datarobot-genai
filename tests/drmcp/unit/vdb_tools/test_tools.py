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
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import datarobot as dr
import pytest
from datarobot.errors import ClientError
from datarobot.models.genai.vector_database import VectorDatabase

from datarobot_genai.drmcputils.clients.datarobot import ThreadSafeDataRobotClient
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.vdb import tools

VDB_ID = "69cbb73789723b6936c6c9e1"


def _vdb_deployment_record(
    deployment_id: str,
    *,
    label: str = "My VDB Deployment",
    status: str = "launching",
) -> dict[str, object]:
    return {
        "id": deployment_id,
        "label": label,
        "status": status,
        "model": {"targetType": "VectorDatabase"},
    }


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
    mock_deployments = [
        {
            "id": "dep1",
            "label": "VDB 1",
            "status": "active",
            "capabilities": {"supportsVectorDatabaseQuerying": True},
            "model": {"targetType": "VectorDatabase"},
        },
        {
            "id": "dep2",
            "label": "Regular Deployment",
            "status": "active",
            "capabilities": {"supportsVectorDatabaseQuerying": False},
            "model": {"targetType": "Binary"},
        },
    ]
    mock_rest_client = MagicMock()
    with (
        patch.object(dr.client, "get_client", return_value=mock_rest_client),
        patch.object(
            dr.utils.pagination,
            "unpaginate",
            return_value=iter(mock_deployments),
        ),
    ):
        result = await tools.vdb_list()
        assert isinstance(result, dict)
        assert result["count"] == 1
        assert result["vector_databases"][0]["deployment_id"] == "dep1"
        assert result["total_count"] == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_list_pagination_params() -> None:
    mock_deployments = [
        {
            "id": f"vdb-{i}",
            "label": f"VDB {i}",
            "status": "active",
            "model": {"targetType": "VectorDatabase"},
        }
        for i in range(12)
    ]
    mock_rest_client = MagicMock()
    with (
        patch.object(dr.client, "get_client", return_value=mock_rest_client),
        patch.object(
            dr.utils.pagination,
            "unpaginate",
            return_value=iter(mock_deployments),
        ),
    ):
        result = await tools.vdb_list(offset=10, limit=25)
        assert result["offset"] == 10
        assert result["limit"] == 25
        assert result["count"] == 2
        assert result["total_count"] == 12
        assert result["vector_databases"][0]["deployment_id"] == "vdb-10"


@pytest.mark.asyncio
async def test_vdb_list_negative_offset_validation() -> None:
    with pytest.raises(ToolError, match="offset must be non-negative"):
        await tools.vdb_list(offset=-1)


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_list_empty() -> None:
    mock_rest_client = MagicMock()
    with (
        patch.object(dr.client, "get_client", return_value=mock_rest_client),
        patch.object(dr.utils.pagination, "unpaginate", return_value=iter([])),
    ):
        result = await tools.vdb_list()
        assert result["count"] == 0
        assert result["vector_databases"] == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_query_success(
    mock_get_client_context_with_token_from_request_header: Mock,
) -> None:
    mock_deployment = SimpleNamespace(
        id="dep1",
        default_prediction_server={
            "id": "srv1",
            "url": "https://pred.example.com",
            "datarobot-key": "key1",
        },
        prediction_environment=None,
    )
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {
                "page_content": "DataRobot is a leading AI platform.",
                "metadata": {"source": "doc1.pdf"},
                "score": 0.92,
            }
        ]
    }
    mock_rest_client = MagicMock()
    mock_rest_client.request.return_value = mock_response
    mock_get_client_context_with_token_from_request_header.return_value.__enter__.return_value = (
        mock_rest_client
    )
    with (
        patch.object(dr.Deployment, "get", return_value=mock_deployment),
        patch(
            "datarobot_genai.drtools.vdb.tools.get_datarobot_access_token",
            return_value="token",
        ),
    ):
        result = await tools.vdb_query(deployment_id="dep1", query="test query")
        assert isinstance(result, dict)
        assert result["count"] == 1
        assert result["deployment_id"] == "dep1"
        assert result["documents"][0]["page_content"] == "DataRobot is a leading AI platform."
        mock_rest_client.request.assert_called_once()
        call_args = mock_rest_client.request.call_args
        assert call_args.args[0] == "POST"
        assert call_args.args[1].endswith("/deployments/dep1/predictions")
        assert call_args.kwargs["json"] == [
            {
                "promptText": "test query",
                "num_results": 5,
                "retrieval_mode": "similarity",
            }
        ]


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_query_parses_prediction_server_response(
    mock_get_client_context_with_token_from_request_header: Mock,
) -> None:
    mock_deployment = SimpleNamespace(
        id="dep1",
        default_prediction_server={
            "id": "srv1",
            "url": "https://pred.example.com",
            "datarobot-key": "key1",
        },
        prediction_environment=None,
    )
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {
                "rowId": 0,
                "prediction": ["chunk one", "chunk two"],
                "predictionValues": [{"label": "relevant", "value": ["chunk one", "chunk two"]}],
            }
        ]
    }
    mock_rest_client = MagicMock()
    mock_rest_client.request.return_value = mock_response
    mock_get_client_context_with_token_from_request_header.return_value.__enter__.return_value = (
        mock_rest_client
    )
    with (
        patch.object(dr.Deployment, "get", return_value=mock_deployment),
        patch(
            "datarobot_genai.drtools.vdb.tools.get_datarobot_access_token",
            return_value="token",
        ),
    ):
        result = await tools.vdb_query(deployment_id="dep1", query="test query")
        assert result["count"] == 2
        assert result["documents"][0]["page_content"] == "chunk one"
        assert result["documents"][1]["page_content"] == "chunk two"


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_list_client_error_404() -> None:
    mock_rest_client = MagicMock()
    with (
        patch.object(dr.client, "get_client", return_value=mock_rest_client),
        patch.object(
            dr.utils.pagination,
            "unpaginate",
            side_effect=ClientError(
                "404 client error: {'message': 'Not Found'}",
                status_code=404,
                json={"message": "Not Found"},
            ),
        ),
    ):
        with pytest.raises(ToolError) as exc_info:
            await tools.vdb_list()
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND
    assert "404" in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_query_client_error_404(
    mock_get_client_context_with_token_from_request_header: Mock,
) -> None:
    mock_deployment = SimpleNamespace(
        id="missing-dep",
        default_prediction_server={
            "id": "srv1",
            "url": "https://pred.example.com",
            "datarobot-key": "key1",
        },
        prediction_environment=None,
    )
    mock_rest_client = MagicMock()
    mock_rest_client.request.side_effect = ClientError(
        "404 client error: {'message': 'Not Found'}",
        status_code=404,
        json={"message": "Not Found"},
    )
    mock_get_client_context_with_token_from_request_header.return_value.__enter__.return_value = (
        mock_rest_client
    )
    with (
        patch.object(dr.Deployment, "get", return_value=mock_deployment),
        patch(
            "datarobot_genai.drtools.vdb.tools.get_datarobot_access_token",
            return_value="token",
        ),
    ):
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


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_create_success() -> None:
    mock_vdb = SimpleNamespace(
        id="vdb-new-1",
        name="My VDB",
        execution_status="new",
        use_case_id="uc1",
        dataset_id="ds1",
    )
    with patch.object(VectorDatabase, "create", return_value=mock_vdb) as mock_create:
        result = await tools.vdb_create(
            dataset_id="ds1",
            use_case_id="uc1",
            name="My VDB",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            chunking_method="recursive",
            chunk_size=256,
            chunk_overlap_percentage=10,
            separators=["\n\n", "\n", " "],
        )
        assert result["vector_database_id"] == "vdb-new-1"
        assert result["name"] == "My VDB"
        assert result["execution_status"] == "new"
        assert result["use_case_id"] == "uc1"
        assert result["dataset_id"] == "ds1"
        assert "note" in result
        assert (
            result["chunking_parameters"]["embedding_model"]
            == "sentence-transformers/all-MiniLM-L6-v2"
        )
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["dataset_id"] == "ds1"
        assert call_kwargs["use_case"] == "uc1"
        assert call_kwargs["name"] == "My VDB"
        assert call_kwargs["chunking_parameters"] is not None
        assert (
            call_kwargs["chunking_parameters"].embedding_model
            == "sentence-transformers/all-MiniLM-L6-v2"
        )


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_create_applies_default_chunking_parameters() -> None:
    mock_vdb = SimpleNamespace(
        id="vdb-new-2",
        name="Vector Database for dataset",
        execution_status="new",
        use_case_id="uc1",
        dataset_id="ds1",
    )
    with patch.object(VectorDatabase, "create", return_value=mock_vdb) as mock_create:
        result = await tools.vdb_create(dataset_id="ds1", use_case_id="uc1")
        assert result["vector_database_id"] == "vdb-new-2"
        chunking = mock_create.call_args.kwargs["chunking_parameters"]
        assert chunking.embedding_model == tools.DEFAULT_EMBEDDING_MODEL
        assert chunking.chunking_method == tools.DEFAULT_CHUNKING_METHOD
        assert chunking.chunk_size == tools.DEFAULT_CHUNK_SIZE
        assert chunking.chunk_overlap_percentage == tools.DEFAULT_CHUNK_OVERLAP_PERCENTAGE
        assert chunking.separators == tools.DEFAULT_SEPARATORS


@pytest.mark.asyncio
async def test_vdb_create_invalid_chunk_size() -> None:
    with pytest.raises(ToolError, match="chunk_size must be between"):
        await tools.vdb_create(dataset_id="ds1", use_case_id="uc1", chunk_size=512)


@pytest.mark.asyncio
async def test_vdb_create_invalid_embedding_model() -> None:
    with pytest.raises(ToolError, match="embedding_model must be one of"):
        await tools.vdb_create(
            dataset_id="ds1",
            use_case_id="uc1",
            embedding_model="text-embedding-3-small",
        )


@pytest.mark.asyncio
async def test_vdb_create_missing_dataset_id() -> None:
    with pytest.raises(ToolError, match="dataset_id must be provided"):
        await tools.vdb_create(use_case_id="uc1")


@pytest.mark.asyncio
async def test_vdb_create_missing_use_case_id() -> None:
    with pytest.raises(ToolError, match="use_case_id must be provided"):
        await tools.vdb_create(dataset_id="ds1")


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_create_client_error() -> None:
    with patch.object(
        VectorDatabase,
        "create",
        side_effect=ClientError(
            "404 client error: {'message': 'Dataset not found'}",
            status_code=404,
            json={"message": "Dataset not found"},
        ),
    ):
        with pytest.raises(ToolError) as exc_info:
            await tools.vdb_create(dataset_id="missing", use_case_id="uc1")
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_deploy_success() -> None:
    mock_server = SimpleNamespace(id="srv1")
    mock_vdb = MagicMock()
    mock_vdb.id = "vdb-123"
    mock_vdb.execution_status = "COMPLETED"
    mock_rest_client = MagicMock()
    mock_rest_client.post.return_value = MagicMock(
        headers={"Location": "https://example.com/async/status/1/"}
    )
    new_deployment = _vdb_deployment_record("dep-vdb-1")
    unpaginate_calls = {"count": 0}

    def unpaginate_side_effect(**kwargs: object) -> Iterator[dict[str, object]]:
        unpaginate_calls["count"] += 1
        if unpaginate_calls["count"] == 1:
            return iter([])
        return iter([new_deployment])

    with (
        patch.object(dr.PredictionServer, "list", return_value=[mock_server]),
        patch.object(VectorDatabase, "get", return_value=mock_vdb) as mock_get,
        patch.object(dr.client, "get_client", return_value=mock_rest_client),
        patch.object(
            dr.utils.pagination,
            "unpaginate",
            side_effect=unpaginate_side_effect,
        ),
    ):
        result = await tools.vdb_deploy(vector_database_id="vdb-123")
        assert result["vector_database_id"] == "vdb-123"
        assert result["deployment_id"] == "dep-vdb-1"
        assert result["label"] == "My VDB Deployment"
        assert result["status"] == "launching"
        assert "note" in result
        mock_get.assert_called_once_with("vdb-123")
        mock_rest_client.post.assert_called_once_with(
            "genai/vectorDatabases/vdb-123/deployments/",
            data={"default_prediction_server_id": "srv1"},
        )
        mock_vdb.deploy.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_deploy_build_not_complete() -> None:
    mock_vdb = MagicMock()
    mock_vdb.execution_status = "RUNNING"
    with patch.object(VectorDatabase, "get", return_value=mock_vdb):
        with pytest.raises(ToolError, match="build is not complete"):
            await tools.vdb_deploy(vector_database_id="vdb-123")


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_get_vector_database_status() -> None:
    mock_vdb = SimpleNamespace(id=VDB_ID, execution_status="RUNNING")
    mock_rest_client = MagicMock()
    mock_rest_client.get.return_value = MagicMock(json=lambda: {"id": VDB_ID})
    with (
        patch.object(dr.client, "get_client", return_value=mock_rest_client),
        patch.object(VectorDatabase, "from_server_data", return_value=mock_vdb),
    ):
        result = await tools.vdb_get(vector_database_id=VDB_ID)
        assert result["vector_database_id"] == VDB_ID
        assert result["execution_status"] == "RUNNING"
        mock_rest_client.get.assert_called_once_with(
            f"genai/vectorDatabases/{VDB_ID}/",
            allow_redirects=False,
        )


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_get_vector_database_target_reached() -> None:
    mock_vdb = SimpleNamespace(id=VDB_ID, execution_status="COMPLETED")
    mock_rest_client = MagicMock()
    mock_rest_client.get.return_value = MagicMock(json=lambda: {"id": VDB_ID})
    with (
        patch.object(dr.client, "get_client", return_value=mock_rest_client),
        patch.object(VectorDatabase, "from_server_data", return_value=mock_vdb),
    ):
        result = await tools.vdb_get(
            vector_database_id=VDB_ID,
            target_status="completed",
        )
        assert result["target_reached"] is True


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_get_deployment_status() -> None:
    mock_rest_client = MagicMock()
    mock_rest_client.get.return_value = MagicMock(
        json=lambda: {
            "id": "dep-1",
            "label": "My VDB",
            "status": "launching",
        }
    )
    with patch.object(dr.client, "get_client", return_value=mock_rest_client):
        result = await tools.vdb_get(deployment_id="dep-1", target_status="active")
        assert result["deployment_id"] == "dep-1"
        assert result["status"] == "launching"
        assert result["target_reached"] is False
        mock_rest_client.get.assert_called_once_with(
            "deployments/dep-1/",
            allow_redirects=False,
        )


@pytest.mark.asyncio
async def test_vdb_get_requires_exactly_one_id() -> None:
    with pytest.raises(ToolError, match="Exactly one of"):
        await tools.vdb_get()
    with pytest.raises(ToolError, match="Exactly one of"):
        await tools.vdb_get(vector_database_id="vdb-1", deployment_id="dep-1")


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_get_build_failure_raises() -> None:
    mock_vdb = SimpleNamespace(id=VDB_ID, execution_status="ERROR")
    mock_rest_client = MagicMock()
    mock_rest_client.get.return_value = MagicMock(json=lambda: {"id": VDB_ID})
    with (
        patch.object(dr.client, "get_client", return_value=mock_rest_client),
        patch.object(VectorDatabase, "from_server_data", return_value=mock_vdb),
    ):
        with pytest.raises(ToolError, match="terminal status"):
            await tools.vdb_get(vector_database_id=VDB_ID)


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_deploy_with_prediction_environment() -> None:
    mock_vdb = MagicMock()
    mock_vdb.id = "vdb-123"
    mock_vdb.execution_status = "COMPLETED"
    mock_rest_client = MagicMock()
    mock_rest_client.post.return_value = MagicMock(
        headers={"Location": "https://example.com/async/status/2/"}
    )
    new_deployment = _vdb_deployment_record("dep-vdb-2", label="Env VDB Deployment")
    unpaginate_calls = {"count": 0}

    def unpaginate_side_effect(**kwargs: object) -> Iterator[dict[str, object]]:
        unpaginate_calls["count"] += 1
        if unpaginate_calls["count"] == 1:
            return iter([])
        return iter([new_deployment])

    with (
        patch.object(VectorDatabase, "get", return_value=mock_vdb),
        patch.object(dr.client, "get_client", return_value=mock_rest_client),
        patch.object(
            dr.utils.pagination,
            "unpaginate",
            side_effect=unpaginate_side_effect,
        ),
    ):
        result = await tools.vdb_deploy(
            vector_database_id="vdb-123",
            prediction_environment_id="env-1",
        )
        assert result["deployment_id"] == "dep-vdb-2"
        mock_rest_client.post.assert_called_once_with(
            "genai/vectorDatabases/vdb-123/deployments/",
            data={"prediction_environment_id": "env-1"},
        )


@pytest.mark.asyncio
async def test_vdb_deploy_missing_vector_database_id() -> None:
    with pytest.raises(ToolError, match="vector_database_id must be provided"):
        await tools.vdb_deploy()


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_deploy_no_prediction_servers() -> None:
    mock_vdb = MagicMock()
    mock_vdb.execution_status = "COMPLETED"
    with (
        patch.object(dr.PredictionServer, "list", return_value=[]),
        patch.object(VectorDatabase, "get", return_value=mock_vdb),
    ):
        with pytest.raises(ToolError, match="No prediction servers available"):
            await tools.vdb_deploy(vector_database_id="vdb-123")


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_deploy_client_error() -> None:
    mock_server = SimpleNamespace(id="srv1")
    mock_vdb = MagicMock()
    mock_vdb.id = "vdb-123"
    mock_vdb.execution_status = "COMPLETED"
    mock_rest_client = MagicMock()
    mock_rest_client.post.side_effect = ClientError(
        "404 client error: {'message': 'Vector database not found'}",
        status_code=404,
        json={"message": "Vector database not found"},
    )
    with (
        patch.object(dr.PredictionServer, "list", return_value=[mock_server]),
        patch.object(VectorDatabase, "get", return_value=mock_vdb),
        patch.object(dr.client, "get_client", return_value=mock_rest_client),
        patch.object(dr.utils.pagination, "unpaginate", return_value=iter([])),
    ):
        with pytest.raises(ToolError) as exc_info:
            await tools.vdb_deploy(vector_database_id="vdb-123")
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND
    assert "404" in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_deploy_poll_timeout() -> None:
    mock_server = SimpleNamespace(id="srv1")
    mock_vdb = MagicMock()
    mock_vdb.id = "vdb-123"
    mock_vdb.execution_status = "COMPLETED"
    mock_rest_client = MagicMock()
    mock_rest_client.post.return_value = MagicMock(
        headers={"Location": "https://example.com/async/status/3/"}
    )
    with (
        patch.object(dr.PredictionServer, "list", return_value=[mock_server]),
        patch.object(VectorDatabase, "get", return_value=mock_vdb),
        patch.object(dr.client, "get_client", return_value=mock_rest_client),
        patch.object(dr.utils.pagination, "unpaginate", return_value=iter([])),
        patch.object(tools, "VDB_DEPLOY_POLL_INTERVAL_SECONDS", 0.0),
        patch.object(tools, "VDB_DEPLOY_POLL_TIMEOUT_SECONDS", 0.0),
    ):
        with pytest.raises(ToolError, match="no new deployment record appeared"):
            await tools.vdb_deploy(vector_database_id="vdb-123")


@pytest.mark.asyncio
async def test_vdb_get_malformed_vector_database_id() -> None:
    with pytest.raises(ToolError, match="Vector database bad-id-### not found") as exc_info:
        await tools.vdb_get(vector_database_id="bad-id-###", target_status="completed")
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_vdb_get_redirect_raises_clean_error() -> None:
    html = "<!DOCTYPE html><html><body>app shell</body></html>"
    mock_rest_client = MagicMock()
    mock_rest_client.get.side_effect = ClientError(
        f"307 client error: {html}",
        status_code=307,
        json={},
    )
    with patch.object(dr.client, "get_client", return_value=mock_rest_client):
        with pytest.raises(ToolError, match="redirected unexpectedly") as exc_info:
            await tools.vdb_get(vector_database_id=VDB_ID)
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND
    assert "app shell" not in str(exc_info.value)
