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
import copy
from unittest.mock import MagicMock

import pytest
import responses

from datarobot_genai.drmcp.core.clients import get_sdk_client
from datarobot_genai.drmcp.core.credentials import get_credentials
from datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata import get_mcp_tool_metadata
from datarobot_genai.drmcp.core.dynamic_tools.deployment.register import (
    register_tools_of_datarobot_deployments,
)
from datarobot_genai.drmcp.core.mcp_instance import TaggedFastMCP
from datarobot_genai.drmcp.core.mcp_instance import mcp


@pytest.fixture
def deployment_id_ok() -> str:
    return "68caa8e45efd6ac465a15a22"


@pytest.fixture
def deployment_id_error() -> str:
    return "68b1bb20b0409ae95c23e2e8"


@pytest.fixture
def deployment_api_response_ok(deployment_id_ok) -> dict:
    return {
        "id": deployment_id_ok,
        "label": "dynamic tool ok",
        "description": "Description of a valid dynamic tool deployment.",
        "createdAt": "2025-09-17T12:26:26.105000Z",
        "model": {
            "id": "68cabf4fa8a449edd6c17cbc",
            "type": "dynamic tool ok",
            "targetName": None,
            "targetType": "Unstructured",
            "projectId": None,
            "projectName": None,
            "unsupervisedMode": False,
            "unstructuredModelKind": True,
            "buildEnvironmentType": "Python",
            "deployedAt": "2025-09-17T14:04:22.910000Z",
            "customModelImage": {
                "customModelId": "686285c7c3bcf256c31ea3f1",
                "customModelName": "list-files",
                "customModelVersionId": "68cabf4fa8a449edd6c17cbc",
                "customModelVersionLabel": "v2.0",
                "executionEnvironmentId": "680fe4949604e9eba46b1775",
                "executionEnvironmentName": "[DataRobot] Python 3.11 GenAI Agents",
                "executionEnvironmentVersionId": "6859c32445c0db87156d4563",
                "executionEnvironmentVersionLabel": "v5",
            },
            "isDeprecated": False,
            "prompt": None,
        },
        "status": "active",
        "modelPackage": {
            "id": "68cabf8062993fde739bf228",
            "name": "list-files (draft v2.0)",
            "registeredModelId": "6862895418f6e4eb5b6a1ec3",
        },
        "defaultPredictionServer": {
            "id": "prediction-server-id",
            "url": "https://prediction-server.example.com",
            "datarobot-key": "test-datarobot-key-123",
        },
        "predictionEnvironment": {
            "id": "pred-env-id",
            "name": "Test Prediction Environment",
            "platform": "aws",
        },
        "tags": [{"id": "68caadd2b456b4a50bfc5c64", "name": "tool", "value": "tool"}],
    }


@pytest.fixture
def deployment_api_response_error(deployment_api_response_ok, deployment_id_error) -> dict:
    deployment_response = copy.deepcopy(deployment_api_response_ok)
    deployment_response["id"] = deployment_id_error
    deployment_response["label"] = "dynamic tool error"
    deployment_response["description"] = "An invalid tool deployment, without input schema."
    return deployment_response


@pytest.fixture
def deployment_info_api_response_ok() -> dict:
    return {
        "codeDir": "/opt/code",
        "drumServer": "flask",
        "drumVersion": "1.17.2",
        "language": "python",
        "modelMetadata": {
            "inputSchema": {
                "properties": {
                    "json": {
                        "properties": {
                            "bar": {
                                "description": "Description of the object.",
                                "title": "Bar parameter",
                                "type": "string",
                            },
                            "foo": {
                                "description": (
                                    "The foo parameter controls how many Foo instances are needed."
                                ),
                                "title": "Foo parameter",
                                "type": "integer",
                            },
                            "otherParam": {
                                "default": "",
                                "description": "Yet another parameter with description.",
                                "title": "Other parameter",
                                "type": "string",
                            },
                        },
                        "required": ["foo", "bar"],
                        "type": "object",
                    },
                },
                "type": "object",
            },
            "name": "AgentTool Example",
            "targetType": "unstructured",
            "type": "inference",
        },
        "predictor": None,
        "targetType": "unstructured",
    }


@pytest.fixture
def deployment_info_api_response_error(deployment_info_api_response_ok) -> dict:
    """Deployment info response without input schema."""
    info_response = copy.deepcopy(deployment_info_api_response_ok)
    del info_response["modelMetadata"]["inputSchema"]
    return info_response


@pytest.fixture
def version_api_response() -> dict:
    return {
        "major": 2,
        "minor": 38,
        "versionString": "2.38.0",
        "releasedVersion": "2.37.0",
    }


@pytest.fixture
def expected_input_schema() -> dict:
    return {
        "properties": {
            "json": {
                "properties": {
                    "bar": {
                        "description": "Description of the object.",
                        "title": "Bar parameter",
                        "type": "string",
                    },
                    "foo": {
                        "description": "The foo parameter controls how many "
                        "Foo instances are needed.",
                        "title": "Foo parameter",
                        "type": "integer",
                    },
                    "other_param": {
                        "default": "",
                        "description": "Yet another parameter with description.",
                        "title": "Other parameter",
                        "type": "string",
                    },
                },
                "required": ["foo", "bar"],
                "type": "object",
            },
        },
        "type": "object",
    }


@pytest.fixture
def datarobot_endpoint() -> str:
    return get_credentials().datarobot.endpoint


@pytest.fixture
def mock_api_responses(
    datarobot_endpoint: str,
    deployment_id_ok: str,
    deployment_id_error: str,
    version_api_response: dict,
    deployment_api_response_ok: dict,
    deployment_api_response_error: dict,
    deployment_info_api_response_ok: dict,
    deployment_info_api_response_error: dict,
):
    """Set up all API endpoint mocks."""
    responses.add(
        responses.GET,
        f"{datarobot_endpoint}/version/",
        json=version_api_response,
    )
    responses.add(
        responses.GET,
        f"{datarobot_endpoint}/deployments/{deployment_id_ok}/",
        json=deployment_api_response_ok,
    )
    responses.add(
        responses.GET,
        f"{datarobot_endpoint}/deployments/{deployment_id_error}/",
        json=deployment_api_response_error,
    )
    responses.add(
        responses.GET,
        f"{datarobot_endpoint}/deployments/{deployment_id_ok}/directAccess/info/",
        json=deployment_info_api_response_ok,
    )
    responses.add(
        responses.GET,
        f"{datarobot_endpoint}/deployments/{deployment_id_error}/directAccess/info/",
        json=deployment_info_api_response_error,
    )
    responses.add(
        responses.GET,
        f"{datarobot_endpoint}/deployments/?tagValues=tool&tagKeys=tool",
        json={
            "count": 2,
            "next": None,
            "previous": None,
            "data": [deployment_api_response_ok, deployment_api_response_error],
            "totalCount": 2,
        },
    )


@pytest.fixture
def sdk_client():
    """Get configured DataRobot SDK client."""
    return get_sdk_client()


@pytest.fixture
def deployment_ok(sdk_client, deployment_id_ok: str):
    """Get deployment object."""
    return sdk_client.Deployment.get(deployment_id_ok)


# todo - change the hardcoded deployment id (deployment_id_ok) to a proper fixture that
# creates a deployment in the test environment
@pytest.mark.skip(reason="Skipping test_get_mcp_tool_metadata until the deployment id is changed")
@responses.activate
async def test_get_mcp_tool_metadata(
    mock_api_responses,
    sdk_client,
    deployment_ok,
    expected_input_schema: dict,
) -> None:
    """Test that get_mcp_tool_metadata returns correct input schema."""
    metadata = get_mcp_tool_metadata(deployment_ok)

    actual_input_schema = metadata.get("input_schema", {})
    assert actual_input_schema == expected_input_schema


@pytest.mark.asyncio
@responses.activate
async def test_dynamic_tool_registration(sdk_client, mock_api_responses) -> None:
    await register_tools_of_datarobot_deployments()

    tool_names = {tool.name for tool in await mcp.list_tools()}

    assert "dynamic_tool_ok" in tool_names, "`dynamic_tool_ok` is missing."
    assert "dynamic_tool_error" not in tool_names, "`dynamic_tool_error` should not be registered."


@pytest.mark.asyncio
async def test_mcp_mapping_methods():
    mcp = TaggedFastMCP()

    # Mock the remove_tool method to avoid actual tool removal,
    # as the actual tool was not registered. This is just to test the
    # mapping methods, and whether remove_tool is called correctly.
    mcp.remove_tool = MagicMock()

    await mcp.set_deployment_mapping("id1", "tool_name_1")
    await mcp.set_deployment_mapping("id2", "tool_name_2")

    deployments = await mcp.get_deployment_mapping()
    assert deployments == {"id1": "tool_name_1", "id2": "tool_name_2"}

    # override existing mapping
    await mcp.set_deployment_mapping("id2", "tool_name_3")
    deployments = await mcp.get_deployment_mapping()
    assert deployments == {"id1": "tool_name_1", "id2": "tool_name_3"}

    # Verify remove_tool was called when overriding
    assert mcp.remove_tool.call_count == 1
    mcp.remove_tool.assert_called_with("tool_name_2")

    # delete first mapping
    await mcp.remove_deployment_mapping("id1")
    deployments = await mcp.get_deployment_mapping()
    assert deployments == {"id2": "tool_name_3"}

    # Verify remove_tool was called for id1
    assert mcp.remove_tool.call_count == 2
    assert mcp.remove_tool.call_args_list[-1][0][0] == "tool_name_1"

    # delete second mapping
    await mcp.remove_deployment_mapping("id2")
    deployments = await mcp.get_deployment_mapping()
    assert deployments == {}

    # Verify remove_tool was called for id2
    assert mcp.remove_tool.call_count == 3
    assert mcp.remove_tool.call_args_list[-1][0][0] == "tool_name_3"
