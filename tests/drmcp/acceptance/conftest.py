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

import os
from pathlib import Path
from typing import Any

import datarobot as dr
import pytest
from datarobot.context import Context as DRContext

from datarobot_genai.drmcp.test_utils.clients.dr_gateway import DRLLMGatewayMCPClient
from datarobot_genai.drmcp.test_utils.mcp_utils_ete import get_dr_llm_gateway_client_config
from datarobot_genai.drtools.core.credentials import get_credentials
from tests.drmcp.stub_credentials import STUB_DATAROBOT_API_TOKEN


@pytest.fixture(scope="session")
def llm_client() -> DRLLMGatewayMCPClient:
    """Create DataRobot LLM Gateway MCP client for the test session."""
    try:
        config = get_dr_llm_gateway_client_config()
        return DRLLMGatewayMCPClient(str(config))
    except ValueError as e:
        raise ValueError(f"Missing required DataRobot environment variables: {e}") from e
    except Exception as e:
        raise ConnectionError(f"Failed to create LLM MCP client: {str(e)}") from e


@pytest.fixture(scope="session")
def diabetes_scoring_small_file_path(test_data_dir: Path) -> Path:
    return test_data_dir / "10k_diabetes_scoring_small.csv"


@pytest.fixture(scope="session")
def nonexistent_file_path() -> str:
    return "nonexistent_file_path"


@pytest.fixture(scope="session")
def deployment_id(classification_project: dict[str, Any]) -> str:
    value = classification_project.get("deployment_id")
    assert isinstance(value, str)
    return value


@pytest.fixture(scope="session")
def nonexistent_deployment_id() -> str:
    return "nonexistent_deployment_id"


@pytest.fixture(scope="session")
def classification_project_id(classification_project: dict[str, Any]) -> str:
    project = classification_project.get("project")
    value = getattr(project, "id", None)
    assert isinstance(value, str)
    return value


@pytest.fixture(scope="session")
def nonexistent_project_id() -> str:
    return "nonexistent_project_id"


@pytest.fixture(scope="session")
def model_id(classification_project: dict[str, Any]) -> str:
    model = classification_project.get("model")
    value = getattr(model, "id", None)
    assert isinstance(value, str)
    return value


@pytest.fixture(scope="session")
def nonexistent_model_id() -> str:
    return "nonexistent_model_id"


@pytest.fixture(scope="session")
def classification_dataset_id(classification_project: dict[str, Any]) -> str:
    value = classification_project.get("source_dataset_id")
    assert isinstance(value, str)
    return value


@pytest.fixture(scope="session")
def nonexistent_dataset_name() -> str:
    return "nonexistent_dataset_name"


def _is_vector_database_deployment(deployment: dict[str, Any]) -> bool:
    capabilities = deployment.get("capabilities") or {}
    if capabilities.get("supportsVectorDatabaseQuerying"):
        return True
    model = deployment.get("model") or {}
    target_type = model.get("targetType") or model.get("target_type")
    return target_type in ("VectorDatabase", "vector_database")


@pytest.fixture(scope="session")
def vdb_deployment_id() -> str:
    """Vector Database deployment id for acceptance tests (env override or first in account)."""
    if env_id := os.environ.get("VDB_DEPLOYMENT_ID_ETE"):
        return env_id

    creds = get_credentials()
    token = creds.datarobot.application_api_token
    if not token or token == STUB_DATAROBOT_API_TOKEN:
        pytest.skip("DATAROBOT_API_TOKEN required for VDB acceptance tests")

    dr.Client(token=token, endpoint=creds.datarobot.endpoint)
    DRContext.use_case = None
    client = dr.client.get_client()
    response = client.get("deployments/", params={"limit": 100})
    deployments = response.json().get("data") or []
    for deployment in deployments:
        if _is_vector_database_deployment(deployment):
            return str(deployment["id"])
    pytest.skip("No Vector Database deployment available; set VDB_DEPLOYMENT_ID_ETE")
