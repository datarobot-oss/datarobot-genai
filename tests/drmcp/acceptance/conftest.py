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

from pathlib import Path
from typing import Any

import pytest

from datarobot_genai.drmcp.test_utils.clients.dr_gateway import DRLLMGatewayMCPClient
from datarobot_genai.drmcp.test_utils.mcp_utils_ete import get_dr_llm_gateway_client_config


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
def dataset_url() -> str:
    return "https://s3.amazonaws.com/datarobot_public_datasets/10k_diabetes_scoring_small.csv"


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
