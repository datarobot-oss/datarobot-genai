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

import pytest
from datarobot.fs import DataRobotFileSystem

from datarobot_genai.drmcp.test_utils.clients.dr_gateway import DRLLMGatewayMCPClient
from datarobot_genai.drmcp.test_utils.mcp_utils_ete import get_dr_llm_gateway_client_config

# Acceptance tests require real DataRobot credentials from .env (mcp_utils_ete loads it on import).
os.environ["MCP_USE_CLIENT_STUBS"] = "false"


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


@pytest.fixture(scope="session")
def workload_id(dr_client: Any) -> str:
    """Workload ID for acceptance tests (``TEST_WORKLOAD_ID`` env or first from API)."""
    override = os.environ.get("TEST_WORKLOAD_ID")
    if override:
        return override
    try:
        result = dr_client.client.get_client().get("workloads/", params={"limit": 1}).json()
        workloads = result.get("data") or []
        if not workloads:
            pytest.skip("No workloads available for acceptance tests")
        return str(workloads[0]["id"])
    except Exception as exc:
        pytest.skip(f"Could not list workloads for acceptance tests: {exc}")


@pytest.fixture(scope="session")
def nonexistent_workload_id() -> str:
    # Workload API validates MongoDB ObjectId shape (24 hex chars); bad format → 422.
    return "000000000000000000000001"


_FILES_API_TEST_FILENAME = "acceptance-test.txt"
_FILES_API_TEST_CONTENT = b"mcp files api acceptance test\n"


@pytest.fixture(scope="session")
def files_api_test_file(dr_client: Any) -> dict[str, str]:
    """Create a small catalog file for Files API acceptance tests."""
    del dr_client  # ensure DataRobot client is configured for the session
    fs = DataRobotFileSystem()
    try:
        catalog_id = fs.create_catalog_item_dir()
        file_path = f"dr://{catalog_id}/{_FILES_API_TEST_FILENAME}"
        fs.pipe_file(file_path, value=_FILES_API_TEST_CONTENT, mode="create")
    except Exception as exc:
        pytest.skip(f"Could not provision Files API test file: {exc}")
    return {"catalog_id": catalog_id, "file_path": file_path}


@pytest.fixture(scope="session")
def files_catalog_id(files_api_test_file: dict[str, str]) -> str:
    """Catalog id for Files API acceptance tests."""
    return files_api_test_file["catalog_id"]


@pytest.fixture(scope="session")
def files_file_path(files_api_test_file: dict[str, str]) -> str:
    """``dr://`` path to a file for Files API acceptance tests."""
    return files_api_test_file["file_path"]


@pytest.fixture(scope="session")
def nonexistent_files_path() -> str:
    """Filesystem path that should not exist (valid catalog id shape, missing file)."""
    return "dr://000000000000000000000001/nonexistent.txt"
