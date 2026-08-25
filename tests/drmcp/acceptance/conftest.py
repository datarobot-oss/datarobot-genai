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


@pytest.fixture(scope="session")
def otel_entity_type() -> str:
    """Entity type for OTel acceptance tests (``TEST_OTEL_ENTITY_TYPE`` env, default deployment)."""
    return os.environ.get("TEST_OTEL_ENTITY_TYPE", "deployment")


@pytest.fixture(scope="session")
def otel_entity_id() -> str:
    """Entity id for OTel acceptance tests, from the required ``TEST_OTEL_ENTITY_ID`` env var.

    Unlike ``workload_id``, there is no generic "pick any entity" fallback here: these cases
    need a *specific* entity known to carry real OTel data (and, for the failure-diagnosis
    cases, at least one real error-status trace) -- an entity with zero OTel data would fail
    these tests for a data reason, not a tool-description reason. See plan §9 step 9's own
    reference deployment ("[agent-application-dev] [agent]", whose median trace is 807k
    tokens) for the kind of entity this should point at.
    """
    value = os.environ.get("TEST_OTEL_ENTITY_ID")
    if not value:
        pytest.skip(
            "TEST_OTEL_ENTITY_ID is not set; OTel acceptance cases need a specific entity "
            "known to carry real OTel traces/logs to be meaningful."
        )
    return value


@pytest.fixture(scope="session")
def otel_failing_trace_id(dr_client: Any, otel_entity_type: str, otel_entity_id: str) -> str:
    """Discover a trace_id with an error span on the configured OTel entity.

    ``TEST_OTEL_FAILING_TRACE_ID`` overrides discovery. Otherwise this calls
    ``GET otel/{entity_type}/{entity_id}/traces/?status=error&limit=1`` directly against the
    DataRobot REST API (bypassing the LLM entirely), the same way ``workload_id`` discovers a
    workload id -- tier 3 tests whether the model picks the right *tool and parameters* given a
    real, live failure, not whether it can also locate one blind with no test-side help.
    """
    override = os.environ.get("TEST_OTEL_FAILING_TRACE_ID")
    if override:
        return override
    try:
        result = (
            dr_client.client.get_client()
            .get(
                f"otel/{otel_entity_type}/{otel_entity_id}/traces/",
                params={"status": "error", "limit": 1},
            )
            .json()
        )
        traces = result.get("traces") or result.get("data") or []
        if not traces:
            pytest.skip(
                f"No error-status OTel traces found for {otel_entity_type}/{otel_entity_id}; "
                "set TEST_OTEL_FAILING_TRACE_ID to a known failing trace_id to run this case."
            )
        return str(traces[0]["trace_id"])
    except Exception as exc:
        pytest.skip(f"Could not discover a failing OTel trace for acceptance tests: {exc}")


@pytest.fixture(scope="session")
def otel_failing_span_id(
    dr_client: Any,
    otel_entity_type: str,
    otel_entity_id: str,
    otel_failing_trace_id: str,
) -> str:
    """Discover the span_id of an ERROR-status span within ``otel_failing_trace_id``.

    ``TEST_OTEL_FAILING_SPAN_ID`` overrides discovery. Otherwise this fetches the trace
    directly (same bypass-the-LLM rationale as ``otel_failing_trace_id``) and picks the first
    span whose ``status_code`` is ``ERROR``.
    """
    override = os.environ.get("TEST_OTEL_FAILING_SPAN_ID")
    if override:
        return override
    try:
        result = (
            dr_client.client.get_client()
            .get(
                f"otel/{otel_entity_type}/{otel_entity_id}/traces/{otel_failing_trace_id}/",
                params={"limit": 100},
            )
            .json()
        )
        spans = result.get("spans") or []
        error_spans = [s for s in spans if s.get("status_code") == "ERROR"]
        if not error_spans:
            pytest.skip(
                f"Trace {otel_failing_trace_id} has no ERROR-status span in the first 100; "
                "set TEST_OTEL_FAILING_SPAN_ID to run this case."
            )
        return str(error_spans[0]["span_id"])
    except Exception as exc:
        pytest.skip(f"Could not discover a failing OTel span for acceptance tests: {exc}")


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
