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
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
from datarobot.errors import ClientError
from datarobot.fs import DataRobotFileSystem

from datarobot_genai.drmcp.test_utils.clients.dr_gateway import DRLLMGatewayMCPClient
from datarobot_genai.drmcp.test_utils.mcp_utils_ete import get_dr_llm_gateway_client_config
from tests.drmcp.helpers.otel_entity import OTEL_ACCEPTANCE_ENTITY_TYPE
from tests.drmcp.helpers.otel_entity import USE_CASE_NAME_PREFIX
from tests.drmcp.helpers.otel_entity import OtelAcceptanceEntity
from tests.drmcp.helpers.otel_entity import cleanup_use_case
from tests.drmcp.helpers.otel_entity import emit_failing_agent_run
from tests.drmcp.helpers.otel_entity import list_error_span_ids
from tests.drmcp.helpers.otel_entity import list_error_trace_ids
from tests.drmcp.helpers.otel_entity import otel_ingest_base_url
from tests.drmcp.helpers.otel_entity import otel_ingest_headers
from tests.drmcp.helpers.otel_entity import provision_use_case
from tests.drmcp.helpers.otel_entity import wait_for_otel_ingestion

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


def _external_otel_entity(dr_client: Any, entity_id: str) -> OtelAcceptanceEntity:
    """Resolve ``TEST_OTEL_ENTITY_ID`` (+ optional type/trace/span overrides) to an entity.

    The opt-in path for running the cases against a *real*, externally-instrumented
    entity -- e.g. plan §9 step 9's reference deployment, whose median trace is 807k
    tokens -- rather than the small provisioned one. ``TEST_OTEL_FAILING_TRACE_ID`` /
    ``TEST_OTEL_FAILING_SPAN_ID`` override discovery; otherwise the newest error-status
    trace and its first ERROR span are looked up directly against the REST API (bypassing
    the LLM entirely) -- tier 3 tests whether the model picks the right *tool and
    parameters* given a real, live failure, not whether it can also locate one blind.
    """
    entity_type = os.environ.get("TEST_OTEL_ENTITY_TYPE", "deployment")
    rest_client = dr_client.client.get_client()
    trace_id = os.environ.get("TEST_OTEL_FAILING_TRACE_ID")
    if not trace_id:
        try:
            trace_ids = list_error_trace_ids(rest_client, entity_type, entity_id)
        except Exception as exc:
            pytest.skip(f"Could not discover a failing OTel trace for acceptance tests: {exc}")
        if not trace_ids:
            pytest.skip(
                f"No error-status OTel traces found for {entity_type}/{entity_id}; set "
                "TEST_OTEL_FAILING_TRACE_ID to a known failing trace_id to run this case."
            )
        trace_id = trace_ids[0]
    span_id = os.environ.get("TEST_OTEL_FAILING_SPAN_ID")
    if not span_id:
        try:
            span_ids = list_error_span_ids(rest_client, entity_type, entity_id, trace_id)
        except Exception as exc:
            pytest.skip(f"Could not discover a failing OTel span for acceptance tests: {exc}")
        if not span_ids:
            pytest.skip(
                f"Trace {trace_id} has no ERROR-status span in the first 100; set "
                "TEST_OTEL_FAILING_SPAN_ID to run this case."
            )
        span_id = span_ids[0]
    return OtelAcceptanceEntity(
        entity_type=entity_type,
        entity_id=entity_id,
        failing_trace_id=trace_id,
        failing_span_id=span_id,
    )


@pytest.fixture(scope="session")
def otel_acceptance_entity(request: pytest.FixtureRequest, dr_client: Any) -> OtelAcceptanceEntity:
    """Provision (or resolve) the entity the OTel acceptance cases run against.

    With no configuration this creates a Use Case (OTel entity type
    ``experiment_container``), OTLP-exports a small failing agentic run into it -- one
    error-status trace whose ``llm.chat`` span errored with a 429 and carries LLM output,
    one healthy trace, and an error-level log line correlated to the failing span -- waits
    until the REST API can read all of it back, and deletes everything at session end.
    That gives the three tier-3 cases a reproducible entity with exactly the data they
    assume, instead of depending on whatever real entity someone happens to have.

    Overrides:

    * ``TEST_OTEL_ENTITY_ID`` (+ ``TEST_OTEL_ENTITY_TYPE``, default ``deployment``) --
      skip provisioning and run against that entity instead; see
      :func:`_external_otel_entity`.
    * ``TEST_OTEL_KEEP_ENTITY`` -- leave the provisioned Use Case and its telemetry in
      place for inspection (``dr xp --entity-id <id>``) instead of deleting them.
    """
    override = os.environ.get("TEST_OTEL_ENTITY_ID")
    if override:
        return _external_otel_entity(dr_client, override)

    rest_client = dr_client.client.get_client()
    run_label = uuid.uuid4().hex[:8]
    name = (
        f"{USE_CASE_NAME_PREFIX} {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())} {run_label}"
    )
    try:
        use_case_id = provision_use_case(
            dr_client,
            name=name,
            description=(
                "Created by datarobot-genai's OTel MCP acceptance tests "
                "(tests/drmcp/acceptance/test_otel_tools.py) as an OTel telemetry target. "
                "Deleted automatically at the end of the run; safe to delete if left behind."
            ),
        )
    except Exception as exc:
        pytest.skip(f"Could not create a Use Case for the OTel acceptance tests: {exc}")

    def _teardown() -> None:
        for warning in cleanup_use_case(dr_client, rest_client, use_case_id):
            print(f"Warning: {warning}")

    if os.environ.get("TEST_OTEL_KEEP_ENTITY"):
        print(
            f"TEST_OTEL_KEEP_ENTITY set: keeping use case {use_case_id} "
            f"(dr xp --entity-id {use_case_id} --enable-logs)"
        )
    else:
        request.addfinalizer(_teardown)

    ingest_base_url = otel_ingest_base_url(rest_client.endpoint)
    try:
        emitted = emit_failing_agent_run(
            ingest_base_url=ingest_base_url,
            headers=otel_ingest_headers(
                OTEL_ACCEPTANCE_ENTITY_TYPE,
                use_case_id,
                rest_client.token or os.environ.get("DATAROBOT_API_TOKEN", ""),
            ),
            run_label=run_label,
        )
    except RuntimeError as exc:
        pytest.fail(f"OTLP export to {ingest_base_url} for use case {use_case_id} failed: {exc}")
    try:
        elapsed = wait_for_otel_ingestion(
            rest_client, OTEL_ACCEPTANCE_ENTITY_TYPE, use_case_id, emitted
        )
    except ClientError as exc:
        if exc.status_code == 403:
            pytest.skip(
                f"Cannot read OTel data for {OTEL_ACCEPTANCE_ENTITY_TYPE}/{use_case_id} "
                f"(403 -- usually the GENAI_EXPERIMENTATION flag or the "
                f"AGENTIC_PREDICTIVE_GOVERNANCE_BUILDER seat license): {exc}"
            )
        raise
    except TimeoutError as exc:
        pytest.fail(str(exc))
    print(
        f"Provisioned OTel acceptance entity {OTEL_ACCEPTANCE_ENTITY_TYPE}/{use_case_id}: "
        f"failing trace {emitted.failing_trace_id} span {emitted.failing_span_id} "
        f"(readable {elapsed:.0f}s after export)"
    )
    return OtelAcceptanceEntity(
        entity_type=OTEL_ACCEPTANCE_ENTITY_TYPE,
        entity_id=use_case_id,
        failing_trace_id=emitted.failing_trace_id,
        failing_span_id=emitted.failing_span_id,
    )


@pytest.fixture(scope="session")
def otel_entity_type(otel_acceptance_entity: OtelAcceptanceEntity) -> str:
    """Entity type for OTel acceptance tests (``experiment_container`` when provisioned)."""
    return otel_acceptance_entity.entity_type


@pytest.fixture(scope="session")
def otel_entity_id(otel_acceptance_entity: OtelAcceptanceEntity) -> str:
    """Entity id for OTel acceptance tests (the provisioned Use Case id by default)."""
    return otel_acceptance_entity.entity_id


@pytest.fixture(scope="session")
def otel_failing_trace_id(otel_acceptance_entity: OtelAcceptanceEntity) -> str:
    """trace_id of an error-status trace on the OTel acceptance entity."""
    return otel_acceptance_entity.failing_trace_id


@pytest.fixture(scope="session")
def otel_failing_span_id(otel_acceptance_entity: OtelAcceptanceEntity) -> str:
    """span_id of an ERROR-status span within ``otel_failing_trace_id``."""
    return otel_acceptance_entity.failing_span_id


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
