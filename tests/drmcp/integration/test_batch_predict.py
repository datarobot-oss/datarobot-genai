# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for batch catalog / project prediction tools (stub DataRobot client)."""

import json
from io import StringIO
from typing import Any

import pandas as pd
import pytest
from mcp.types import TextContent

from datarobot_genai.drmcp.test_utils.mcp_utils_integration import integration_test_mcp_session
from datarobot_genai.drmcp.test_utils.stubs.dr_client_stubs import STUB_BATCH_PREDICTION_JOB_ID
from datarobot_genai.drmcp.test_utils.stubs.dr_client_stubs import STUB_PREDICT_CATALOG_DATASET_ID
from datarobot_genai.drmcp.test_utils.stubs.dr_client_stubs import STUB_PROJECT_ID


@pytest.mark.asyncio
class TestBatchPredictToolsIntegration:
    """MCP integration tests for predict_by_ai_catalog, status, and results (stdio stub server)."""

    async def test_batch_predict_tools_registered(self) -> None:
        """Batch prediction tools are registered with predictive tools enabled."""
        async with integration_test_mcp_session() as session:
            listed = await session.list_tools()
            tool_names = [t.name for t in listed.tools]
            assert "predict_by_ai_catalog" in tool_names
            assert "predict_from_project_data" in tool_names
            assert "get_batch_prediction_job_status" in tool_names
            assert "get_batch_prediction_results" in tool_names

    async def test_predict_by_ai_catalog_submit_status_and_results(
        self, classification_project: dict[str, Any]
    ) -> None:
        """Submit batch scoring from catalog, poll status, and read inline CSV (stub job)."""
        deployment_id = classification_project["deployment_id"]
        async with integration_test_mcp_session() as session:
            submit = await session.call_tool(
                "predict_by_ai_catalog",
                {
                    "deployment_id": deployment_id,
                    "dataset_id": STUB_PREDICT_CATALOG_DATASET_ID,
                },
            )
            assert not submit.isError
            assert isinstance(submit.content[0], TextContent)
            payload = json.loads(submit.content[0].text)
            assert payload["job_id"] == STUB_BATCH_PREDICTION_JOB_ID
            assert payload["deployment_id"] == deployment_id
            assert payload["batch_job_status"] == "COMPLETED"
            assert payload.get("url")
            assert "get_batch_prediction_job_status" in payload["note"]

            status = await session.call_tool(
                "get_batch_prediction_job_status",
                {"job_id": STUB_BATCH_PREDICTION_JOB_ID},
            )
            assert not status.isError
            st = json.loads(status.content[0].text)
            assert st["job_id"] == STUB_BATCH_PREDICTION_JOB_ID
            assert st["batch_job_status"] == "COMPLETED"
            assert st.get("url")

            results = await session.call_tool(
                "get_batch_prediction_results",
                {"job_id": STUB_BATCH_PREDICTION_JOB_ID},
            )
            assert not results.isError
            body = json.loads(results.content[0].text)
            assert body["job_id"] == STUB_BATCH_PREDICTION_JOB_ID
            assert body["mime_type"] == "text/csv"
            assert body["size_bytes"] > 0
            assert "sentiment_PREDICTION" in body["data"]
            df = pd.read_csv(StringIO(body["data"]))
            assert len(df) >= 1

    async def test_predict_from_project_data_submit_status_and_results(
        self, classification_project: dict[str, Any]
    ) -> None:
        """Submit batch scoring from project holdout, poll, and download stub CSV."""
        deployment_id = classification_project["deployment_id"]
        async with integration_test_mcp_session() as session:
            submit = await session.call_tool(
                "predict_from_project_data",
                {
                    "deployment_id": deployment_id,
                    "project_id": STUB_PROJECT_ID,
                    "partition": "holdout",
                },
            )
            assert not submit.isError
            payload = json.loads(submit.content[0].text)
            assert payload["job_id"] == STUB_BATCH_PREDICTION_JOB_ID

            status = await session.call_tool(
                "get_batch_prediction_job_status",
                {"job_id": payload["job_id"]},
            )
            assert not status.isError

            results = await session.call_tool(
                "get_batch_prediction_results",
                {"job_id": payload["job_id"]},
            )
            assert not results.isError
            body = json.loads(results.content[0].text)
            assert "data" in body

    async def test_predict_by_ai_catalog_empty_deployment_id_validation(self) -> None:
        """Empty deployment_id returns a tool validation error."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "predict_by_ai_catalog",
                {
                    "deployment_id": "   ",
                    "dataset_id": STUB_PREDICT_CATALOG_DATASET_ID,
                },
            )
            assert result.isError
            text = result.content[0].text  # type: ignore[union-attr]
            assert "deployment_id" in text
