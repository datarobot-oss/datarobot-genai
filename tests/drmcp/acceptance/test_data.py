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

import base64
import inspect
from typing import Any

import pytest

from datarobot_genai.drmcp.test_utils.mcp_utils_ete import ete_test_mcp_session
from datarobot_genai.drmcp.test_utils.tool_base_ete import ETETestExpectations
from datarobot_genai.drmcp.test_utils.tool_base_ete import ToolBaseE2E
from datarobot_genai.drmcp.test_utils.tool_base_ete import ToolCallTestExpectations

# Tiny CSV for base64 upload E2E: model must pass this exact base64 string to the tool.
_TINY_UPLOAD_CSV = "col_a,col_b\n1,2\n"
TINY_UPLOAD_CSV_BASE64 = base64.b64encode(_TINY_UPLOAD_CSV.encode("utf-8")).decode("ascii")


@pytest.fixture(scope="session")
def expectations_for_upload_dataset_to_ai_catalog_success_from_base64() -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="upload_dataset_to_ai_catalog",
                parameters={
                    "file_content_base64": TINY_UPLOAD_CSV_BASE64,
                    "dataset_filename": "ete_tiny.csv",
                },
                result={"dataset_id": "", "dataset_version_id": "", "dataset_name": ""},
            ),
        ],
        llm_response_content_contains_expectations=[
            "dataset has been successfully uploaded",
            "dataset has been uploaded",
            "successfully",
            "uploaded",
            "upload complete",
            "dataset id",
            "done",
            "registered",
            "ete_tiny.csv",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_upload_dataset_to_ai_catalog_success_from_url() -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="upload_dataset_to_ai_catalog",
                parameters={
                    "file_url": "https://s3.amazonaws.com/datarobot_public_datasets/10k_diabetes.csv"
                },
                result={"dataset_id": "", "dataset_version_id": "", "dataset_name": ""},
            ),
        ],
        llm_response_content_contains_expectations=[
            "dataset has been successfully uploaded",
            "dataset has been uploaded",
            "successfully",
            "uploaded",
            "upload complete",
            "dataset id",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_list_ai_catalog_items_success() -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="list_ai_catalog_items",
                parameters={},
                result={"datasets": {}, "count": 0},
            )
        ],
        llm_response_content_contains_expectations=[
            "10k_diabetes_scoring_small.csv",
            "datasets",
            "dataset",
        ],
    )


@pytest.mark.asyncio
class TestDataE2E(ToolBaseE2E):
    """End-to-end tests for data-related functionality."""

    @pytest.mark.parametrize(
        "prompt_template",
        [
            """
        I'm working on a machine learning project and I need to upload a dataset to the
        DataRobot AI Catalog. Can you help me upload the dataset from the URL {file_url}?
        """
        ],
    )
    async def test_upload_dataset_to_ai_catalog_success_from_url(
        self,
        llm_client: Any,
        expectations_for_upload_dataset_to_ai_catalog_success_from_url: ETETestExpectations,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(
            file_url="https://s3.amazonaws.com/datarobot_public_datasets/10k_diabetes.csv"
        )
        async with ete_test_mcp_session() as session:
            await self._run_test_with_expectations(
                prompt,
                expectations_for_upload_dataset_to_ai_catalog_success_from_url,
                llm_client,
                session,
                (
                    inspect.currentframe().f_code.co_name  # type: ignore[union-attr]
                    if inspect.currentframe()
                    else "test_upload_dataset_to_ai_catalog_success_from_url"
                ),
            )

    @pytest.mark.parametrize(
        "prompt_template",
        [
            """
        Register this file in the DataRobot AI Catalog from raw bytes: use file_content_base64 with
        this exact value (copy it character for character, no spaces added or removed): {b64}
        Use dataset_filename exactly: ete_tiny.csv
        Do not use file_url or a local file path.
        """
        ],
    )
    async def test_upload_dataset_to_ai_catalog_success_from_base64(
        self,
        llm_client: Any,
        expectations_for_upload_dataset_to_ai_catalog_success_from_base64: ETETestExpectations,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(b64=TINY_UPLOAD_CSV_BASE64)
        async with ete_test_mcp_session() as session:
            await self._run_test_with_expectations(
                prompt,
                expectations_for_upload_dataset_to_ai_catalog_success_from_base64,
                llm_client,
                session,
                (
                    inspect.currentframe().f_code.co_name  # type: ignore[union-attr]
                    if inspect.currentframe()
                    else "test_upload_dataset_to_ai_catalog_success_from_base64"
                ),
            )

    @pytest.mark.parametrize(
        "prompt",
        [
            """
        I'm working on a machine learning project and I need to find out which datasets are
        available in the DataRobot AI Catalog. Can you help me list all the datasets in the
        AI Catalog?
        """
        ],
    )
    async def test_list_ai_catalog_items_success(
        self,
        llm_client: Any,
        expectations_for_list_ai_catalog_items_success: ETETestExpectations,
        prompt: str,
    ) -> None:
        async with ete_test_mcp_session() as session:
            await self._run_test_with_expectations(
                prompt,
                expectations_for_list_ai_catalog_items_success,
                llm_client,
                session,
                (
                    inspect.currentframe().f_code.co_name  # type: ignore[union-attr]
                    if inspect.currentframe()
                    else "test_list_ai_catalog_items_success"
                ),
            )
