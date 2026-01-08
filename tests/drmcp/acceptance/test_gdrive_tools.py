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
import inspect
from typing import Any

import pytest

from datarobot_genai.drmcp import ETETestExpectations
from datarobot_genai.drmcp import ToolBaseE2E
from datarobot_genai.drmcp import ToolCallTestExpectations
from datarobot_genai.drmcp import ete_test_mcp_session
from datarobot_genai.drmcp.test_utils.tool_base_ete import SHOULD_NOT_BE_EMPTY


@pytest.fixture(scope="session")
def list_files_no_of_results() -> int:
    return 4


@pytest.fixture(scope="session")
def gdrive_folder_id() -> str:
    # This is public folder. It's available even without oauth (only api key needed)
    return "0B8sNeWRxvhpmN19oaGNmdnp2bzQ"


@pytest.fixture(scope="session")
def expectations_for_gdrive_list_files_success(
    gdrive_folder_id: str, list_files_no_of_results: int
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="google_drive_list_files",
                parameters={
                    "query": f"'{gdrive_folder_id}' in parents",
                    "limit": list_files_no_of_results,
                    "page_size": list_files_no_of_results,
                },
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "dictionario-interlingua-polonese.pdf",
            "wiechowski - interlingua.pdf",
            "dyplomword2007pdf.pdf",
            "openoffice math.pdf",
        ],
    )


@pytest.mark.asyncio
class TestGdriveToolsE2E(ToolBaseE2E):
    """End-to-end tests for gdrive tools."""

    @pytest.mark.parametrize(
        "prompt_template",
        [
            "Please list google drive files from folder '{folder_id}'. "
            "Return {number_of_results} files."
        ],
    )
    async def test_gdrive_list_files_success(
        self,
        openai_llm_client: Any,
        expectations_for_gdrive_list_files_success: ETETestExpectations,
        gdrive_folder_id: str,
        list_files_no_of_results: int,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(
            folder_id="0B8sNeWRxvhpmN19oaGNmdnp2bzQ", number_of_results=list_files_no_of_results
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_gdrive_list_files_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_gdrive_list_files_success,
                openai_llm_client,
                session,
                test_name,
            )
