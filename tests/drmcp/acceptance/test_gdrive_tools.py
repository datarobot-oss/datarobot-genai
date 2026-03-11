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
import os
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
def gdrive_pdf_file_id() -> str:
    return "0B8sNeWRxvhpmc3hDaE1SWV9LTkk"


@pytest.fixture(scope="session")
def expectations_for_gdrive_list_files_success(
    gdrive_folder_id: str, list_files_no_of_results: int
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="gdrive_find_contents",
                parameters={
                    "folder_id": gdrive_folder_id,
                    "query": "mimeType='application/pdf'",
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


@pytest.fixture(scope="session")
def expectations_for_gdrive_read_content_success(
    gdrive_pdf_file_id: str,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="gdrive_read_content",
                parameters={
                    "file_id": gdrive_pdf_file_id,
                },
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "OpenOffice",
        ],
    )


@pytest.mark.skipif(not os.getenv("ENABLE_GDRIVE_TOOLS"), reason="Gdrive tools are not enabled")
@pytest.mark.asyncio
class TestGdriveToolsE2E(ToolBaseE2E):
    """End-to-end tests for gdrive tools."""

    @pytest.mark.parametrize(
        "prompt_template",
        [
            "Please list all pdf files from google drive folder '{folder_id}'. "
            "Return {number_of_results} files."
        ],
    )
    async def test_gdrive_find_contents_success(
        self,
        llm_client: Any,
        expectations_for_gdrive_list_files_success: ETETestExpectations,
        gdrive_folder_id: str,
        list_files_no_of_results: int,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(
            folder_id=gdrive_folder_id, number_of_results=list_files_no_of_results
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_gdrive_find_contents_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_gdrive_list_files_success,
                llm_client,
                session,
                test_name,
            )

    @pytest.mark.parametrize(
        "prompt_template",
        [
            "Please read the content of the Google Drive file with ID '{file_id}' "
            "and tell me what the document is about."
        ],
    )
    async def test_gdrive_read_content_success(
        self,
        llm_client: Any,
        expectations_for_gdrive_read_content_success: ETETestExpectations,
        gdrive_pdf_file_id: str,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(file_id=gdrive_pdf_file_id)

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_gdrive_read_content_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_gdrive_read_content_success,
                llm_client,
                session,
                test_name,
            )

    @pytest.mark.skip(reason="Creates real files in Google Drive without cleanup - run manually")
    @pytest.mark.parametrize(
        "prompt_template",
        [
            "Please create a new text file in Google Drive named 'test-acceptance-file.txt' "
            "with the content 'Hello from acceptance test'."
        ],
    )
    async def test_gdrive_create_file_success(
        self,
        llm_client: Any,
        prompt_template: str,
    ) -> None:
        expectations = ETETestExpectations(
            tool_calls_expected=[
                ToolCallTestExpectations(
                    name="gdrive_create_file",
                    parameters={
                        "name": "test-acceptance-file.txt",
                        "mime_type": "text/plain",
                    },
                    result=SHOULD_NOT_BE_EMPTY,
                ),
            ],
            llm_response_content_contains_expectations=["created", "test-acceptance-file.txt"],
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_gdrive_create_file_success"
            await self._run_test_with_expectations(
                prompt_template,
                expectations,
                llm_client,
                session,
                test_name,
            )

    @pytest.mark.skip(reason="Modifies real files in Google Drive - run manually")
    @pytest.mark.parametrize(
        "prompt_template",
        [
            "Please rename the Google Drive file with ID '{file_id}' to 'renamed-test-file.txt' "
            "and star it."
        ],
    )
    async def test_gdrive_update_metadata_success(
        self,
        llm_client: Any,
        prompt_template: str,
    ) -> None:
        # Note: Replace with a real file ID when running manually
        test_file_id = "test_file_id_placeholder"
        prompt = prompt_template.format(file_id=test_file_id)

        expectations = ETETestExpectations(
            tool_calls_expected=[
                ToolCallTestExpectations(
                    name="gdrive_update_metadata",
                    parameters={
                        "file_id": test_file_id,
                        "new_name": "renamed-test-file.txt",
                        "starred": True,
                    },
                    result=SHOULD_NOT_BE_EMPTY,
                ),
            ],
            llm_response_content_contains_expectations=["renamed", "starred"],
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_gdrive_update_metadata_success"
            await self._run_test_with_expectations(
                prompt,
                expectations,
                llm_client,
                session,
                test_name,
            )

    @pytest.mark.skip(reason="Modifies real files in Google Drive - run manually")
    @pytest.mark.parametrize(
        "prompt_template",
        ["Please share the Google Drive file with ID '{file_id}' to '{email}' as a reader."],
    )
    async def test_gdrive_manage_access_success(
        self,
        llm_client: Any,
        prompt_template: str,
    ) -> None:
        # Note: Replace with a real file ID and email when running manually
        test_file_id = "1pNg6bRCbsYlzfUJk5_2qaeMHflVfhCbJ"  # "test_file_id_placeholder"
        test_email = "wojciech.wierzchowski@datarobot.com"  # "dummy@user.com"
        prompt = prompt_template.format(file_id=test_file_id, email=test_email)

        expectations = ETETestExpectations(
            tool_calls_expected=[
                ToolCallTestExpectations(
                    name="gdrive_manage_access",
                    parameters={
                        "file_id": test_file_id,
                        "action": "add",
                        "role": "reader",
                        "email_address": test_email,
                    },
                    result=SHOULD_NOT_BE_EMPTY,
                ),
            ],
            llm_response_content_contains_expectations=["successfully shared"],
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_gdrive_manage_access_success"
            await self._run_test_with_expectations(
                prompt,
                expectations,
                llm_client,
                session,
                test_name,
            )
