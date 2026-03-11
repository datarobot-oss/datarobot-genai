# Copyright 2026 DataRobot, Inc.
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
import uuid
from typing import Any

import pytest

from datarobot_genai.drmcp import ETETestExpectations
from datarobot_genai.drmcp import ToolBaseE2E
from datarobot_genai.drmcp import ToolCallTestExpectations
from datarobot_genai.drmcp import ete_test_mcp_session
from datarobot_genai.drmcp.test_utils.tool_base_ete import SHOULD_NOT_BE_EMPTY


@pytest.fixture(scope="session")
def expectations_for_microsoft_graph_search_content_success() -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="microsoft_graph_search_content",
                parameters={
                    "search_query": "test query",
                },
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ]
    )


@pytest.mark.skipif(
    not os.getenv("ENABLE_MICROSOFT_GRAPH_TOOLS"), reason="Microsoft Graph tools are not enabled"
)
@pytest.mark.asyncio
class TestMicrosoftGraphToolsE2E(ToolBaseE2E):
    """End-to-end tests for Microsoft Graph tools."""

    @pytest.mark.parametrize(
        "prompt_template",
        ["Please search for files in the Microsoft Graph with the query 'test query'."],
    )
    async def test_microsoft_graph_search_content_success(
        self,
        llm_client: Any,
        expectations_for_microsoft_graph_search_content_success: ETETestExpectations,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = (
                frame.f_code.co_name if frame else "test_microsoft_graph_search_content_success"
            )
            await self._run_test_with_expectations(
                prompt,
                expectations_for_microsoft_graph_search_content_success,
                llm_client,
                session,
                test_name,
            )

    @pytest.mark.skip(reason="Creates real files in OneDrive without cleanup - run manually")
    async def test_microsoft_create_file_success(
        self,
        openai_llm_client: Any,
    ) -> None:
        """Test creating a new file in OneDrive.

        Note: This test creates a real file in OneDrive. The file name includes
        a UUID to ensure uniqueness and avoid conflicts. Files created by this test
        will need to be manually cleaned up or will remain in the drive.
        """
        unique_filename = f"mcp-e2e-test-{uuid.uuid4().hex[:8]}.txt"
        content = "This is a test file created by MCP E2E tests."

        expectations = ETETestExpectations(
            tool_calls_expected=[
                ToolCallTestExpectations(
                    name="microsoft_create_file",
                    parameters={
                        "file_name": unique_filename,
                        "content_text": content,
                    },
                    result=SHOULD_NOT_BE_EMPTY,
                ),
            ],
            llm_response_content_contains_expectations=[
                unique_filename,
                "created",
            ],
        )

        prompt = f"Create a file named `{unique_filename}` with content `{content}` in my OneDrive."

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_microsoft_create_file_success"
            await self._run_test_with_expectations(
                prompt,
                expectations,
                openai_llm_client,
                session,
                test_name,
            )

    @pytest.mark.skip(
        reason="Share real files in Sharepoint/OneDrive without cleanup - run manually"
    )
    async def test_microsoft_microsoft_graph_share_item_success(
        self,
        openai_llm_client: Any,
    ) -> None:
        """Test sharing a file in OneDrive/Sharepoint.

        Note: This test shares a real file.
        Files shared by this test should be manually cleaned up.
        """
        # Note: Below variables are placeholders. You should manually change them
        # to correctly test bevaviour.
        file_id = "dummy_file_id"  # Adjust manually
        document_library_id = "dummy_document_library_id"  # Adjust manually
        recipient_email = "dummy@user.com"  # Adjust manually

        expectations = ETETestExpectations(
            tool_calls_expected=[
                ToolCallTestExpectations(
                    name="microsoft_graph_share_item",
                    parameters={
                        "file_id": file_id,
                        "document_library_id": document_library_id,
                        "recipient_emails": [recipient_email],
                        "role": "read",
                        "send_invitation": False,
                    },
                    result=SHOULD_NOT_BE_EMPTY,
                ),
            ],
            llm_response_content_contains_expectations=[
                "shared",
            ],
        )

        prompt = (
            f"Share OneDrive file `{file_id}` "
            f"from document library `{document_library_id}` "
            f"to {recipient_email} as reader."
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = (
                frame.f_code.co_name
                if frame
                else "test_microsoft_microsoft_graph_share_item_success"
            )
            await self._run_test_with_expectations(
                prompt,
                expectations,
                openai_llm_client,
                session,
                test_name,
            )

    @pytest.mark.skip(reason="Modifies real data in SharePoint/OneDrive - run manually")
    async def test_microsoft_update_metadata_success(
        self,
        openai_llm_client: Any,
    ) -> None:
        """Test updating metadata on a SharePoint list item or drive item.

        Note: This test modifies real data. It requires a valid item ID and
        appropriate context (site_id + list_id for SharePoint, or document_library_id
        for OneDrive). Run manually with proper test data.
        """
        # Example for drive item - update these values for your test environment
        test_item_id = "YOUR_ITEM_ID"
        test_drive_id = "YOUR_DRIVE_ID"
        new_description = f"Updated by MCP E2E test - {uuid.uuid4().hex[:8]}"

        expectations = ETETestExpectations(
            tool_calls_expected=[
                ToolCallTestExpectations(
                    name="microsoft_update_metadata",
                    parameters={
                        "item_id": test_item_id,
                        "fields_to_update": {"description": new_description},
                        "document_library_id": test_drive_id,
                    },
                    result=SHOULD_NOT_BE_EMPTY,
                ),
            ],
            llm_response_content_contains_expectations=[
                "updated",
            ],
        )

        prompt = (
            f"Update the metadata of the item with ID `{test_item_id}` "
            f"in drive `{test_drive_id}`. Set the description to `{new_description}`."
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_microsoft_update_metadata_success"
            await self._run_test_with_expectations(
                prompt,
                expectations,
                openai_llm_client,
                session,
                test_name,
            )
