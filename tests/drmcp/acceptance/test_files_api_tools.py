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

"""Acceptance tests for DataRobot Files API MCP tools."""

import inspect
import os
from typing import Any

import pytest

from datarobot_genai.drmcp import ETETestExpectations
from datarobot_genai.drmcp import ToolBaseE2E
from datarobot_genai.drmcp import ToolCallTestExpectations
from datarobot_genai.drmcp import ete_test_mcp_session


@pytest.fixture(scope="session")
def expectations_for_file_list_success() -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="file_list",
                parameters={"path": "dr://"},
                result={"path": "dr://"},
            ),
        ],
        llm_response_content_contains_expectations=[
            "catalog",
            "file",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_file_info_success(files_file_path: str) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="file_info",
                parameters={"path": files_file_path},
                result={"type": "file"},
            ),
        ],
        llm_response_content_contains_expectations=[
            "file",
            "size",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_file_read_success(files_file_path: str) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="file_read",
                parameters={"path": files_file_path},
                result={"encoding": "", "content": ""},
            ),
        ],
        llm_response_content_contains_expectations=[
            "file",
            "content",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_file_sign_success(files_file_path: str) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="file_sign",
                parameters={"path": files_file_path},
                result={"url": "", "expiration": 0},
            ),
        ],
        llm_response_content_contains_expectations=[
            "url",
            "download",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_file_info_not_found(
    nonexistent_files_path: str,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="file_info",
                parameters={"path": nonexistent_files_path},
                result="[not_found]",
            ),
        ],
        llm_response_content_contains_expectations=[
            "not found",
            "does not exist",
            "unable",
            nonexistent_files_path,
        ],
    )


@pytest.mark.skipif(
    not os.getenv("ENABLE_FILES_API_TOOLS"),
    reason="Files API tools are not enabled on the MCP server",
)
@pytest.mark.asyncio
class TestFilesApiToolsE2E(ToolBaseE2E):
    """End-to-end acceptance tests for Files API MCP tools."""

    async def test_file_list_success(
        self,
        llm_client: Any,
        expectations_for_file_list_success: ETETestExpectations,
    ) -> None:
        """LLM lists catalog items at the filesystem root via file_list."""
        prompt = (
            "List the top-level catalog items in the DataRobot filesystem. "
            "Use file_list with path='dr://' and provide a detailed summary in complete "
            "sentences of what catalog items and files you find."
        )
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_file_list_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_file_list_success,
                llm_client,
                session,
                test_name,
            )

    async def test_file_info_success(
        self,
        llm_client: Any,
        files_file_path: str,
        expectations_for_file_info_success: ETETestExpectations,
    ) -> None:
        """LLM fetches file metadata via file_info."""
        prompt = (
            f"Use file_info to inspect the DataRobot filesystem file at '{files_file_path}'. "
            "Provide a detailed summary in complete sentences describing the file type, "
            "size in bytes, format, and created_at timestamp from the tool result."
        )
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_file_info_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_file_info_success,
                llm_client,
                session,
                test_name,
            )

    async def test_file_read_success(
        self,
        llm_client: Any,
        files_file_path: str,
        expectations_for_file_read_success: ETETestExpectations,
    ) -> None:
        """LLM reads a file inline via file_read."""
        prompt = (
            f"Read the contents of '{files_file_path}' using file_read. "
            "Provide a detailed summary in complete sentences of the file encoding, "
            "how many bytes were read, and what the content says."
        )
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_file_read_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_file_read_success,
                llm_client,
                session,
                test_name,
            )

    async def test_file_sign_success(
        self,
        llm_client: Any,
        files_file_path: str,
        expectations_for_file_sign_success: ETETestExpectations,
    ) -> None:
        """LLM creates a signed download URL via file_sign."""
        prompt = (
            f"Create a temporary signed download URL for '{files_file_path}' using "
            "file_sign with expiration=300. Provide a detailed summary in complete sentences "
            "that includes the signed URL and how long it remains valid."
        )
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_file_sign_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_file_sign_success,
                llm_client,
                session,
                test_name,
            )

    async def test_file_info_not_found(
        self,
        llm_client: Any,
        nonexistent_files_path: str,
        expectations_for_file_info_not_found: ETETestExpectations,
    ) -> None:
        """LLM handles a missing filesystem path from file_info."""
        prompt = (
            f"Fetch metadata for '{nonexistent_files_path}' with file_info. "
            "If the path cannot be found, explain in detail in complete sentences "
            "what error occurred and why the lookup failed."
        )
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_file_info_not_found"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_file_info_not_found,
                llm_client,
                session,
                test_name,
            )
