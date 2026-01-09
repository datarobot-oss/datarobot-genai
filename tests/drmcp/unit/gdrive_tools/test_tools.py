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
from collections.abc import Iterator
from unittest.mock import patch

import pytest

from datarobot_genai.drmcp.tools.clients.gdrive import GoogleDriveFile
from datarobot_genai.drmcp.tools.clients.gdrive import PaginatedResult
from datarobot_genai.drmcp.tools.gdrive.tools import google_drive_list_files


@pytest.fixture
def get_gdrive_access_token_mock() -> Iterator[None]:
    with patch(
        "datarobot_genai.drmcp.tools.gdrive.tools.get_gdrive_access_token",
        return_value="token",
    ):
        yield


@pytest.fixture
def gdrive_files() -> list[GoogleDriveFile]:
    return [
        GoogleDriveFile(id="dummy_id_1", name="Dummy file 1", mime_type="pdf"),
        GoogleDriveFile(id="dummy_id_2", name="Dummy file 2", mime_type="docx"),
    ]


@pytest.fixture
def gdrive_next_page_token() -> str:
    return "dummy next page token"


@pytest.fixture
def gdrive_client_list_files_with_next_page_mock(
    gdrive_files: list[GoogleDriveFile], gdrive_next_page_token: str
) -> Iterator[PaginatedResult]:
    with patch("datarobot_genai.drmcp.tools.clients.gdrive.GoogleDriveClient.list_files") as mock:
        response = PaginatedResult(files=gdrive_files, next_page_token=gdrive_next_page_token)
        mock.return_value = response
        yield response


@pytest.fixture
def gdrive_client_list_files_without_next_page_mock(
    gdrive_files: list[GoogleDriveFile],
) -> Iterator[PaginatedResult]:
    with patch("datarobot_genai.drmcp.tools.clients.gdrive.GoogleDriveClient.list_files") as mock:
        response = PaginatedResult(files=gdrive_files, next_page_token=None)
        mock.return_value = response
        yield response


class TestGdriveListFiles:
    """Gdrive list files tool test."""

    @pytest.mark.asyncio
    async def test_gdrive_list_files_when_next_page_available_happy_path(
        self,
        get_gdrive_access_token_mock: None,
        gdrive_files: list[GoogleDriveFile],
        gdrive_next_page_token: str,
        gdrive_client_list_files_with_next_page_mock: PaginatedResult,
    ) -> None:
        """Gdrive list files -- happy path."""
        tool_result = await google_drive_list_files(fields=["id", "name"])

        content, structured_content = tool_result.to_mcp_result()
        assert (
            content[0].text == "Successfully listed 2 files. "
            f"Next page token needed to fetch more data: {gdrive_next_page_token}"
        )
        assert structured_content == {
            "files": [{"id": file.id, "name": file.name} for file in gdrive_files],
            "count": len(gdrive_files),
            "nextPageToken": gdrive_next_page_token,
        }

    @pytest.mark.asyncio
    async def test_gdrive_list_files_when_no_more_pages_happy_path(
        self,
        get_gdrive_access_token_mock: None,
        gdrive_files: list[GoogleDriveFile],
        gdrive_client_list_files_without_next_page_mock: PaginatedResult,
    ) -> None:
        """Gdrive list files -- happy path."""
        tool_result = await google_drive_list_files(fields=["id", "name"])

        content, structured_content = tool_result.to_mcp_result()
        assert content[0].text == "Successfully listed 2 files. There're no more pages."
        assert structured_content == {
            "files": [{"id": file.id, "name": file.name} for file in gdrive_files],
            "count": len(gdrive_files),
            "nextPageToken": None,
        }
