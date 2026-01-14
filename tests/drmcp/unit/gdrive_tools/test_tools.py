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

from datarobot_genai.drmcp.core.exceptions import MCPError
from datarobot_genai.drmcp.tools.clients.gdrive import GoogleDriveError
from datarobot_genai.drmcp.tools.clients.gdrive import GoogleDriveFile
from datarobot_genai.drmcp.tools.clients.gdrive import GoogleDriveFileContent
from datarobot_genai.drmcp.tools.clients.gdrive import PaginatedResult
from datarobot_genai.drmcp.tools.gdrive.tools import gdrive_find_contents
from datarobot_genai.drmcp.tools.gdrive.tools import gdrive_read_content


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
    async def test_gdrive_find_contents_when_next_page_available_happy_path(
        self,
        get_gdrive_access_token_mock: None,
        gdrive_files: list[GoogleDriveFile],
        gdrive_next_page_token: str,
        gdrive_client_list_files_with_next_page_mock: PaginatedResult,
    ) -> None:
        """Gdrive list files -- happy path."""
        tool_result = await gdrive_find_contents(fields=["id", "name"])

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
    async def test_gdrive_find_contents_when_no_more_pages_happy_path(
        self,
        get_gdrive_access_token_mock: None,
        gdrive_files: list[GoogleDriveFile],
        gdrive_client_list_files_without_next_page_mock: PaginatedResult,
    ) -> None:
        """Gdrive list files -- happy path."""
        tool_result = await gdrive_find_contents(fields=["id", "name"])

        content, structured_content = tool_result.to_mcp_result()
        assert content[0].text == "Successfully listed 2 files. There're no more pages."
        assert structured_content == {
            "files": [{"id": file.id, "name": file.name} for file in gdrive_files],
            "count": len(gdrive_files),
            "nextPageToken": None,
        }


class TestGdriveReadContent:
    """Gdrive read content tool tests."""

    @pytest.fixture
    def gdrive_file_content(self) -> GoogleDriveFileContent:
        return GoogleDriveFileContent(
            id="doc123",
            name="My Document",
            mime_type="text/markdown",
            content="# Hello World\n\nThis is content.",
            original_mime_type="application/vnd.google-apps.document",
            was_exported=True,
            size=1024,
            web_view_link="https://docs.google.com/document/d/doc123/edit",
        )

    @pytest.fixture
    def gdrive_client_read_file_content_mock(
        self, gdrive_file_content: GoogleDriveFileContent
    ) -> Iterator[GoogleDriveFileContent]:
        with patch(
            "datarobot_genai.drmcp.tools.clients.gdrive.GoogleDriveClient.read_file_content"
        ) as mock:
            mock.return_value = gdrive_file_content
            yield gdrive_file_content

    @pytest.mark.asyncio
    async def test_gdrive_read_content_happy_path(
        self,
        get_gdrive_access_token_mock: None,
        gdrive_file_content: GoogleDriveFileContent,
        gdrive_client_read_file_content_mock: GoogleDriveFileContent,
    ) -> None:
        """Gdrive read content -- happy path."""
        tool_result = await gdrive_read_content(file_id="doc123")

        content, structured_content = tool_result.to_mcp_result()
        assert "Successfully retrieved content of 'My Document'" in content[0].text
        assert "text/markdown" in content[0].text
        # Should show export info since was_exported is True
        assert "exported from" in content[0].text
        assert structured_content["id"] == "doc123"
        assert structured_content["name"] == "My Document"
        assert structured_content["mimeType"] == "text/markdown"
        assert structured_content["content"] == "# Hello World\n\nThis is content."
        assert structured_content["originalMimeType"] == "application/vnd.google-apps.document"
        assert structured_content["wasExported"] is True
        assert structured_content["size"] == 1024
        assert structured_content["webViewLink"] == "https://docs.google.com/document/d/doc123/edit"

    @pytest.mark.asyncio
    async def test_gdrive_read_content_csv_file(
        self,
        get_gdrive_access_token_mock: None,
    ) -> None:
        """Gdrive read content -- CSV file (from Google Sheet)."""
        csv_content = GoogleDriveFileContent(
            id="sheet123",
            name="My Sheet",
            mime_type="text/csv",
            content="Name,Age\nAlice,30",
            original_mime_type="application/vnd.google-apps.spreadsheet",
            was_exported=True,
        )

        with patch(
            "datarobot_genai.drmcp.tools.clients.gdrive.GoogleDriveClient.read_file_content"
        ) as mock:
            mock.return_value = csv_content
            tool_result = await gdrive_read_content(file_id="sheet123")

        content, structured_content = tool_result.to_mcp_result()
        assert "text/csv" in content[0].text
        assert structured_content["content"] == "Name,Age\nAlice,30"
        assert structured_content["wasExported"] is True

    @pytest.mark.asyncio
    async def test_gdrive_read_content_regular_text_file(
        self,
        get_gdrive_access_token_mock: None,
    ) -> None:
        """Gdrive read content -- regular text file (not exported)."""
        text_content = GoogleDriveFileContent(
            id="txt123",
            name="readme.txt",
            mime_type="text/plain",
            content="Hello World",
            original_mime_type="text/plain",
            was_exported=False,
        )

        with patch(
            "datarobot_genai.drmcp.tools.clients.gdrive.GoogleDriveClient.read_file_content"
        ) as mock:
            mock.return_value = text_content
            tool_result = await gdrive_read_content(file_id="txt123")

        content, structured_content = tool_result.to_mcp_result()
        # Should NOT show export info since was_exported is False
        assert "exported from" not in content[0].text
        assert structured_content["wasExported"] is False

    @pytest.mark.asyncio
    async def test_gdrive_read_content_empty_file_id(
        self,
        get_gdrive_access_token_mock: None,
    ) -> None:
        """Gdrive read content -- empty file_id raises error."""
        with pytest.raises(MCPError, match="file_id.*cannot be empty"):
            await gdrive_read_content(file_id="")

    @pytest.mark.asyncio
    async def test_gdrive_read_content_whitespace_file_id(
        self,
        get_gdrive_access_token_mock: None,
    ) -> None:
        """Gdrive read content -- whitespace file_id raises error."""
        with pytest.raises(MCPError, match="file_id.*cannot be empty"):
            await gdrive_read_content(file_id="   ")

    @pytest.mark.asyncio
    async def test_gdrive_read_content_file_not_found(
        self,
        get_gdrive_access_token_mock: None,
    ) -> None:
        """Gdrive read content -- file not found raises error."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.gdrive.GoogleDriveClient.read_file_content"
        ) as mock:
            mock.side_effect = GoogleDriveError("File with ID 'nonexistent' not found.")
            with pytest.raises(MCPError, match="not found"):
                await gdrive_read_content(file_id="nonexistent")

    @pytest.mark.asyncio
    async def test_gdrive_read_content_binary_file_error(
        self,
        get_gdrive_access_token_mock: None,
    ) -> None:
        """Gdrive read content -- binary file raises error."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.gdrive.GoogleDriveClient.read_file_content"
        ) as mock:
            mock.side_effect = GoogleDriveError(
                "Binary files are not supported for reading. "
                "File 'photo.jpg' has MIME type 'image/jpeg'."
            )
            with pytest.raises(MCPError, match="Binary files are not supported"):
                await gdrive_read_content(file_id="img123")
