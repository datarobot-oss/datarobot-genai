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
import json
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

from datarobot_genai.drtools.clients.gdrive import GOOGLE_DRIVE_FOLDER_MIME
from datarobot_genai.drtools.clients.gdrive import GoogleDriveError
from datarobot_genai.drtools.clients.gdrive import GoogleDriveFile
from datarobot_genai.drtools.clients.gdrive import GoogleDriveFileContent
from datarobot_genai.drtools.clients.gdrive import PaginatedResult
from datarobot_genai.drtools.gdrive.tools import gdrive_create_file
from datarobot_genai.drtools.gdrive.tools import gdrive_find_contents
from datarobot_genai.drtools.gdrive.tools import gdrive_manage_access
from datarobot_genai.drtools.gdrive.tools import gdrive_read_content
from datarobot_genai.drtools.gdrive.tools import gdrive_update_metadata


@pytest.fixture
def get_gdrive_access_token_mock() -> Iterator[None]:
    with patch(
        "datarobot_genai.drtools.gdrive.tools.get_gdrive_access_token",
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
    with patch("datarobot_genai.drtools.clients.gdrive.GoogleDriveClient.list_files") as mock:
        response = PaginatedResult(files=gdrive_files, next_page_token=gdrive_next_page_token)
        mock.return_value = response
        yield response


@pytest.fixture
def gdrive_client_list_files_without_next_page_mock(
    gdrive_files: list[GoogleDriveFile],
) -> Iterator[PaginatedResult]:
    with patch("datarobot_genai.drtools.clients.gdrive.GoogleDriveClient.list_files") as mock:
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
        expected = {
            "files": [{"id": file.id, "name": file.name} for file in gdrive_files],
            "count": len(gdrive_files),
            "nextPageToken": gdrive_next_page_token,
        }
        assert json.loads(content[0].text) == expected
        assert structured_content == expected

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
        expected = {
            "files": [{"id": file.id, "name": file.name} for file in gdrive_files],
            "count": len(gdrive_files),
            "nextPageToken": None,
        }
        assert json.loads(content[0].text) == expected
        assert structured_content == expected


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
            "datarobot_genai.drtools.clients.gdrive.GoogleDriveClient.read_file_content"
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
        text_data = json.loads(content[0].text)
        assert text_data["id"] == "doc123"
        assert text_data["mimeType"] == "text/markdown"
        assert structured_content["id"] == "doc123"
        assert structured_content["name"] == "My Document"
        assert structured_content["mimeType"] == "text/markdown"
        assert structured_content["content"] == "# Hello World\n\nThis is content."
        assert structured_content["originalMimeType"] == "application/vnd.google-apps.document"
        assert structured_content["wasExported"] is True
        assert structured_content["size"] == 1024
        assert structured_content["webViewLink"] == "https://docs.google.com/document/d/doc123/edit"
        assert structured_content["wasExported"] is True

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
            "datarobot_genai.drtools.clients.gdrive.GoogleDriveClient.read_file_content"
        ) as mock:
            mock.return_value = csv_content
            tool_result = await gdrive_read_content(file_id="sheet123")

        content, structured_content = tool_result.to_mcp_result()
        assert json.loads(content[0].text)["mimeType"] == "text/csv"
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
            "datarobot_genai.drtools.clients.gdrive.GoogleDriveClient.read_file_content"
        ) as mock:
            mock.return_value = text_content
            tool_result = await gdrive_read_content(file_id="txt123")

        content, structured_content = tool_result.to_mcp_result()
        assert json.loads(content[0].text)["wasExported"] is False
        assert structured_content["wasExported"] is False

    @pytest.mark.asyncio
    async def test_gdrive_read_content_empty_file_id(
        self,
        get_gdrive_access_token_mock: None,
    ) -> None:
        """Gdrive read content -- empty file_id raises error."""
        with pytest.raises(ToolError, match="file_id.*cannot be empty"):
            await gdrive_read_content(file_id="")

    @pytest.mark.asyncio
    async def test_gdrive_read_content_whitespace_file_id(
        self,
        get_gdrive_access_token_mock: None,
    ) -> None:
        """Gdrive read content -- whitespace file_id raises error."""
        with pytest.raises(ToolError, match="file_id.*cannot be empty"):
            await gdrive_read_content(file_id="   ")

    @pytest.mark.asyncio
    async def test_gdrive_read_content_file_not_found(
        self,
        get_gdrive_access_token_mock: None,
    ) -> None:
        """Gdrive read content -- file not found raises error."""
        with patch(
            "datarobot_genai.drtools.clients.gdrive.GoogleDriveClient.read_file_content"
        ) as mock:
            mock.side_effect = GoogleDriveError("File with ID 'nonexistent' not found.")
            with pytest.raises(ToolError, match="not found"):
                await gdrive_read_content(file_id="nonexistent")

    @pytest.mark.asyncio
    async def test_gdrive_read_content_binary_file_error(
        self,
        get_gdrive_access_token_mock: None,
    ) -> None:
        """Gdrive read content -- binary file raises error."""
        with patch(
            "datarobot_genai.drtools.clients.gdrive.GoogleDriveClient.read_file_content"
        ) as mock:
            mock.side_effect = GoogleDriveError(
                "Binary files are not supported for reading. "
                "File 'photo.jpg' has MIME type 'image/jpeg'."
            )
            with pytest.raises(ToolError, match="Binary files are not supported"):
                await gdrive_read_content(file_id="img123")


class TestGdriveCreateFile:
    """Gdrive create file tool tests."""

    @pytest.fixture
    def created_file(self) -> GoogleDriveFile:
        return GoogleDriveFile(
            id="new_file_123",
            name="My New File.txt",
            mime_type="text/plain",
            web_view_link="https://drive.google.com/file/d/new_file_123/view",
        )

    @pytest.fixture
    def created_google_doc(self) -> GoogleDriveFile:
        return GoogleDriveFile(
            id="new_doc_123",
            name="My New Document",
            mime_type="application/vnd.google-apps.document",
            web_view_link="https://docs.google.com/document/d/new_doc_123/edit",
        )

    @pytest.fixture
    def created_folder(self) -> GoogleDriveFile:
        return GoogleDriveFile(
            id="new_folder_123",
            name="My New Folder",
            mime_type=GOOGLE_DRIVE_FOLDER_MIME,
            web_view_link="https://drive.google.com/drive/folders/new_folder_123",
        )

    @pytest.mark.asyncio
    async def test_gdrive_create_file_happy_path(
        self,
        get_gdrive_access_token_mock: None,
        created_file: GoogleDriveFile,
    ) -> None:
        """Gdrive create file -- happy path."""
        with patch("datarobot_genai.drtools.clients.gdrive.GoogleDriveClient.create_file") as mock:
            mock.return_value = created_file
            tool_result = await gdrive_create_file(name="My New File.txt", mime_type="text/plain")

        content, structured_content = tool_result.to_mcp_result()
        assert json.loads(content[0].text) == structured_content
        assert structured_content["id"] == "new_file_123"
        assert structured_content["name"] == "My New File.txt"
        assert structured_content["mimeType"] == "text/plain"
        assert (
            structured_content["webViewLink"] == "https://drive.google.com/file/d/new_file_123/view"
        )

    @pytest.mark.asyncio
    async def test_gdrive_create_file_with_content(
        self,
        get_gdrive_access_token_mock: None,
        created_file: GoogleDriveFile,
    ) -> None:
        """Gdrive create file -- with initial content."""
        with patch("datarobot_genai.drtools.clients.gdrive.GoogleDriveClient.create_file") as mock:
            mock.return_value = created_file
            tool_result = await gdrive_create_file(
                name="My New File.txt", mime_type="text/plain", initial_content="Hello, World!"
            )

        content, structured_content = tool_result.to_mcp_result()
        assert json.loads(content[0].text)["id"] == "new_file_123"
        assert structured_content["id"] == "new_file_123"

    @pytest.mark.asyncio
    async def test_gdrive_create_google_doc(
        self,
        get_gdrive_access_token_mock: None,
        created_google_doc: GoogleDriveFile,
    ) -> None:
        """Gdrive create file -- Google Doc with content."""
        with patch("datarobot_genai.drtools.clients.gdrive.GoogleDriveClient.create_file") as mock:
            mock.return_value = created_google_doc
            tool_result = await gdrive_create_file(
                name="My New Document",
                mime_type="application/vnd.google-apps.document",
                initial_content="# My Report\n\nContent here.",
            )

        content, structured_content = tool_result.to_mcp_result()
        assert json.loads(content[0].text)["mimeType"] == "application/vnd.google-apps.document"
        assert structured_content["mimeType"] == "application/vnd.google-apps.document"

    @pytest.mark.asyncio
    async def test_gdrive_create_folder(
        self,
        get_gdrive_access_token_mock: None,
        created_folder: GoogleDriveFile,
    ) -> None:
        """Gdrive create file -- folder creation."""
        with patch("datarobot_genai.drtools.clients.gdrive.GoogleDriveClient.create_file") as mock:
            mock.return_value = created_folder
            tool_result = await gdrive_create_file(
                name="My New Folder", mime_type=GOOGLE_DRIVE_FOLDER_MIME
            )

        content, structured_content = tool_result.to_mcp_result()
        assert json.loads(content[0].text)["mimeType"] == GOOGLE_DRIVE_FOLDER_MIME
        assert structured_content["mimeType"] == GOOGLE_DRIVE_FOLDER_MIME

    @pytest.mark.asyncio
    async def test_gdrive_create_file_empty_name(
        self,
        get_gdrive_access_token_mock: None,
    ) -> None:
        """Gdrive create file -- empty name raises error."""
        with pytest.raises(ToolError, match="name.*cannot be empty"):
            await gdrive_create_file(name="", mime_type="text/plain")

    @pytest.mark.asyncio
    async def test_gdrive_create_file_whitespace_name(
        self,
        get_gdrive_access_token_mock: None,
    ) -> None:
        """Gdrive create file -- whitespace name raises error."""
        with pytest.raises(ToolError, match="name.*cannot be empty"):
            await gdrive_create_file(name="   ", mime_type="text/plain")

    @pytest.mark.asyncio
    async def test_gdrive_create_file_empty_mime_type(
        self,
        get_gdrive_access_token_mock: None,
    ) -> None:
        """Gdrive create file -- empty mime_type raises error."""
        with pytest.raises(ToolError, match="mime_type.*cannot be empty"):
            await gdrive_create_file(name="file.txt", mime_type="")

    @pytest.mark.asyncio
    async def test_gdrive_create_file_error_handling(
        self,
        get_gdrive_access_token_mock: None,
    ) -> None:
        """Gdrive create file -- GoogleDriveError is propagated."""
        with patch("datarobot_genai.drtools.clients.gdrive.GoogleDriveClient.create_file") as mock:
            mock.side_effect = GoogleDriveError("Parent folder not found.")
            with pytest.raises(ToolError, match="Parent folder not found"):
                await gdrive_create_file(
                    name="file.txt", mime_type="text/plain", parent_id="nonexistent"
                )


class TestGdriveManageAccess:
    """Gdrive manage access tool tests."""

    @pytest.mark.asyncio
    async def test_gdrive_add_role_happy_path(self, get_gdrive_access_token_mock: None) -> None:
        """Gdrive add role -- happy path."""
        new_permission_id = "dummy_permission_id"
        with patch(
            "datarobot_genai.drtools.clients.gdrive.GoogleDriveClient.manage_access"
        ) as mock:
            mock.return_value = new_permission_id
            tool_result = await gdrive_manage_access(
                file_id="dummy_file_id", action="add", role="reader", email_address="dummy@user.com"
            )

        content, structured_content = tool_result.to_mcp_result()
        assert json.loads(content[0].text)["affectedFileId"] == "dummy_file_id"
        assert json.loads(content[0].text)["newPermissionId"] == new_permission_id
        assert structured_content["affectedFileId"] == "dummy_file_id"
        assert structured_content["newPermissionId"] == new_permission_id

    @pytest.mark.asyncio
    async def test_gdrive_update_role_happy_path(self, get_gdrive_access_token_mock: None) -> None:
        """Gdrive update role -- happy path."""
        with patch(
            "datarobot_genai.drtools.clients.gdrive.GoogleDriveClient.manage_access"
        ) as mock:
            permission_id = "dummy_permission_id"
            mock.return_value = permission_id
            tool_result = await gdrive_manage_access(
                file_id="dummy_file_id",
                action="update",
                role="reader",
                permission_id=permission_id,
            )

        content, structured_content = tool_result.to_mcp_result()
        assert json.loads(content[0].text)["affectedFileId"] == "dummy_file_id"
        assert structured_content["affectedFileId"] == "dummy_file_id"

    @pytest.mark.asyncio
    async def test_gdrive_remove_role_happy_path(self, get_gdrive_access_token_mock: None) -> None:
        """Gdrive remove role -- happy path."""
        with patch(
            "datarobot_genai.drtools.clients.gdrive.GoogleDriveClient.manage_access"
        ) as mock:
            permission_id = "dummy_permission_id"
            mock.return_value = permission_id
            tool_result = await gdrive_manage_access(
                file_id="dummy_file_id", action="remove", permission_id=permission_id
            )

        content, structured_content = tool_result.to_mcp_result()
        assert json.loads(content[0].text)["affectedFileId"] == "dummy_file_id"
        assert structured_content["affectedFileId"] == "dummy_file_id"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "function_kwargs,error_message",
        [
            ({"file_id": "", "action": "add"}, "file_id.*cannot be empty"),
            (
                {"file_id": "dummy_file_id", "action": "add", "email_address": ""},
                "email_address.*is required for action 'add'",
            ),
            (
                {"file_id": "dummy_file_id", "action": "update", "permission_id": ""},
                "permission_id.*is required for action.*update",
            ),
            (
                {"file_id": "dummy_file_id", "action": "remove", "permission_id": ""},
                "permission_id.*is required for action.*remove",
            ),
            (
                {
                    "file_id": "dummy_file_id",
                    "action": "add",
                    "email_address": "dummy@email.com",
                    "role": "",
                },
                "role.*is required for action.*add",
            ),
            (
                {
                    "file_id": "dummy_file_id",
                    "action": "update",
                    "permission_id": "dummy_permission_id",
                    "role": "",
                },
                "role.*is required for action.*update",
            ),
        ],
    )
    async def test_gdrive_manage_access_input_validation(
        self,
        get_gdrive_access_token_mock: None,
        function_kwargs: dict,
        error_message: str,
    ) -> None:
        """Gdrive manage access -- input validation."""
        with pytest.raises(ToolError, match=error_message):
            await gdrive_manage_access(**function_kwargs)

    @pytest.mark.asyncio
    async def test_gdrive_manage_access_when_error_in_client(
        self,
        get_gdrive_access_token_mock: None,
    ) -> None:
        """Gdrive manage access -- error in client."""
        error_msg = "Dummy Drive API Error."
        with pytest.raises(ToolError, match=error_msg):
            with patch(
                "datarobot_genai.drtools.clients.gdrive.GoogleDriveClient.manage_access"
            ) as mock:
                mock.side_effect = GoogleDriveError(error_msg)
                await gdrive_manage_access(
                    file_id="dummy_file_id",
                    action="add",
                    email_address="dummy@email.com",
                    role="reader",
                )


@pytest.fixture
def gdrive_client_update_file_metadata_mock() -> Iterator[GoogleDriveFile]:
    with patch(
        "datarobot_genai.drtools.clients.gdrive.GoogleDriveClient.update_file_metadata"
    ) as mock:
        updated_file = GoogleDriveFile(
            id="file_123",
            name="Updated File.txt",
            mime_type="text/plain",
            starred=True,
            trashed=False,
        )
        mock.return_value = updated_file
        yield updated_file


@pytest.fixture
def gdrive_client_update_file_metadata_error_mock() -> Iterator[None]:
    with patch(
        "datarobot_genai.drtools.clients.gdrive.GoogleDriveClient.update_file_metadata"
    ) as mock:
        mock.side_effect = GoogleDriveError("File not found.")
        yield


class TestGdriveUpdateMetadata:
    """Gdrive update metadata tool tests."""

    @pytest.mark.asyncio
    async def test_gdrive_update_metadata_happy_path(
        self,
        get_gdrive_access_token_mock: None,
        gdrive_client_update_file_metadata_mock: GoogleDriveFile,
    ) -> None:
        """Gdrive update metadata -- happy path with multiple updates."""
        tool_result = await gdrive_update_metadata(
            file_id="file_123", new_name="Updated File.txt", starred=True, trash=False
        )

        content, structured_content = tool_result.to_mcp_result()
        assert json.loads(content[0].text)["name"] == "Updated File.txt"
        assert json.loads(content[0].text)["starred"] is True
        assert structured_content["id"] == "file_123"
        assert structured_content["name"] == "Updated File.txt"
        assert structured_content["starred"] is True

    @pytest.mark.asyncio
    async def test_gdrive_update_metadata_empty_file_id(
        self,
        get_gdrive_access_token_mock: None,
    ) -> None:
        """Gdrive update metadata -- empty file_id raises error."""
        with pytest.raises(ToolError, match="file_id.*cannot be empty"):
            await gdrive_update_metadata(file_id="", new_name="New.txt")

    @pytest.mark.asyncio
    async def test_gdrive_update_metadata_when_error_in_client(
        self,
        get_gdrive_access_token_mock: None,
        gdrive_client_update_file_metadata_error_mock: None,
    ) -> None:
        """Gdrive update metadata -- error in client."""
        with pytest.raises(ToolError, match="not found"):
            await gdrive_update_metadata(file_id="nonexistent", new_name="New.txt")
