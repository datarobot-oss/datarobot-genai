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

import httpx
import pytest

from datarobot_genai.drmcp.tools.clients.gdrive import GOOGLE_WORKSPACE_EXPORT_MIMES
from datarobot_genai.drmcp.tools.clients.gdrive import GoogleDriveClient
from datarobot_genai.drmcp.tools.clients.gdrive import GoogleDriveError
from datarobot_genai.drmcp.tools.clients.gdrive import GoogleDriveFile
from datarobot_genai.drmcp.tools.clients.gdrive import GoogleDriveFileContent


def make_response(
    status_code: int, json_data: dict | None = None, text: str | None = None
) -> httpx.Response:
    """Create a mock httpx.Response with a request attached."""
    request = httpx.Request("GET", "https://www.googleapis.com/drive/v3/files")
    if json_data is not None:
        return httpx.Response(status_code, json=json_data, request=request)
    return httpx.Response(status_code, text=text or "", request=request)


class TestGoogleDriveClient:
    """Test GoogleDriveClient class."""

    @pytest.fixture
    def mock_access_token(self) -> str:
        """Mock access token."""
        return "test_access_token_123"

    @pytest.fixture
    def mock_file_metadata(self) -> dict:
        """Mock Google Drive file metadata response."""
        return {
            "id": "file123",
            "name": "Test Document.docx",
            "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "size": "1024",
            "webViewLink": "https://drive.google.com/file/d/file123/view",
            "createdTime": "2025-01-01T00:00:00.000Z",
            "modifiedTime": "2025-01-10T12:00:00.000Z",
        }

    @pytest.fixture
    def mock_google_doc_metadata(self) -> dict:
        """Mock Google Doc file metadata response."""
        return {
            "id": "doc123",
            "name": "My Google Doc",
            "mimeType": "application/vnd.google-apps.document",
            "webViewLink": "https://docs.google.com/document/d/doc123/edit",
            "createdTime": "2025-01-01T00:00:00.000Z",
            "modifiedTime": "2025-01-10T12:00:00.000Z",
        }

    @pytest.fixture
    def mock_google_sheet_metadata(self) -> dict:
        """Mock Google Sheet file metadata response."""
        return {
            "id": "sheet123",
            "name": "My Google Sheet",
            "mimeType": "application/vnd.google-apps.spreadsheet",
            "webViewLink": "https://docs.google.com/spreadsheets/d/sheet123/edit",
            "createdTime": "2025-01-01T00:00:00.000Z",
            "modifiedTime": "2025-01-10T12:00:00.000Z",
        }

    @pytest.fixture
    def mock_image_metadata(self) -> dict:
        """Mock image file metadata response."""
        return {
            "id": "img123",
            "name": "photo.jpg",
            "mimeType": "image/jpeg",
            "size": "2048000",
            "webViewLink": "https://drive.google.com/file/d/img123/view",
        }

    @pytest.fixture
    def mock_text_file_metadata(self) -> dict:
        """Mock text file metadata response."""
        return {
            "id": "txt123",
            "name": "readme.txt",
            "mimeType": "text/plain",
            "size": "512",
            "webViewLink": "https://drive.google.com/file/d/txt123/view",
        }

    @pytest.mark.asyncio
    async def test_get_file_metadata_success(
        self, mock_access_token: str, mock_file_metadata: dict
    ) -> None:
        """Test successfully getting file metadata."""
        async with GoogleDriveClient(mock_access_token) as client:

            async def mock_get(url: str, params: dict | None = None) -> httpx.Response:
                return make_response(200, mock_file_metadata)

            client._client.get = mock_get

            result = await client.get_file_metadata("file123")

            assert isinstance(result, GoogleDriveFile)
            assert result.id == "file123"
            assert result.name == "Test Document.docx"
            assert (
                result.mime_type
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

    @pytest.mark.asyncio
    async def test_get_file_metadata_not_found(self, mock_access_token: str) -> None:
        """Test getting metadata for non-existent file."""
        async with GoogleDriveClient(mock_access_token) as client:

            async def mock_get(url: str, params: dict | None = None) -> httpx.Response:
                return make_response(404)

            client._client.get = mock_get

            with pytest.raises(GoogleDriveError, match="not found"):
                await client.get_file_metadata("nonexistent")

    @pytest.mark.asyncio
    async def test_get_file_metadata_permission_denied(self, mock_access_token: str) -> None:
        """Test getting metadata without permission."""
        async with GoogleDriveClient(mock_access_token) as client:

            async def mock_get(url: str, params: dict | None = None) -> httpx.Response:
                return make_response(403)

            client._client.get = mock_get

            with pytest.raises(GoogleDriveError, match="Permission denied"):
                await client.get_file_metadata("protected_file")

    @pytest.mark.asyncio
    async def test_read_file_content_google_doc(
        self, mock_access_token: str, mock_google_doc_metadata: dict
    ) -> None:
        """Test reading content from a Google Doc (exports to markdown)."""
        markdown_content = "# My Document\n\nThis is the content."

        async with GoogleDriveClient(mock_access_token) as client:
            call_count = {"get": 0}

            async def mock_get(url: str, params: dict | None = None) -> httpx.Response:
                call_count["get"] += 1
                if "/export" in url:
                    # Export endpoint
                    assert params is not None
                    assert params.get("mimeType") == "text/markdown"
                    return make_response(200, text=markdown_content)
                else:
                    # Metadata endpoint
                    return make_response(200, mock_google_doc_metadata)

            client._client.get = mock_get

            result = await client.read_file_content("doc123")

            assert isinstance(result, GoogleDriveFileContent)
            assert result.id == "doc123"
            assert result.name == "My Google Doc"
            assert result.mime_type == "text/markdown"
            assert result.content == markdown_content
            assert result.original_mime_type == "application/vnd.google-apps.document"
            assert result.was_exported is True
            assert call_count["get"] == 2  # metadata + export

    @pytest.mark.asyncio
    async def test_read_file_content_google_doc_custom_format(
        self, mock_access_token: str, mock_google_doc_metadata: dict
    ) -> None:
        """Test reading Google Doc with custom target format."""
        plain_content = "My Document\n\nThis is the content."

        async with GoogleDriveClient(mock_access_token) as client:

            async def mock_get(url: str, params: dict | None = None) -> httpx.Response:
                if "/export" in url:
                    assert params is not None
                    assert params.get("mimeType") == "text/plain"
                    return make_response(200, text=plain_content)
                else:
                    return make_response(200, mock_google_doc_metadata)

            client._client.get = mock_get

            result = await client.read_file_content("doc123", target_format="text/plain")

            assert result.mime_type == "text/plain"
            assert result.content == plain_content
            assert result.was_exported is True

    @pytest.mark.asyncio
    async def test_read_file_content_google_sheet(
        self, mock_access_token: str, mock_google_sheet_metadata: dict
    ) -> None:
        """Test reading content from a Google Sheet (exports to CSV)."""
        csv_content = "Name,Age,City\nAlice,30,NYC\nBob,25,LA"

        async with GoogleDriveClient(mock_access_token) as client:

            async def mock_get(url: str, params: dict | None = None) -> httpx.Response:
                if "/export" in url:
                    assert params is not None
                    assert params.get("mimeType") == "text/csv"
                    return make_response(200, text=csv_content)
                else:
                    return make_response(200, mock_google_sheet_metadata)

            client._client.get = mock_get

            result = await client.read_file_content("sheet123")

            assert result.mime_type == "text/csv"
            assert result.content == csv_content

    @pytest.mark.asyncio
    async def test_read_file_content_text_file(
        self, mock_access_token: str, mock_text_file_metadata: dict
    ) -> None:
        """Test reading content from a regular text file (downloads directly)."""
        file_content = "This is a plain text file content."

        async with GoogleDriveClient(mock_access_token) as client:

            async def mock_get(url: str, params: dict | None = None) -> httpx.Response:
                if params and params.get("alt") == "media":
                    # Download endpoint
                    return make_response(200, text=file_content)
                else:
                    # Metadata endpoint
                    return make_response(200, mock_text_file_metadata)

            client._client.get = mock_get

            result = await client.read_file_content("txt123")

            assert result.mime_type == "text/plain"
            assert result.content == file_content
            assert result.original_mime_type == "text/plain"
            assert result.was_exported is False
            assert result.size == 512  # From mock_text_file_metadata

    @pytest.mark.asyncio
    async def test_read_file_content_binary_file_error(
        self, mock_access_token: str, mock_image_metadata: dict
    ) -> None:
        """Test that reading binary files raises an error."""
        async with GoogleDriveClient(mock_access_token) as client:

            async def mock_get(url: str, params: dict | None = None) -> httpx.Response:
                return make_response(200, mock_image_metadata)

            client._client.get = mock_get

            with pytest.raises(GoogleDriveError, match="Binary files are not supported"):
                await client.read_file_content("img123")

    @pytest.mark.asyncio
    async def test_read_file_content_folder_error(self, mock_access_token: str) -> None:
        """Test that reading folders raises an error."""
        folder_metadata = {
            "id": "folder123",
            "name": "My Folder",
            "mimeType": "application/vnd.google-apps.folder",
        }

        async with GoogleDriveClient(mock_access_token) as client:

            async def mock_get(url: str, params: dict | None = None) -> httpx.Response:
                return make_response(200, folder_metadata)

            client._client.get = mock_get

            with pytest.raises(GoogleDriveError, match="Cannot read content of a folder"):
                await client.read_file_content("folder123")

    @pytest.mark.asyncio
    async def test_read_file_content_not_found(self, mock_access_token: str) -> None:
        """Test reading non-existent file."""
        async with GoogleDriveClient(mock_access_token) as client:

            async def mock_get(url: str, params: dict | None = None) -> httpx.Response:
                return make_response(404)

            client._client.get = mock_get

            with pytest.raises(GoogleDriveError, match="not found"):
                await client.read_file_content("nonexistent")

    @pytest.mark.asyncio
    async def test_read_file_content_pdf_text_extraction(self, mock_access_token: str) -> None:
        """Test reading PDF with text extraction."""
        pdf_metadata = {
            "id": "pdf123",
            "name": "document.pdf",
            "mimeType": "application/pdf",
            "size": 1024,
        }
        pdf_bytes = b"%PDF-1.4 fake pdf content"

        async with GoogleDriveClient(mock_access_token) as client:
            call_count = 0

            async def mock_get(url: str, params: dict | None = None) -> httpx.Response:
                nonlocal call_count
                call_count += 1
                request = httpx.Request("GET", "https://www.googleapis.com/drive/v3/files")
                if call_count == 1:  # metadata request
                    return make_response(200, pdf_metadata)
                else:  # download request
                    return httpx.Response(200, content=pdf_bytes, request=request)

            client._client.get = mock_get

            # Mock the PDF extraction to avoid needing a real PDF
            def mock_extract(pdf_bytes: bytes) -> str:
                return "Extracted PDF text content"

            client._extract_text_from_pdf = mock_extract

            result = await client.read_file_content("pdf123")

            assert result.id == "pdf123"
            assert result.name == "document.pdf"
            assert result.mime_type == "text/plain"
            assert result.original_mime_type == "application/pdf"
            assert result.was_exported is True
            assert result.content == "Extracted PDF text content"


class TestGoogleDriveFileContent:
    """Test GoogleDriveFileContent model."""

    def test_as_flat_dict_basic(self) -> None:
        """Test as_flat_dict returns correct structure for basic fields."""
        content = GoogleDriveFileContent(
            id="file123",
            name="test.md",
            mime_type="text/markdown",
            content="# Hello World",
            original_mime_type="application/vnd.google-apps.document",
            was_exported=True,
        )

        result = content.as_flat_dict()

        assert result["id"] == "file123"
        assert result["name"] == "test.md"
        assert result["mimeType"] == "text/markdown"
        assert result["content"] == "# Hello World"
        assert result["originalMimeType"] == "application/vnd.google-apps.document"
        assert result["wasExported"] is True

    def test_as_flat_dict_with_optional_fields(self) -> None:
        """Test as_flat_dict includes optional fields when present."""
        content = GoogleDriveFileContent(
            id="file123",
            name="test.txt",
            mime_type="text/plain",
            content="Hello",
            original_mime_type="text/plain",
            was_exported=False,
            size=1024,
            web_view_link="https://drive.google.com/file/d/file123/view",
        )

        result = content.as_flat_dict()

        assert result["size"] == 1024
        assert result["webViewLink"] == "https://drive.google.com/file/d/file123/view"

    def test_as_flat_dict_excludes_none_optional_fields(self) -> None:
        """Test as_flat_dict excludes optional fields when None."""
        content = GoogleDriveFileContent(
            id="file123",
            name="test.txt",
            mime_type="text/plain",
            content="Hello",
            original_mime_type="text/plain",
            was_exported=False,
        )

        result = content.as_flat_dict()

        assert "size" not in result
        assert "webViewLink" not in result


class TestIsBinaryMimeType:
    """Test _is_binary_mime_type helper."""

    def test_image_is_binary(self) -> None:
        """Test that image types are detected as binary."""
        assert GoogleDriveClient._is_binary_mime_type("image/jpeg") is True
        assert GoogleDriveClient._is_binary_mime_type("image/png") is True
        assert GoogleDriveClient._is_binary_mime_type("image/gif") is True

    def test_audio_video_is_binary(self) -> None:
        """Test that audio/video types are detected as binary."""
        assert GoogleDriveClient._is_binary_mime_type("audio/mp3") is True
        assert GoogleDriveClient._is_binary_mime_type("video/mp4") is True

    def test_pdf_is_not_binary(self) -> None:
        """Test that PDF is NOT detected as binary (we support PDF text extraction)."""
        assert GoogleDriveClient._is_binary_mime_type("application/pdf") is False

    def test_text_is_not_binary(self) -> None:
        """Test that text types are not detected as binary."""
        assert GoogleDriveClient._is_binary_mime_type("text/plain") is False
        assert GoogleDriveClient._is_binary_mime_type("text/markdown") is False
        assert GoogleDriveClient._is_binary_mime_type("text/csv") is False

    def test_json_is_not_binary(self) -> None:
        """Test that JSON is not detected as binary."""
        assert GoogleDriveClient._is_binary_mime_type("application/json") is False

    def test_google_workspace_is_not_binary(self) -> None:
        """Test that Google Workspace types are not detected as binary."""
        for mime_type in GOOGLE_WORKSPACE_EXPORT_MIMES.keys():
            assert GoogleDriveClient._is_binary_mime_type(mime_type) is False
