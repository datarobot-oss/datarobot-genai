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

from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import aiohttp
import pytest

from datarobot_genai.drmcp.tools.external.google_drive import GoogleDriveClient
from datarobot_genai.drmcp.tools.external.google_drive import GoogleDriveFile
from datarobot_genai.drmcp.tools.external.google_drive import list_google_drive_items


@pytest.fixture
def access_token() -> str:
    """Return a test OAuth access token."""
    return "test-oauth-token-12345"


@pytest.fixture
def sample_api_response() -> dict[str, Any]:
    """Return a sample Google Drive API response with file data."""
    return {
        "files": [
            {
                "id": "file1",
                "name": "Document.pdf",
                "mimeType": "application/pdf",
                "modifiedTime": "2025-01-15T10:30:00Z",
                "size": "1024000",
            },
            {
                "id": "file2",
                "name": "Spreadsheet.xlsx",
                "mimeType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "modifiedTime": "2025-01-16T14:20:00Z",
                "size": "2048000",
            },
            {
                "id": "file3",
                "name": "notes.txt",
                "mimeType": "text/plain",
            },
        ]
    }


@pytest.fixture
def empty_api_response() -> dict[str, Any]:
    """Return an empty Google Drive API response."""
    return {"files": []}


@pytest.fixture
def mock_aiohttp_response() -> AsyncMock:
    """Create a mock aiohttp response with proper async support.

    Note: Uses AsyncMock for async methods (json) and MagicMock for sync methods
    (raise_for_status).
    """
    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = AsyncMock()
    return mock_response


@pytest.fixture
def mock_aiohttp_session(mock_aiohttp_response: AsyncMock) -> MagicMock:
    """Create a mock aiohttp ClientSession with proper context manager support.

    This uses MagicMock (not AsyncMock) for the session because we need to manually
    configure __aenter__ and __aexit__ to properly support async context managers.
    AsyncMock with spec doesn't handle context managers correctly.
    """
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock()
    mock_session.get = MagicMock()
    mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_aiohttp_response)
    mock_session.get.return_value.__aexit__ = AsyncMock()
    return mock_session


@pytest.fixture
def mock_error_response() -> Any:
    """Create a factory for mock error responses.

    Returns a factory function that creates properly configured error mocks
    for testing error handling scenarios.
    """

    def _create_error_response(status: int, message: str) -> tuple[MagicMock, Any]:
        error = aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=status,
            message=message,
        )

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = error
        mock_response.json = AsyncMock(return_value={})

        # Create session context manager
        mock_get_cm = MagicMock()
        mock_get_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get.return_value = mock_get_cm

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        return mock_session_cm, error

    return _create_error_response


class TestGoogleDriveFile:
    """Test suite for GoogleDriveFile dataclass."""

    def test_from_api_response_complete_data(self) -> None:
        """Test creating GoogleDriveFile from complete API response data."""
        api_data = {
            "id": "test123",
            "name": "test.pdf",
            "mimeType": "application/pdf",
            "modifiedTime": "2025-01-15T10:30:00Z",
            "size": "1024",
        }

        file = GoogleDriveFile.from_api_response(api_data)

        assert file.id == "test123"
        assert file.name == "test.pdf"
        assert file.mime_type == "application/pdf"
        assert file.modified_time == "2025-01-15T10:30:00Z"
        assert file.size == "1024"

    def test_from_api_response_minimal_data(self) -> None:
        """Test creating GoogleDriveFile with minimal required fields."""
        api_data = {
            "id": "minimal123",
            "name": "minimal.txt",
        }

        file = GoogleDriveFile.from_api_response(api_data)

        assert file.id == "minimal123"
        assert file.name == "minimal.txt"
        assert file.mime_type == "unknown"
        assert file.modified_time is None
        assert file.size is None

    def test_format_summary(self) -> None:
        """Test formatting file information as summary string."""
        file = GoogleDriveFile(
            id="doc123",
            name="My Document.pdf",
            mime_type="application/pdf",
        )

        summary = file.format_summary()

        assert summary == "doc123: My Document.pdf (application/pdf)"


class TestGoogleDriveClient:
    """Test suite for GoogleDriveClient class."""

    @pytest.mark.asyncio
    async def test_list_files_success(
        self,
        access_token: str,
        sample_api_response: dict[str, Any],
        mock_aiohttp_session: MagicMock,
        mock_aiohttp_response: AsyncMock,
    ) -> None:
        """Test successful file listing from Google Drive."""
        client = GoogleDriveClient(access_token=access_token)

        mock_aiohttp_response.json.return_value = sample_api_response

        with patch("aiohttp.ClientSession", return_value=mock_aiohttp_session):
            files = await client.list_files(page_size=10)

        assert len(files) == 3
        assert files[0].id == "file1"
        assert files[0].name == "Document.pdf"
        assert files[1].id == "file2"
        assert files[2].id == "file3"

        mock_aiohttp_session.get.assert_called_once()
        call_args = mock_aiohttp_session.get.call_args
        assert "files" in call_args[0][0]
        assert call_args[1]["params"]["pageSize"] == 10
        assert call_args[1]["headers"]["Authorization"] == f"Bearer {access_token}"

    @pytest.mark.asyncio
    async def test_list_files_empty_response(
        self,
        access_token: str,
        empty_api_response: dict[str, Any],
        mock_aiohttp_session: MagicMock,
        mock_aiohttp_response: AsyncMock,
    ) -> None:
        """Test listing files when Google Drive returns no files."""
        client = GoogleDriveClient(access_token=access_token)

        mock_aiohttp_response.json.return_value = empty_api_response

        with patch("aiohttp.ClientSession", return_value=mock_aiohttp_session):
            files = await client.list_files()

        assert len(files) == 0

    @pytest.mark.asyncio
    async def test_list_files_page_size_validation(
        self,
        access_token: str,
        mock_aiohttp_session: MagicMock,
        mock_aiohttp_response: AsyncMock,
    ) -> None:
        """Test that page_size is clamped to valid range (1-1000)."""
        client = GoogleDriveClient(access_token=access_token)

        mock_aiohttp_response.json.return_value = {"files": []}

        with patch("aiohttp.ClientSession", return_value=mock_aiohttp_session):
            # Test too large
            await client.list_files(page_size=2000)
            assert mock_aiohttp_session.get.call_args[1]["params"]["pageSize"] == 1000

            # Test too small
            await client.list_files(page_size=-5)
            assert mock_aiohttp_session.get.call_args[1]["params"]["pageSize"] == 1

    @pytest.mark.asyncio
    async def test_list_files_api_error(self, access_token: str, mock_error_response: Any) -> None:
        """Test handling of API error responses."""
        client = GoogleDriveClient(access_token=access_token)

        mock_session_cm, error = mock_error_response(403, "Forbidden")

        with patch("aiohttp.ClientSession", return_value=mock_session_cm):
            with pytest.raises(aiohttp.ClientResponseError) as exc_info:
                await client.list_files()

            assert exc_info.value.status == 403


class TestListGoogleDriveItemsTool:
    """Test suite for list_google_drive_items tool function."""

    @pytest.mark.asyncio
    async def test_list_google_drive_items_success(
        self,
        access_token: str,
        sample_api_response: dict[str, Any],
        mock_aiohttp_session: MagicMock,
        mock_aiohttp_response: AsyncMock,
    ) -> None:
        """Test successful listing of Google Drive items through the tool."""
        mock_aiohttp_response.json.return_value = sample_api_response

        with (
            patch(
                "datarobot_genai.drmcp.tools.external.google_drive.get_access_token",
                return_value=access_token,
            ),
            patch("aiohttp.ClientSession", return_value=mock_aiohttp_session),
        ):
            result = await list_google_drive_items(page_size=30)

        assert "file1: Document.pdf (application/pdf)" in result
        assert "file2: Spreadsheet.xlsx" in result
        assert "file3: notes.txt" in result
        assert len(result.split("\n")) == 3

    @pytest.mark.asyncio
    async def test_list_google_drive_items_empty(
        self,
        access_token: str,
        empty_api_response: dict[str, Any],
        mock_aiohttp_session: MagicMock,
        mock_aiohttp_response: AsyncMock,
    ) -> None:
        """Test tool behavior when no Google Drive items are found."""
        mock_aiohttp_response.json.return_value = empty_api_response

        with (
            patch(
                "datarobot_genai.drmcp.tools.external.google_drive.get_access_token",
                return_value=access_token,
            ),
            patch("aiohttp.ClientSession", return_value=mock_aiohttp_session),
        ):
            result = await list_google_drive_items()

        assert result == "No Google Drive items found."

    @pytest.mark.asyncio
    async def test_list_google_drive_items_api_error(
        self, access_token: str, mock_error_response: Any
    ) -> None:
        """Test tool error handling when API returns error status."""
        mock_session_cm, error = mock_error_response(401, "Unauthorized")

        with (
            patch(
                "datarobot_genai.drmcp.tools.external.google_drive.get_access_token",
                return_value=access_token,
            ),
            patch("aiohttp.ClientSession", return_value=mock_session_cm),
        ):
            result = await list_google_drive_items()

        assert "Error accessing Google Drive: 401" in result

    @pytest.mark.asyncio
    async def test_list_google_drive_items_connection_error(self, access_token: str) -> None:
        """Test tool error handling for network connection errors."""
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("Connection failed"))
        mock_session.__aexit__ = AsyncMock()

        with (
            patch(
                "datarobot_genai.drmcp.tools.external.google_drive.get_access_token",
                return_value=access_token,
            ),
            patch("aiohttp.ClientSession", return_value=mock_session),
        ):
            result = await list_google_drive_items()

        assert "Error connecting to Google Drive" in result
        assert "Connection failed" in result
