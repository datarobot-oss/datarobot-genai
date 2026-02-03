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

from collections.abc import Iterator
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

from datarobot_genai.drmcp.tools.clients.microsoft_graph import MicrosoftGraphError
from datarobot_genai.drmcp.tools.clients.microsoft_graph import MicrosoftGraphItem
from datarobot_genai.drmcp.tools.microsoft_graph.tools import microsoft_create_file
from datarobot_genai.drmcp.tools.microsoft_graph.tools import microsoft_graph_search_content
from datarobot_genai.drmcp.tools.microsoft_graph.tools import microsoft_graph_share_item


@pytest.fixture
def get_microsoft_graph_access_token_mock() -> Iterator[None]:
    """Mock Microsoft Graph access token retrieval."""
    with patch(
        "datarobot_genai.drmcp.tools.microsoft_graph.tools.get_microsoft_graph_access_token",
        return_value="test_token",
    ):
        yield


@pytest.fixture
def mock_graph_items() -> list[MicrosoftGraphItem]:
    """Mock Microsoft Graph items."""
    return [
        MicrosoftGraphItem(
            id="item1",
            name="document.docx",
            web_url="https://example.sharepoint.com/document.docx",
            size=2048,
            created_datetime="2025-01-01T00:00:00Z",
            last_modified_datetime="2025-01-10T12:00:00Z",
            is_folder=False,
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            drive_id="drive1",
            parent_folder_id="parent1",
        ),
        MicrosoftGraphItem(
            id="item2",
            name="My Folder",
            web_url="https://example.sharepoint.com/folder",
            is_folder=True,
            drive_id="drive1",
            parent_folder_id="parent1",
        ),
    ]


@pytest.fixture
def mock_client_search_success(mock_graph_items: list[MicrosoftGraphItem]) -> Iterator[AsyncMock]:
    """Mock successful client search."""
    with patch(
        "datarobot_genai.drmcp.tools.microsoft_graph.tools.MicrosoftGraphClient"
    ) as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.search_content = AsyncMock(return_value=mock_graph_items)
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_client_share_item_success() -> Iterator[None]:
    """Mock successful client share item method."""
    with patch("datarobot_genai.drmcp.tools.microsoft_graph.tools.MicrosoftGraphClient.share_item"):
        yield


class TestMicrosoftGraphSearchContent:
    """Test microsoft_graph_search_content tool."""

    @pytest.mark.asyncio
    async def test_search_content_success(
        self,
        get_microsoft_graph_access_token_mock: None,
        mock_client_search_success: AsyncMock,
        mock_graph_items: list[MicrosoftGraphItem],
    ) -> None:
        """Test successful content search."""
        result = await microsoft_graph_search_content(search_query="test query")

        assert result.structured_content["query"] == "test query"
        assert result.structured_content["count"] == 2
        assert len(result.structured_content["results"]) == 2
        assert result.structured_content["results"][0]["id"] == "item1"
        assert result.structured_content["results"][0]["name"] == "document.docx"
        assert result.structured_content["results"][1]["id"] == "item2"
        assert result.structured_content["results"][1]["isFolder"] is True

    @pytest.mark.asyncio
    async def test_search_content_with_site_url(
        self,
        get_microsoft_graph_access_token_mock: None,
        mock_client_search_success: AsyncMock,
    ) -> None:
        """Test search with site URL."""
        site_url = "https://tenant.sharepoint.com/sites/sitename"
        result = await microsoft_graph_search_content(search_query="test", site_url=site_url)

        assert result.structured_content["siteUrl"] == site_url
        mock_client_search_success.search_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_content_with_site_id(
        self,
        get_microsoft_graph_access_token_mock: None,
        mock_client_search_success: AsyncMock,
    ) -> None:
        """Test search with site ID."""
        site_id = "site123"
        result = await microsoft_graph_search_content(search_query="test", site_id=site_id)

        assert result.structured_content["siteId"] == site_id
        call_kwargs = mock_client_search_success.search_content.call_args[1]
        assert call_kwargs["site_id"] == site_id

    @pytest.mark.asyncio
    async def test_search_content_with_pagination(
        self,
        get_microsoft_graph_access_token_mock: None,
        mock_client_search_success: AsyncMock,
    ) -> None:
        """Test search with pagination parameters."""
        result = await microsoft_graph_search_content(search_query="test", from_offset=50, size=100)

        assert result.structured_content["from"] == 50
        assert result.structured_content["size"] == 100
        call_kwargs = mock_client_search_success.search_content.call_args[1]
        assert call_kwargs["from_offset"] == 50
        assert call_kwargs["size"] == 100

    @pytest.mark.asyncio
    async def test_search_content_with_filters(
        self,
        get_microsoft_graph_access_token_mock: None,
        mock_client_search_success: AsyncMock,
    ) -> None:
        """Test search with filters."""
        filters = ["fileType:docx", "size>1000"]
        await microsoft_graph_search_content(search_query="test", filters=filters)

        call_kwargs = mock_client_search_success.search_content.call_args[1]
        assert call_kwargs["filters"] == filters

    @pytest.mark.asyncio
    async def test_search_content_with_entity_types(
        self,
        get_microsoft_graph_access_token_mock: None,
        mock_client_search_success: AsyncMock,
    ) -> None:
        """Test search with entity types."""
        entity_types = ["site", "list"]
        await microsoft_graph_search_content(search_query="test", entity_types=entity_types)

        call_kwargs = mock_client_search_success.search_content.call_args[1]
        assert call_kwargs["entity_types"] == entity_types

    @pytest.mark.asyncio
    async def test_search_content_empty_query(
        self,
        get_microsoft_graph_access_token_mock: None,
    ) -> None:
        """Test search with empty query raises error."""
        with pytest.raises(ToolError) as exc_info:
            await microsoft_graph_search_content(search_query="")
        assert "cannot be empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_search_content_invalid_site_url(
        self,
        get_microsoft_graph_access_token_mock: None,
    ) -> None:
        """Test search with invalid site URL raises error."""
        with pytest.raises(ToolError) as exc_info:
            await microsoft_graph_search_content(search_query="test", site_url="invalid-url")
        assert "invalid" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_search_content_oauth_error(self) -> None:
        """Test search when OAuth token retrieval fails."""
        with patch(
            "datarobot_genai.drmcp.tools.microsoft_graph.tools.get_microsoft_graph_access_token",
            return_value=ToolError("OAuth error"),
        ):
            with pytest.raises(ToolError) as exc_info:
                await microsoft_graph_search_content(search_query="test")
            assert "oauth" in str(exc_info.value).lower() or "error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_search_content_client_error(
        self,
        get_microsoft_graph_access_token_mock: None,
    ) -> None:
        """Test search when client raises error."""
        with patch(
            "datarobot_genai.drmcp.tools.microsoft_graph.tools.MicrosoftGraphClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.search_content = AsyncMock(side_effect=MicrosoftGraphError("Client error"))
            mock_client_class.return_value = mock_client

            with pytest.raises(ToolError) as exc_info:
                await microsoft_graph_search_content(search_query="test")
            assert "client error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_search_content_unexpected_error(
        self,
        get_microsoft_graph_access_token_mock: None,
    ) -> None:
        """Test search when unexpected error occurs."""
        with patch(
            "datarobot_genai.drmcp.tools.microsoft_graph.tools.MicrosoftGraphClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.search_content = AsyncMock(side_effect=Exception("Unexpected error"))
            mock_client_class.return_value = mock_client

            with pytest.raises(ToolError) as exc_info:
                await microsoft_graph_search_content(search_query="test")
            assert "unexpected error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_search_content_with_include_hidden(
        self,
        get_microsoft_graph_access_token_mock: None,
        mock_client_search_success: AsyncMock,
    ) -> None:
        """Test search with include_hidden_content parameter."""
        await microsoft_graph_search_content(search_query="test", include_hidden_content=True)

        call_kwargs = mock_client_search_success.search_content.call_args[1]
        assert call_kwargs["include_hidden_content"] is True

    @pytest.mark.asyncio
    async def test_search_content_with_region(
        self,
        get_microsoft_graph_access_token_mock: None,
        mock_client_search_success: AsyncMock,
    ) -> None:
        """Test search with region parameter."""
        await microsoft_graph_search_content(search_query="test", region="NAM")

        call_kwargs = mock_client_search_success.search_content.call_args[1]
        assert call_kwargs["region"] == "NAM"


class TestMicrosoftGraphShareItem:
    """Test microsoft_graph_share_item tool."""

    @pytest.mark.asyncio
    async def test_share_item_success(
        self,
        get_microsoft_graph_access_token_mock: None,
        mock_client_share_item_success: AsyncMock,
    ) -> None:
        """Test successful share item."""
        result = await microsoft_graph_share_item(
            file_id="dummy_file_id",
            document_library_id="dummy_document_library_id",
            recipient_emails=["dummy@user.com", "dummy2@user.com"],
            role="read",
        )

        assert result.structured_content["fileId"] == "dummy_file_id"
        assert result.structured_content["documentLibraryId"] == "dummy_document_library_id"
        assert result.structured_content["recipientEmails"] == ["dummy@user.com", "dummy2@user.com"]
        assert result.structured_content["role"] == "read"
        assert result.structured_content["n"] == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "function_kwargs,error_message",
        [
            (
                {
                    "file_id": "",
                    "document_library_id": "dummy_document_library_id",
                    "recipient_emails": ["dummy@user.com"],
                    "role": "read",
                },
                "file_id.*cannot be empty",
            ),
            (
                {
                    "file_id": "dummy_file_id",
                    "document_library_id": "",
                    "recipient_emails": ["dummy@user.com"],
                    "role": "read",
                },
                "document_library_id.*cannot be empty",
            ),
            (
                {
                    "file_id": "dummy_file_id",
                    "document_library_id": "dummy_document_library_id",
                    "recipient_emails": [],
                    "role": "read",
                },
                "you must provide at least one 'recipient'",
            ),
        ],
    )
    async def test_share_item_input_validation(
        self,
        get_microsoft_graph_access_token_mock: None,
        function_kwargs: dict,
        error_message: str,
    ) -> None:
        """Test share item -- input validation."""
        with pytest.raises(ToolError, match=error_message):
            await microsoft_graph_share_item(**function_kwargs)

    @pytest.mark.asyncio
    async def test_share_item_oauth_error(self) -> None:
        """Test share item when OAuth token retrieval fails."""
        with patch(
            "datarobot_genai.drmcp.tools.microsoft_graph.tools.get_microsoft_graph_access_token",
            return_value=ToolError("OAuth error"),
        ):
            with pytest.raises(ToolError) as exc_info:
                await microsoft_graph_share_item(
                    file_id="dummy_file_id",
                    document_library_id="dummy_document_library_id",
                    recipient_emails=["dummy@user.com", "dummy2@user.com"],
                    role="read",
                )
            assert "oauth" in str(exc_info.value).lower() or "error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_share_item_client_error(
        self,
        get_microsoft_graph_access_token_mock: None,
    ) -> None:
        """Test share item when client raises error."""
        with patch(
            "datarobot_genai.drmcp.tools.microsoft_graph.tools.MicrosoftGraphClient.share_item"
        ) as mock_fn:
            mock_fn.side_effect = MicrosoftGraphError("Client error")

            with pytest.raises(ToolError) as exc_info:
                await microsoft_graph_share_item(
                    file_id="dummy_file_id",
                    document_library_id="dummy_document_library_id",
                    recipient_emails=["dummy@user.com", "dummy2@user.com"],
                    role="read",
                )
            assert "client error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_share_item_unexpected_error(
        self,
        get_microsoft_graph_access_token_mock: None,
    ) -> None:
        """Test share item when unexpected error occurs."""
        with patch(
            "datarobot_genai.drmcp.tools.microsoft_graph.tools.MicrosoftGraphClient.share_item"
        ) as mock_fn:
            mock_fn.side_effect = Exception("Unexpected error")

            with pytest.raises(ToolError) as exc_info:
                await microsoft_graph_share_item(
                    file_id="dummy_file_id",
                    document_library_id="dummy_document_library_id",
                    recipient_emails=["dummy@user.com", "dummy2@user.com"],
                    role="read",
                )
            assert "unexpected error" in str(exc_info.value).lower()


class TestMicrosoftCreateFile:
    """Test microsoft_create_file tool."""

    @pytest.fixture
    def mock_created_file(self) -> MicrosoftGraphItem:
        """Mock created file."""
        return MicrosoftGraphItem(
            id="new_file_123",
            name="report.txt",
            web_url="https://example.sharepoint.com/sites/test/Shared%20Documents/report.txt",
            size=25,
            created_datetime="2025-01-14T12:00:00Z",
            last_modified_datetime="2025-01-14T12:00:00Z",
            is_folder=False,
            mime_type="text/plain",
            drive_id="drive123",
            parent_folder_id="root",
        )

    @pytest.fixture
    def mock_client_create_success(
        self, mock_created_file: MicrosoftGraphItem
    ) -> Iterator[AsyncMock]:
        """Mock successful client file creation with OneDrive support."""
        with patch(
            "datarobot_genai.drmcp.tools.microsoft_graph.tools.MicrosoftGraphClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.create_file = AsyncMock(return_value=mock_created_file)
            mock_client.get_personal_drive_id = AsyncMock(return_value="personal_drive_123")
            mock_client_class.return_value = mock_client
            yield mock_client

    @pytest.mark.asyncio
    async def test_create_file_sharepoint(
        self,
        get_microsoft_graph_access_token_mock: None,
        mock_client_create_success: AsyncMock,
    ) -> None:
        """Test file creation in SharePoint with explicit document_library_id."""
        result = await microsoft_create_file(
            file_name="report.txt",
            content_text="Content",
            document_library_id="drive123",
        )

        assert result.structured_content["destination"] == "sharepoint"
        assert result.structured_content["driveId"] == "drive123"

        call_kwargs = mock_client_create_success.create_file.call_args[1]
        assert call_kwargs["drive_id"] == "drive123"

    @pytest.mark.asyncio
    async def test_create_file_onedrive_auto(
        self,
        get_microsoft_graph_access_token_mock: None,
        mock_client_create_success: AsyncMock,
    ) -> None:
        """Test file creation in personal OneDrive when no library specified."""
        result = await microsoft_create_file(
            file_name="notes.txt",
            content_text="My notes",
        )

        assert result.structured_content["destination"] == "onedrive"
        assert result.structured_content["driveId"] == "personal_drive_123"

        mock_client_create_success.get_personal_drive_id.assert_called_once()
        call_kwargs = mock_client_create_success.create_file.call_args[1]
        assert call_kwargs["drive_id"] == "personal_drive_123"

    @pytest.mark.asyncio
    async def test_create_file_validation_errors(
        self,
        get_microsoft_graph_access_token_mock: None,
    ) -> None:
        """Test validation errors for missing required fields."""
        with pytest.raises(ToolError, match="file_name is required"):
            await microsoft_create_file(file_name="", content_text="Content")

        with pytest.raises(ToolError, match="content_text is required"):
            await microsoft_create_file(file_name="test.txt", content_text="")

    @pytest.mark.asyncio
    async def test_create_file_client_error(
        self,
        get_microsoft_graph_access_token_mock: None,
    ) -> None:
        """Test error handling for client exceptions."""
        with patch(
            "datarobot_genai.drmcp.tools.microsoft_graph.tools.MicrosoftGraphClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.create_file = AsyncMock(
                side_effect=MicrosoftGraphError("Permission denied")
            )
            mock_client_class.return_value = mock_client

            with pytest.raises(ToolError, match="[Pp]ermission denied"):
                await microsoft_create_file(
                    file_name="report.txt",
                    content_text="Content",
                    document_library_id="drive123",
                )
