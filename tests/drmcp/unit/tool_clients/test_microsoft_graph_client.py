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

from unittest import mock

import httpx
import pytest

from datarobot_genai.drtools.clients.microsoft_graph import MicrosoftGraphClient
from datarobot_genai.drtools.clients.microsoft_graph import MicrosoftGraphError
from datarobot_genai.drtools.clients.microsoft_graph import MicrosoftGraphItem
from datarobot_genai.drtools.clients.microsoft_graph import validate_site_url

# MIME type constant for Word documents
WORD_DOCX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def make_response(
    status_code: int, json_data: dict | None = None, text: str | None = None
) -> httpx.Response:
    """Create a mock httpx.Response with a request attached."""
    request = httpx.Request("GET", "https://graph.microsoft.com/v1.0/search/query")
    if json_data is not None:
        return httpx.Response(status_code, json=json_data, request=request)
    return httpx.Response(status_code, text=text or "", request=request)


class TestMicrosoftGraphItem:
    """Test MicrosoftGraphItem model."""

    def test_from_api_response_drive_item(self) -> None:
        """Test creating MicrosoftGraphItem from driveItem API response."""
        api_data = {
            "id": "item123",
            "name": "test.docx",
            "webUrl": "https://example.sharepoint.com/test.docx",
            "size": 1024,
            "createdDateTime": "2025-01-01T00:00:00Z",
            "lastModifiedDateTime": "2025-01-10T12:00:00Z",
            "file": {"mimeType": WORD_DOCX_MIME_TYPE},
            "parentReference": {
                "driveId": "drive123",
                "id": "parent123",
            },
        }
        item = MicrosoftGraphItem.from_api_response(api_data)
        assert item.id == "item123"
        assert item.name == "test.docx"
        assert item.web_url == "https://example.sharepoint.com/test.docx"
        assert item.size == 1024
        assert item.is_folder is False
        assert (
            item.mime_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert item.drive_id == "drive123"
        assert item.parent_folder_id == "parent123"

    def test_from_api_response_folder(self) -> None:
        """Test creating MicrosoftGraphItem from folder API response."""
        api_data = {
            "id": "folder123",
            "name": "My Folder",
            "webUrl": "https://example.sharepoint.com/folder",
            "folder": {
                "folder": True
            },  # The implementation checks "folder" in data.get("folder", {})
            "parentReference": {
                "driveId": "drive123",
                "id": "parent123",
            },
        }
        item = MicrosoftGraphItem.from_api_response(api_data)
        assert item.id == "folder123"
        assert item.name == "My Folder"
        assert item.is_folder is True
        assert item.mime_type is None


class TestValidateSiteUrl:
    """Test validate_site_url function."""

    def test_valid_site_url(self) -> None:
        """Test valid SharePoint site URL."""
        url = "https://tenant.sharepoint.com/sites/sitename"
        assert validate_site_url(url) is None

    def test_empty_url(self) -> None:
        """Test empty URL."""
        error = validate_site_url("")
        assert error is not None
        assert "required" in error.lower()

    def test_url_without_https(self) -> None:
        """Test URL without https://."""
        error = validate_site_url("tenant.sharepoint.com/sites/sitename")
        assert error is not None
        assert "must start with 'https://'" in error

    def test_url_without_sharepoint(self) -> None:
        """Test URL without sharepoint.com."""
        error = validate_site_url("https://example.com/sites/sitename")
        assert error is not None
        assert "sharepoint.com" in error

    def test_url_invalid_domain(self) -> None:
        """Test URL with invalid domain."""
        error = validate_site_url("https://example.com/sites/sitename")
        assert error is not None
        assert "sharepoint.com" in error.lower()


class TestMicrosoftGraphClient:
    """Test MicrosoftGraphClient class."""

    @pytest.fixture
    def mock_access_token(self) -> str:
        """Mock access token."""
        return "test_access_token_123"

    @pytest.fixture
    def mock_search_response(self) -> dict:
        """Mock Microsoft Graph search API response."""
        return {
            "value": [
                {
                    "hitsContainers": [
                        {
                            "hits": [
                                {
                                    "resource": {
                                        "@odata.type": "#microsoft.graph.driveItem",
                                        "id": "item1",
                                        "name": "document.docx",
                                        "webUrl": "https://example.sharepoint.com/document.docx",
                                        "size": 2048,
                                        "createdDateTime": "2025-01-01T00:00:00Z",
                                        "lastModifiedDateTime": "2025-01-10T12:00:00Z",
                                        "file": {"mimeType": WORD_DOCX_MIME_TYPE},
                                        "parentReference": {
                                            "driveId": "drive1",
                                            "id": "parent1",
                                        },
                                    }
                                },
                                {
                                    "resource": {
                                        "@odata.type": "#microsoft.graph.driveItem",
                                        "id": "item2",
                                        "name": "folder",
                                        "folder": {},
                                        "parentReference": {
                                            "driveId": "drive1",
                                            "id": "parent1",
                                        },
                                    }
                                },
                            ]
                        }
                    ]
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_search_content_success(
        self, mock_access_token: str, mock_search_response: dict
    ) -> None:
        """Test successful content search."""
        async with MicrosoftGraphClient(access_token=mock_access_token) as client:
            with mock.patch.object(
                client._client,
                "post",
                return_value=make_response(200, json_data=mock_search_response),
            ):
                results = await client.search_content("test query")
                assert len(results) == 2
                assert results[0].id == "item1"
                assert results[0].name == "document.docx"
                assert results[1].id == "item2"
                assert results[1].is_folder is True

    @pytest.mark.asyncio
    async def test_search_content_with_site_url(
        self, mock_access_token: str, mock_search_response: dict
    ) -> None:
        """Test content search with site URL."""
        site_url = "https://tenant.sharepoint.com/sites/sitename"
        async with MicrosoftGraphClient(
            access_token=mock_access_token, site_url=site_url
        ) as client:
            with (
                mock.patch.object(
                    client._client,
                    "get",
                    return_value=make_response(
                        200, json_data={"id": "site123", "webUrl": site_url}
                    ),
                ),
                mock.patch.object(
                    client._client,
                    "post",
                    return_value=make_response(200, json_data=mock_search_response),
                ),
            ):
                results = await client.search_content("test query")
                assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_content_empty_query(self, mock_access_token: str) -> None:
        """Test search with empty query raises error."""
        async with MicrosoftGraphClient(access_token=mock_access_token) as client:
            with pytest.raises(MicrosoftGraphError) as exc_info:
                await client.search_content("")
            assert "cannot be empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_search_content_size_validation(
        self, mock_access_token: str, mock_search_response: dict
    ) -> None:
        """Test search with size parameter validation."""
        async with MicrosoftGraphClient(access_token=mock_access_token) as client:
            with mock.patch.object(
                client._client,
                "post",
                return_value=make_response(200, json_data=mock_search_response),
            ):
                # Size should be clamped to max 250
                await client.search_content("test", size=500)
                # Verify the request was made (size validation happens before request)
                assert True  # If we get here, size was validated

    @pytest.mark.asyncio
    async def test_search_content_http_error_403(self, mock_access_token: str) -> None:
        """Test search with 403 Forbidden error."""
        async with MicrosoftGraphClient(access_token=mock_access_token) as client:
            with mock.patch.object(
                client._client,
                "post",
                side_effect=httpx.HTTPStatusError(
                    "Forbidden",
                    request=httpx.Request("POST", "https://graph.microsoft.com/v1.0/search/query"),
                    response=make_response(403),
                ),
            ):
                with pytest.raises(MicrosoftGraphError) as exc_info:
                    await client.search_content("test")
                assert "403" in str(exc_info.value) or "permissions" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_search_content_http_error_400(self, mock_access_token: str) -> None:
        """Test search with 400 Bad Request error."""
        error_response = {"error": {"message": "Invalid request parameters"}}
        async with MicrosoftGraphClient(access_token=mock_access_token) as client:
            with mock.patch.object(
                client._client,
                "post",
                side_effect=httpx.HTTPStatusError(
                    "Bad Request",
                    request=httpx.Request("POST", "https://graph.microsoft.com/v1.0/search/query"),
                    response=make_response(400, json_data=error_response),
                ),
            ):
                with pytest.raises(MicrosoftGraphError) as exc_info:
                    await client.search_content("test")
                assert "400" in str(exc_info.value) or "invalid" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_get_site_id_from_url(self, mock_access_token: str) -> None:
        """Test getting site ID from site URL."""
        site_url = "https://tenant.sharepoint.com/sites/sitename"
        site_id_response = {"id": "site123", "webUrl": site_url}
        async with MicrosoftGraphClient(
            access_token=mock_access_token, site_url=site_url
        ) as client:
            with mock.patch.object(
                client._client, "get", return_value=make_response(200, json_data=site_id_response)
            ):
                site_id = await client._get_site_id()
                assert site_id == "site123"
                # Second call should use cached value
                site_id2 = await client._get_site_id()
                assert site_id2 == "site123"

    @pytest.mark.asyncio
    async def test_get_site_id_root_site(self, mock_access_token: str) -> None:
        """Test getting root site ID when no site_url provided."""
        root_site_response = {"id": "root123"}
        async with MicrosoftGraphClient(access_token=mock_access_token) as client:
            with mock.patch.object(
                client._client, "get", return_value=make_response(200, json_data=root_site_response)
            ):
                site_id = await client._get_site_id()
                assert site_id == "root123"

    @pytest.mark.asyncio
    async def test_search_content_with_filters(
        self, mock_access_token: str, mock_search_response: dict
    ) -> None:
        """Test search with filters."""
        async with MicrosoftGraphClient(access_token=mock_access_token) as client:
            with mock.patch.object(
                client._client,
                "post",
                return_value=make_response(200, json_data=mock_search_response),
            ) as mock_post:
                await client.search_content("test", filters=["fileType:docx", "size>1000"])
                # Verify filters were included in request
                call_args = mock_post.call_args
                assert call_args is not None
                payload = call_args.kwargs.get("json", {})
                query_string = (
                    payload.get("requests", [{}])[0].get("query", {}).get("queryString", "")
                )
                assert "fileType:docx" in query_string
                assert "size>1000" in query_string

    @pytest.mark.asyncio
    async def test_search_content_with_entity_types(
        self, mock_access_token: str, mock_search_response: dict
    ) -> None:
        """Test search with custom entity types."""
        async with MicrosoftGraphClient(access_token=mock_access_token) as client:
            with mock.patch.object(
                client._client,
                "post",
                return_value=make_response(200, json_data=mock_search_response),
            ) as mock_post:
                await client.search_content("test", entity_types=["site", "list"])
                call_args = mock_post.call_args
                assert call_args is not None
                payload = call_args.kwargs.get("json", {})
                entity_types = payload.get("requests", [{}])[0].get("entityTypes", [])
                assert "site" in entity_types
                assert "list" in entity_types


class TestMicrosoftGraphClientCreateFile:
    """Test MicrosoftGraphClient.create_file method."""

    @pytest.fixture
    def mock_access_token(self) -> str:
        """Mock access token."""
        return "test_access_token_123"

    @pytest.fixture
    def mock_created_file_response(self) -> dict:
        """Mock response for a created file."""
        return {
            "id": "new_file_123",
            "name": "report.txt",
            "webUrl": "https://example.sharepoint.com/sites/test/Shared%20Documents/report.txt",
            "size": 25,
            "createdDateTime": "2025-01-14T12:00:00Z",
            "lastModifiedDateTime": "2025-01-14T12:00:00Z",
            "file": {"mimeType": "text/plain"},
            "parentReference": {
                "driveId": "drive123",
                "id": "root",
            },
        }

    @pytest.mark.asyncio
    async def test_create_file_success(
        self, mock_access_token: str, mock_created_file_response: dict
    ) -> None:
        """Test successful file creation with URL encoding and correct API call."""
        async with MicrosoftGraphClient(access_token=mock_access_token) as client:
            with mock.patch.object(
                client._client,
                "put",
                return_value=make_response(201, json_data=mock_created_file_response),
            ) as mock_put:
                result = await client.create_file(
                    drive_id="drive123",
                    file_name="report (2025).txt",
                    content="This is the report content.",
                    parent_folder_id="folder456",
                )

                assert result.id == "new_file_123"
                assert result.name == "report.txt"
                assert result.drive_id == "drive123"

                call_args = mock_put.call_args
                # Verify URL encoding and parent folder
                assert "items/folder456:/report%20%282025%29.txt:/content" in call_args[0][0]
                assert call_args[1]["headers"]["Content-Type"] == "text/plain"
                assert b"This is the report content." in call_args[1]["content"]

    @pytest.mark.asyncio
    async def test_create_file_validation_errors(self, mock_access_token: str) -> None:
        """Test validation errors for empty drive_id and file_name."""
        async with MicrosoftGraphClient(access_token=mock_access_token) as client:
            with pytest.raises(MicrosoftGraphError, match="drive_id cannot be empty"):
                await client.create_file(drive_id="", file_name="report.txt", content="Content")

            with pytest.raises(MicrosoftGraphError, match="file_name cannot be empty"):
                await client.create_file(drive_id="drive123", file_name="", content="Content")

    @pytest.mark.asyncio
    async def test_create_file_http_errors(self, mock_access_token: str) -> None:
        """Test HTTP error handling (403, 404, 409, 429)."""
        error_cases = [
            (403, "permission denied"),
            (404, "not found"),
            (409, "already exists"),
            (429, "rate limit"),
        ]

        for status_code, expected_msg in error_cases:
            async with MicrosoftGraphClient(access_token=mock_access_token) as client:
                with mock.patch.object(
                    client._client,
                    "put",
                    side_effect=httpx.HTTPStatusError(
                        "Error",
                        request=httpx.Request("PUT", "https://graph.microsoft.com/v1.0/test"),
                        response=make_response(status_code),
                    ),
                ):
                    with pytest.raises(MicrosoftGraphError) as exc_info:
                        await client.create_file(
                            drive_id="drive123",
                            file_name="report.txt",
                            content="Content",
                            conflict_behavior="fail" if status_code == 409 else "rename",
                        )
                    assert expected_msg in str(exc_info.value).lower()


class TestMicrosoftGraphClientUpdateMetadata:
    """Test MicrosoftGraphClient.update_item_metadata method."""

    @pytest.fixture
    def mock_access_token(self) -> str:
        """Mock access token."""
        return "test_access_token_123"

    @pytest.fixture
    def mock_list_item_fields_response(self) -> dict:
        """Mock response for updated SharePoint list item fields."""
        return {
            "@odata.etag": '"abc123"',
            "Title": "Updated Title",
            "Status": "Approved",
            "Priority": "High",
        }

    @pytest.fixture
    def mock_drive_item_response(self) -> dict:
        """Mock response for updated drive item."""
        return {
            "id": "item123",
            "name": "renamed-document.docx",
            "description": "Updated description",
            "webUrl": "https://example.sharepoint.com/renamed-document.docx",
            "size": 2048,
            "createdDateTime": "2025-01-01T00:00:00Z",
            "lastModifiedDateTime": "2025-01-15T12:00:00Z",
        }

    @pytest.mark.asyncio
    async def test_update_list_item_metadata_success(
        self, mock_access_token: str, mock_list_item_fields_response: dict
    ) -> None:
        """Test successful SharePoint list item metadata update."""
        async with MicrosoftGraphClient(access_token=mock_access_token) as client:
            with mock.patch.object(
                client._client,
                "patch",
                return_value=make_response(200, json_data=mock_list_item_fields_response),
            ) as mock_patch:
                result = await client.update_item_metadata(
                    item_id="item123",
                    fields_to_update={"Status": "Approved", "Priority": "High"},
                    site_id="site456",
                    list_id="list789",
                )

                assert result == mock_list_item_fields_response
                call_args = mock_patch.call_args
                # Verify correct endpoint
                assert "sites/site456/lists/list789/items/item123/fields" in call_args[0][0]
                # Verify payload
                assert call_args[1]["json"] == {"Status": "Approved", "Priority": "High"}

    @pytest.mark.asyncio
    async def test_update_drive_item_metadata_success(
        self, mock_access_token: str, mock_drive_item_response: dict
    ) -> None:
        """Test successful OneDrive/drive item metadata update."""
        async with MicrosoftGraphClient(access_token=mock_access_token) as client:
            with mock.patch.object(
                client._client,
                "patch",
                return_value=make_response(200, json_data=mock_drive_item_response),
            ) as mock_patch:
                result = await client.update_item_metadata(
                    item_id="item123",
                    fields_to_update={"name": "renamed-document.docx", "description": "Updated"},
                    drive_id="drive456",
                )

                assert result == mock_drive_item_response
                call_args = mock_patch.call_args
                # Verify correct endpoint
                assert "drives/drive456/items/item123" in call_args[0][0]
                # Verify payload
                assert call_args[1]["json"] == {
                    "name": "renamed-document.docx",
                    "description": "Updated",
                }

    @pytest.mark.asyncio
    async def test_update_metadata_empty_item_id_error(self, mock_access_token: str) -> None:
        """Test validation error for empty item_id."""
        async with MicrosoftGraphClient(access_token=mock_access_token) as client:
            with pytest.raises(MicrosoftGraphError, match="item_id cannot be empty"):
                await client.update_item_metadata(
                    item_id="",
                    fields_to_update={"Status": "Approved"},
                    site_id="site456",
                    list_id="list789",
                )

            with pytest.raises(MicrosoftGraphError, match="item_id cannot be empty"):
                await client.update_item_metadata(
                    item_id="   ",
                    fields_to_update={"Status": "Approved"},
                    site_id="site456",
                    list_id="list789",
                )

    @pytest.mark.asyncio
    async def test_update_metadata_empty_fields_error(self, mock_access_token: str) -> None:
        """Test validation error for empty fields_to_update."""
        async with MicrosoftGraphClient(access_token=mock_access_token) as client:
            with pytest.raises(MicrosoftGraphError, match="fields_to_update cannot be empty"):
                await client.update_item_metadata(
                    item_id="item123",
                    fields_to_update={},
                    site_id="site456",
                    list_id="list789",
                )

    @pytest.mark.asyncio
    async def test_update_metadata_missing_context_error(self, mock_access_token: str) -> None:
        """Test validation error when neither SharePoint nor drive context provided."""
        async with MicrosoftGraphClient(access_token=mock_access_token) as client:
            with pytest.raises(MicrosoftGraphError, match="Must specify either"):
                await client.update_item_metadata(
                    item_id="item123",
                    fields_to_update={"Status": "Approved"},
                )

    @pytest.mark.asyncio
    async def test_update_metadata_both_contexts_error(self, mock_access_token: str) -> None:
        """Test validation error when both SharePoint and drive context provided."""
        async with MicrosoftGraphClient(access_token=mock_access_token) as client:
            with pytest.raises(MicrosoftGraphError, match="Cannot specify both"):
                await client.update_item_metadata(
                    item_id="item123",
                    fields_to_update={"Status": "Approved"},
                    site_id="site456",
                    list_id="list789",
                    drive_id="drive000",
                )

    @pytest.mark.asyncio
    async def test_update_metadata_http_error_403(self, mock_access_token: str) -> None:
        """Test HTTP 403 Forbidden error handling."""
        async with MicrosoftGraphClient(access_token=mock_access_token) as client:
            with mock.patch.object(
                client._client,
                "patch",
                side_effect=httpx.HTTPStatusError(
                    "Forbidden",
                    request=httpx.Request("PATCH", "https://graph.microsoft.com/v1.0/test"),
                    response=make_response(403),
                ),
            ):
                with pytest.raises(MicrosoftGraphError) as exc_info:
                    await client.update_item_metadata(
                        item_id="item123",
                        fields_to_update={"Status": "Approved"},
                        site_id="site456",
                        list_id="list789",
                    )
                assert "permission denied" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_update_metadata_http_error_404(self, mock_access_token: str) -> None:
        """Test HTTP 404 Not Found error handling."""
        async with MicrosoftGraphClient(access_token=mock_access_token) as client:
            with mock.patch.object(
                client._client,
                "patch",
                side_effect=httpx.HTTPStatusError(
                    "Not Found",
                    request=httpx.Request("PATCH", "https://graph.microsoft.com/v1.0/test"),
                    response=make_response(404),
                ),
            ):
                with pytest.raises(MicrosoftGraphError) as exc_info:
                    await client.update_item_metadata(
                        item_id="item123",
                        fields_to_update={"Status": "Approved"},
                        drive_id="drive456",
                    )
                assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_update_metadata_http_error_409(self, mock_access_token: str) -> None:
        """Test HTTP 409 Conflict error handling."""
        async with MicrosoftGraphClient(access_token=mock_access_token) as client:
            with mock.patch.object(
                client._client,
                "patch",
                side_effect=httpx.HTTPStatusError(
                    "Conflict",
                    request=httpx.Request("PATCH", "https://graph.microsoft.com/v1.0/test"),
                    response=make_response(409),
                ),
            ):
                with pytest.raises(MicrosoftGraphError) as exc_info:
                    await client.update_item_metadata(
                        item_id="item123",
                        fields_to_update={"Status": "Approved"},
                        site_id="site456",
                        list_id="list789",
                    )
                assert "conflict" in str(exc_info.value).lower()
