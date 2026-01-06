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

from unittest.mock import AsyncMock
from unittest.mock import patch

import httpx
import pytest

from datarobot_genai.drmcp.tools.clients.confluence import ConfluenceClient
from datarobot_genai.drmcp.tools.clients.confluence import ConfluenceComment
from datarobot_genai.drmcp.tools.clients.confluence import ConfluenceError
from datarobot_genai.drmcp.tools.clients.confluence import ConfluencePage


def make_response(
    status_code: int, json_data: dict, cloud_id: str, method: str = "GET"
) -> httpx.Response:
    """Create a mock httpx.Response with a request attached."""
    request = httpx.Request(method, f"https://api.atlassian.com/ex/confluence/{cloud_id}")
    return httpx.Response(status_code, json=json_data, request=request)


class TestConfluenceClient:
    """Test ConfluenceClient class."""

    @pytest.fixture
    def mock_access_token(self) -> str:
        """Mock access token."""
        return "test_access_token_456"

    @pytest.fixture
    def mock_cloud_id(self) -> str:
        """Mock cloud ID."""
        return "test-cloud-id-123"

    @pytest.fixture
    def mock_page_response(self) -> dict:
        """Mock Confluence REST API page response."""
        return {
            "id": "12345",
            "title": "Test Page",
            "space": {"id": "67890", "key": "TEST"},
            "body": {"storage": {"value": "<p>Content</p>"}},
        }

    @pytest.mark.asyncio
    async def test_get_page_by_id_success(
        self, mock_access_token: str, mock_cloud_id: str, mock_page_response: dict
    ) -> None:
        """Test successfully getting a page by ID."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.confluence.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with ConfluenceClient(mock_access_token) as client:

                async def mock_get(_url: str, **_kwargs: dict) -> httpx.Response:
                    return make_response(200, mock_page_response, mock_cloud_id)

                client._client.get = mock_get

                result = await client.get_page_by_id("12345")

                assert result.page_id == "12345"
                assert result.title == "Test Page"
                assert result.space_key == "TEST"
                assert result.body == "<p>Content</p>"

    @pytest.mark.asyncio
    async def test_get_page_by_id_not_found(
        self, mock_access_token: str, mock_cloud_id: str
    ) -> None:
        """Test getting a page that doesn't exist."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.confluence.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with ConfluenceClient(mock_access_token) as client:

                async def mock_get(_url: str, **_kwargs: dict) -> httpx.Response:
                    return make_response(404, {"message": "Not found"}, mock_cloud_id)

                client._client.get = mock_get

                with pytest.raises(ConfluenceError, match="not found") as exc_info:
                    await client.get_page_by_id("nonexistent")
                assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_page_by_title_success(
        self, mock_access_token: str, mock_cloud_id: str, mock_page_response: dict
    ) -> None:
        """Test successfully getting a page by title."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.confluence.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with ConfluenceClient(mock_access_token) as client:

                async def mock_get(_url: str, **_kwargs: dict) -> httpx.Response:
                    return make_response(200, {"results": [mock_page_response]}, mock_cloud_id)

                client._client.get = mock_get

                result = await client.get_page_by_title("Test Page", "TEST")

                assert result.page_id == "12345"
                assert result.title == "Test Page"

    @pytest.mark.asyncio
    async def test_get_page_by_title_not_found(
        self, mock_access_token: str, mock_cloud_id: str
    ) -> None:
        """Test getting a page by title that doesn't exist."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.confluence.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with ConfluenceClient(mock_access_token) as client:

                async def mock_get(_url: str, **_kwargs: dict) -> httpx.Response:
                    return make_response(200, {"results": []}, mock_cloud_id)

                client._client.get = mock_get

                with pytest.raises(ConfluenceError, match="not found") as exc_info:
                    await client.get_page_by_title("Nonexistent", "TEST")
                assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_create_page_success(self, mock_access_token: str, mock_cloud_id: str) -> None:
        """Test successful page creation."""
        mock_create_response = {
            "id": "54321",
            "title": "New Page",
            "space": {"id": "67890", "key": "TEST"},
            "body": {"storage": {"value": "<p>New Content</p>"}},
        }
        with patch(
            "datarobot_genai.drmcp.tools.clients.confluence.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with ConfluenceClient(mock_access_token) as client:

                async def mock_post(_url: str, **_kwargs: dict) -> httpx.Response:
                    return make_response(200, mock_create_response, mock_cloud_id, "POST")

                client._client.post = mock_post

                result = await client.create_page(
                    space_key="TEST",
                    title="New Page",
                    body_content="<p>New Content</p>",
                )

                assert result.page_id == "54321"
                assert result.title == "New Page"
                assert result.space_key == "TEST"

    @pytest.mark.asyncio
    async def test_create_page_with_parent(
        self, mock_access_token: str, mock_cloud_id: str
    ) -> None:
        """Test page creation with parent ID."""
        mock_create_response = {
            "id": "54321",
            "title": "Child Page",
            "space": {"id": "67890", "key": "TEST"},
            "body": {"storage": {"value": "<p>Child Content</p>"}},
        }
        with patch(
            "datarobot_genai.drmcp.tools.clients.confluence.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with ConfluenceClient(mock_access_token) as client:
                captured_kwargs: dict = {}

                async def mock_post(_url: str, **kwargs: dict) -> httpx.Response:
                    captured_kwargs.update(kwargs)
                    return make_response(200, mock_create_response, mock_cloud_id, "POST")

                client._client.post = mock_post

                result = await client.create_page(
                    space_key="TEST",
                    title="Child Page",
                    body_content="<p>Child Content</p>",
                    parent_id=12345,
                )

                assert result.page_id == "54321"
                assert result.title == "Child Page"
                assert captured_kwargs["json"]["ancestors"] == [{"id": 12345}]

    @pytest.mark.asyncio
    async def test_create_page_parent_not_found(
        self, mock_access_token: str, mock_cloud_id: str
    ) -> None:
        """Test page creation when parent page doesn't exist."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.confluence.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with ConfluenceClient(mock_access_token) as client:

                async def mock_post(_url: str, **_kwargs: dict) -> httpx.Response:
                    return make_response(
                        404,
                        {"message": "No ancestor with id 99999 found"},
                        mock_cloud_id,
                        "POST",
                    )

                client._client.post = mock_post

                with pytest.raises(
                    ConfluenceError, match="Parent page with ID '99999' not found"
                ) as exc_info:
                    await client.create_page(
                        space_key="TEST",
                        title="Child Page",
                        body_content="<p>Content</p>",
                        parent_id=99999,
                    )
                assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_create_page_space_not_found(
        self, mock_access_token: str, mock_cloud_id: str
    ) -> None:
        """Test page creation when space doesn't exist."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.confluence.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with ConfluenceClient(mock_access_token) as client:

                async def mock_post(_url: str, **_kwargs: dict) -> httpx.Response:
                    return make_response(
                        404,
                        {"message": "No space with key INVALID found"},
                        mock_cloud_id,
                        "POST",
                    )

                client._client.post = mock_post

                with pytest.raises(ConfluenceError, match="Space 'INVALID' not found") as exc_info:
                    await client.create_page(
                        space_key="INVALID",
                        title="New Page",
                        body_content="<p>Content</p>",
                    )
                assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_create_page_duplicate_title(
        self, mock_access_token: str, mock_cloud_id: str
    ) -> None:
        """Test page creation when title already exists in space."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.confluence.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with ConfluenceClient(mock_access_token) as client:

                async def mock_post(_url: str, **_kwargs: dict) -> httpx.Response:
                    return make_response(
                        409,
                        {"message": "A page with this title already exists"},
                        mock_cloud_id,
                        "POST",
                    )

                client._client.post = mock_post

                with pytest.raises(
                    ConfluenceError, match="already exists in space 'TEST'"
                ) as exc_info:
                    await client.create_page(
                        space_key="TEST",
                        title="Existing Page",
                        body_content="<p>Content</p>",
                    )
                assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_create_page_permission_denied(
        self, mock_access_token: str, mock_cloud_id: str
    ) -> None:
        """Test page creation when user lacks permissions."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.confluence.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with ConfluenceClient(mock_access_token) as client:

                async def mock_post(_url: str, **_kwargs: dict) -> httpx.Response:
                    return make_response(
                        403,
                        {"message": "User does not have permission"},
                        mock_cloud_id,
                        "POST",
                    )

                client._client.post = mock_post

                with pytest.raises(ConfluenceError, match="Permission denied") as exc_info:
                    await client.create_page(
                        space_key="RESTRICTED",
                        title="New Page",
                        body_content="<p>Content</p>",
                    )
                assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_create_page_invalid_content(
        self, mock_access_token: str, mock_cloud_id: str
    ) -> None:
        """Test page creation with invalid storage format content."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.confluence.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with ConfluenceClient(mock_access_token) as client:

                async def mock_post(_url: str, **_kwargs: dict) -> httpx.Response:
                    return make_response(
                        400,
                        {"message": "Invalid storage format: unclosed tag"},
                        mock_cloud_id,
                        "POST",
                    )

                client._client.post = mock_post

                with pytest.raises(ConfluenceError, match="Invalid request") as exc_info:
                    await client.create_page(
                        space_key="TEST",
                        title="New Page",
                        body_content="<p>Unclosed tag",
                    )
                assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_create_page_rate_limited(
        self, mock_access_token: str, mock_cloud_id: str
    ) -> None:
        """Test page creation when rate limited."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.confluence.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with ConfluenceClient(mock_access_token) as client:

                async def mock_post(_url: str, **_kwargs: dict) -> httpx.Response:
                    return make_response(
                        429,
                        {"message": "Rate limit exceeded"},
                        mock_cloud_id,
                        "POST",
                    )

                client._client.post = mock_post

                with pytest.raises(ConfluenceError, match="Rate limit exceeded") as exc_info:
                    await client.create_page(
                        space_key="TEST",
                        title="New Page",
                        body_content="<p>Content</p>",
                    )
                assert exc_info.value.status_code == 429

    def test_as_flat_dict(self) -> None:
        """Test ConfluencePage.as_flat_dict method."""
        page = ConfluencePage(
            page_id="12345",
            title="Test Page",
            space_id="67890",
            space_key="TEST",
            body="<p>Content</p>",
        )

        result = page.as_flat_dict()

        assert result == {
            "page_id": "12345",
            "title": "Test Page",
            "space_id": "67890",
            "space_key": "TEST",
            "body": "<p>Content</p>",
        }

    @pytest.mark.asyncio
    async def test_add_comment_success(self, mock_access_token: str, mock_cloud_id: str) -> None:
        """Test successful comment addition."""
        mock_comment_response = {
            "id": "98765",
            "type": "comment",
            "body": {"storage": {"value": "<p>Test comment</p>"}},
        }
        with patch(
            "datarobot_genai.drmcp.tools.clients.confluence.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with ConfluenceClient(mock_access_token) as client:
                captured_kwargs: dict = {}

                async def mock_post(_url: str, **kwargs: dict) -> httpx.Response:
                    captured_kwargs.update(kwargs)
                    return make_response(200, mock_comment_response, mock_cloud_id, "POST")

                client._client.post = mock_post

                result = await client.add_comment(
                    page_id="12345",
                    comment_body="<p>Test comment</p>",
                )

                assert result.comment_id == "98765"
                assert result.page_id == "12345"
                assert result.body == "<p>Test comment</p>"
                # Verify request payload structure
                assert captured_kwargs["json"]["type"] == "comment"
                assert captured_kwargs["json"]["container"] == {"id": "12345", "type": "page"}
                assert captured_kwargs["json"]["body"]["storage"]["value"] == "<p>Test comment</p>"
                assert captured_kwargs["json"]["body"]["storage"]["representation"] == "storage"

    @pytest.mark.asyncio
    async def test_add_comment_page_not_found(
        self, mock_access_token: str, mock_cloud_id: str
    ) -> None:
        """Test comment addition when page doesn't exist."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.confluence.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with ConfluenceClient(mock_access_token) as client:

                async def mock_post(_url: str, **_kwargs: dict) -> httpx.Response:
                    return make_response(
                        404,
                        {"message": "Content not found"},
                        mock_cloud_id,
                        "POST",
                    )

                client._client.post = mock_post

                with pytest.raises(
                    ConfluenceError, match="Page with ID '99999' not found"
                ) as exc_info:
                    await client.add_comment(
                        page_id="99999",
                        comment_body="<p>Test comment</p>",
                    )
                assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_add_comment_permission_denied(
        self, mock_access_token: str, mock_cloud_id: str
    ) -> None:
        """Test comment addition when user lacks permissions."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.confluence.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with ConfluenceClient(mock_access_token) as client:

                async def mock_post(_url: str, **_kwargs: dict) -> httpx.Response:
                    return make_response(
                        403,
                        {"message": "User does not have permission"},
                        mock_cloud_id,
                        "POST",
                    )

                client._client.post = mock_post

                with pytest.raises(ConfluenceError, match="Permission denied") as exc_info:
                    await client.add_comment(
                        page_id="12345",
                        comment_body="<p>Test comment</p>",
                    )
                assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_add_comment_invalid_content(
        self, mock_access_token: str, mock_cloud_id: str
    ) -> None:
        """Test comment addition with invalid storage format content."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.confluence.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with ConfluenceClient(mock_access_token) as client:

                async def mock_post(_url: str, **_kwargs: dict) -> httpx.Response:
                    return make_response(
                        400,
                        {"message": "Invalid storage format: unclosed tag"},
                        mock_cloud_id,
                        "POST",
                    )

                client._client.post = mock_post

                with pytest.raises(ConfluenceError, match="Invalid request") as exc_info:
                    await client.add_comment(
                        page_id="12345",
                        comment_body="<p>Unclosed tag",
                    )
                assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_add_comment_rate_limited(
        self, mock_access_token: str, mock_cloud_id: str
    ) -> None:
        """Test comment addition when rate limited."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.confluence.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with ConfluenceClient(mock_access_token) as client:

                async def mock_post(_url: str, **_kwargs: dict) -> httpx.Response:
                    return make_response(
                        429,
                        {"message": "Rate limit exceeded"},
                        mock_cloud_id,
                        "POST",
                    )

                client._client.post = mock_post

                with pytest.raises(ConfluenceError, match="Rate limit exceeded") as exc_info:
                    await client.add_comment(
                        page_id="12345",
                        comment_body="<p>Test comment</p>",
                    )
                assert exc_info.value.status_code == 429

    def test_confluence_comment_as_flat_dict(self) -> None:
        """Test ConfluenceComment.as_flat_dict method."""
        comment = ConfluenceComment(
            comment_id="98765",
            page_id="12345",
            body="<p>Test comment</p>",
        )

        result = comment.as_flat_dict()

        assert result == {
            "comment_id": "98765",
            "page_id": "12345",
            "body": "<p>Test comment</p>",
        }
