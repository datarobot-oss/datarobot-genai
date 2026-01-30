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

from datarobot_genai.drmcp.tools.clients.confluence import ConfluenceComment
from datarobot_genai.drmcp.tools.clients.confluence import ConfluenceError
from datarobot_genai.drmcp.tools.clients.confluence import ConfluencePage
from datarobot_genai.drmcp.tools.clients.confluence import ContentSearchResult
from datarobot_genai.drmcp.tools.confluence.tools import confluence_add_comment
from datarobot_genai.drmcp.tools.confluence.tools import confluence_get_page
from datarobot_genai.drmcp.tools.confluence.tools import confluence_search
from datarobot_genai.drmcp.tools.confluence.tools import confluence_update_page


@pytest.fixture
def get_atlassian_access_token_mock() -> Iterator[None]:
    with patch(
        "datarobot_genai.drmcp.tools.confluence.tools.get_atlassian_access_token",
        return_value="token",
    ):
        yield


@pytest.fixture
def confluence_client_get_page_by_id_mock() -> Iterator[ConfluencePage]:
    with patch(
        "datarobot_genai.drmcp.tools.clients.confluence.ConfluenceClient.get_page_by_id"
    ) as mock:
        page = ConfluencePage(
            page_id="12345",
            title="Test Page",
            space_id="67890",
            space_key="TEST",
            body="<p>Test content</p>",
            version=1,
        )
        mock.return_value = page
        yield page


@pytest.fixture
def confluence_client_get_page_by_title_mock() -> Iterator[ConfluencePage]:
    with patch(
        "datarobot_genai.drmcp.tools.clients.confluence.ConfluenceClient.get_page_by_title"
    ) as mock:
        page = ConfluencePage(
            page_id="12345",
            title="Test Page",
            space_id="67890",
            space_key="TEST",
            body="<p>Test content</p>",
            version=1,
        )
        mock.return_value = page
        yield page


@pytest.fixture
def confluence_client_get_page_error_mock() -> Iterator[None]:
    with patch(
        "datarobot_genai.drmcp.tools.clients.confluence.ConfluenceClient.get_page_by_id"
    ) as mock:
        mock.side_effect = ConfluenceError("Page not found", status_code=404)
        yield


class TestConfluenceGetPage:
    """Confluence get page tool tests."""

    @pytest.mark.asyncio
    async def test_confluence_get_page_by_id_happy_path(
        self,
        get_atlassian_access_token_mock: None,
        confluence_client_get_page_by_id_mock: ConfluencePage,
    ) -> None:
        """Confluence get page by ID -- happy path."""
        page_id = "12345"

        tool_result = await confluence_get_page(page_id_or_title=page_id)

        content, structured_content = tool_result.to_mcp_result()
        expected = {
            "page_id": "12345",
            "title": "Test Page",
            "space_id": "67890",
            "space_key": "TEST",
            "body": "<p>Test content</p>",
            "version": 1,
        }
        assert json.loads(content[0].text) == expected
        assert structured_content == expected

    @pytest.mark.asyncio
    async def test_confluence_get_page_by_title_happy_path(
        self,
        get_atlassian_access_token_mock: None,
        confluence_client_get_page_by_title_mock: ConfluencePage,
    ) -> None:
        """Confluence get page by title -- happy path."""
        page_title = "Test Page"
        space_key = "TEST"

        tool_result = await confluence_get_page(page_id_or_title=page_title, space_key=space_key)

        content, structured_content = tool_result.to_mcp_result()
        assert json.loads(content[0].text)["title"] == "Test Page"
        assert structured_content["title"] == "Test Page"
        assert structured_content["space_key"] == "TEST"

    @pytest.mark.asyncio
    async def test_confluence_get_page_by_title_without_space_key(
        self,
        get_atlassian_access_token_mock: None,
    ) -> None:
        """Confluence get page by title without space_key -- should raise error."""
        page_title = "Test Page"

        with pytest.raises(ToolError, match="space_key.*required"):
            await confluence_get_page(page_id_or_title=page_title)

    @pytest.mark.asyncio
    async def test_confluence_get_page_when_error_in_client(
        self,
        get_atlassian_access_token_mock: None,
        confluence_client_get_page_error_mock: None,
    ) -> None:
        """Confluence get page -- error in client."""
        page_id = "12345"

        with pytest.raises(ToolError, match="Page not found"):
            await confluence_get_page(page_id_or_title=page_id)

    @pytest.mark.asyncio
    async def test_confluence_get_page_empty_page_id(
        self,
        get_atlassian_access_token_mock: None,
    ) -> None:
        """Confluence get page with empty page_id -- should raise error."""
        page_id = ""

        with pytest.raises(ToolError, match="cannot be empty"):
            await confluence_get_page(page_id_or_title=page_id)


@pytest.fixture
def confluence_client_add_comment_mock() -> Iterator[ConfluenceComment]:
    with patch(
        "datarobot_genai.drmcp.tools.clients.confluence.ConfluenceClient.add_comment"
    ) as mock:
        comment = ConfluenceComment(
            comment_id="98765",
            page_id="12345",
            body="<p>Test comment</p>",
        )
        mock.return_value = comment
        yield comment


@pytest.fixture
def confluence_client_add_comment_error_mock() -> Iterator[None]:
    with patch(
        "datarobot_genai.drmcp.tools.clients.confluence.ConfluenceClient.add_comment"
    ) as mock:
        mock.side_effect = ConfluenceError("Page not found", status_code=404)
        yield


class TestConfluenceAddComment:
    """Confluence add comment tool tests."""

    @pytest.mark.asyncio
    async def test_confluence_add_comment_happy_path(
        self,
        get_atlassian_access_token_mock: None,
        confluence_client_add_comment_mock: ConfluenceComment,
    ) -> None:
        """Confluence add comment -- happy path."""
        page_id = "12345"
        comment_body = "<p>Test comment</p>"

        tool_result = await confluence_add_comment(page_id=page_id, comment_body=comment_body)

        content, structured_content = tool_result.to_mcp_result()
        expected = {"comment_id": "98765", "page_id": "12345"}
        assert json.loads(content[0].text) == expected
        assert structured_content == expected

    @pytest.mark.asyncio
    async def test_confluence_add_comment_empty_page_id(
        self,
        get_atlassian_access_token_mock: None,
    ) -> None:
        """Confluence add comment with empty page_id -- should raise error."""
        page_id = ""
        comment_body = "<p>Test comment</p>"

        with pytest.raises(ToolError, match="page_id.*cannot be empty"):
            await confluence_add_comment(page_id=page_id, comment_body=comment_body)

    @pytest.mark.asyncio
    async def test_confluence_add_comment_empty_comment_body(
        self,
        get_atlassian_access_token_mock: None,
    ) -> None:
        """Confluence add comment with empty comment_body -- should raise error."""
        page_id = "12345"
        comment_body = ""

        with pytest.raises(ToolError, match="comment_body.*cannot be empty"):
            await confluence_add_comment(page_id=page_id, comment_body=comment_body)

    @pytest.mark.asyncio
    async def test_confluence_add_comment_when_error_in_client(
        self,
        get_atlassian_access_token_mock: None,
        confluence_client_add_comment_error_mock: None,
    ) -> None:
        """Confluence add comment -- error in client."""
        page_id = "12345"
        comment_body = "<p>Test comment</p>"

        with pytest.raises(ToolError, match="Page not found"):
            await confluence_add_comment(page_id=page_id, comment_body=comment_body)


@pytest.fixture
def confluence_client_search_mock() -> Iterator[list[ContentSearchResult]]:
    with patch(
        "datarobot_genai.drmcp.tools.clients.confluence.ConfluenceClient.search_confluence_content"
    ) as confluence_client_search:
        results = [
            ContentSearchResult(
                id="12345",
                title="Test Page",
                type="page",
                space_key="TEST",
                space_name="Test Space",
                excerpt="<p>Content</p>",
                last_modified="2025-01-06T10:00:00.000Z",
                url="https://example.atlassian.net/wiki/spaces/TEST/pages/12345",
            ),
            ContentSearchResult(
                id="67890",
                title="Another Page",
                type="page",
                space_key="TEST",
                space_name="Test Space",
                excerpt="<p>More content</p>",
                last_modified="2025-01-05T09:00:00.000Z",
                url="https://example.atlassian.net/wiki/spaces/TEST/pages/67890",
            ),
        ]
        confluence_client_search.return_value = results
        yield results


@pytest.fixture
def confluence_client_search_error_mock() -> Iterator[None]:
    with patch(
        "datarobot_genai.drmcp.tools.clients.confluence.ConfluenceClient.search_confluence_content"
    ) as confluence_client_search:
        confluence_client_search.side_effect = ConfluenceError("Search failed", status_code=500)
        yield


class TestConfluenceSearch:
    """Confluence search tool tests."""

    @pytest.mark.asyncio
    async def test_confluence_search_happy_path(
        self,
        get_atlassian_access_token_mock: None,
        confluence_client_search_mock: list[ContentSearchResult],
    ) -> None:
        """Confluence search -- happy path."""
        cql_query = "type=page AND space=TEST"

        tool_result = await confluence_search(cql_query=cql_query)

        content, structured_content = tool_result.to_mcp_result()
        expected = {
            "data": [
                {
                    "id": "12345",
                    "title": "Test Page",
                    "type": "page",
                    "spaceKey": "TEST",
                    "spaceName": "Test Space",
                    "excerpt": "<p>Content</p>",
                    "lastModified": "2025-01-06T10:00:00.000Z",
                    "url": "https://example.atlassian.net/wiki/spaces/TEST/pages/12345",
                },
                {
                    "id": "67890",
                    "title": "Another Page",
                    "type": "page",
                    "spaceKey": "TEST",
                    "spaceName": "Test Space",
                    "excerpt": "<p>More content</p>",
                    "lastModified": "2025-01-05T09:00:00.000Z",
                    "url": "https://example.atlassian.net/wiki/spaces/TEST/pages/67890",
                },
            ],
            "count": 2,
        }
        assert json.loads(content[0].text) == expected
        assert structured_content == expected

    @pytest.mark.asyncio
    async def test_confluence_search_when_error_in_client(
        self, get_atlassian_access_token_mock: None, confluence_client_search_error_mock: None
    ) -> None:
        """Confluence search -- error in client."""
        cql_query = "type=page AND space=TEST"

        with pytest.raises(ToolError, match="Search failed"):
            await confluence_search(cql_query=cql_query)

    @pytest.mark.asyncio
    async def test_confluence_search_empty_query(
        self, get_atlassian_access_token_mock: None
    ) -> None:
        """Confluence search -- empty query validation."""
        with pytest.raises(ToolError, match="cannot be empty"):
            await confluence_search(cql_query="")

    @pytest.mark.asyncio
    async def test_confluence_search_max_results_too_low(
        self, get_atlassian_access_token_mock: None
    ) -> None:
        """Confluence search -- max_results below 1 should raise error."""
        with pytest.raises(ToolError, match="max_results.*must be between 1 and 100"):
            await confluence_search(cql_query="type=page", max_results=0)

    @pytest.mark.asyncio
    async def test_confluence_search_max_results_too_high(
        self, get_atlassian_access_token_mock: None
    ) -> None:
        """Confluence search -- max_results above 100 should raise error."""
        with pytest.raises(ToolError, match="max_results.*must be between 1 and 100"):
            await confluence_search(cql_query="type=page", max_results=101)


@pytest.fixture
def confluence_client_get_page_mock() -> Iterator[ConfluencePage]:
    with patch(
        "datarobot_genai.drmcp.tools.clients.confluence.ConfluenceClient.get_page_by_id"
    ) as mock:
        page = ConfluencePage(
            page_id="12345",
            title="Test Page",
            space_id="67890",
            space_key="TEST",
            body="<p>Full body content from page fetch</p>",
            version=1,
        )
        mock.return_value = page
        yield page


class TestConfluenceSearchIncludeBody:
    """Confluence search with include_body parameter tests."""

    @pytest.mark.asyncio
    async def test_confluence_search_with_include_body(
        self,
        get_atlassian_access_token_mock: None,
        confluence_client_search_mock: list[ContentSearchResult],
        confluence_client_get_page_mock: ConfluencePage,
    ) -> None:
        """Confluence search with include_body=True fetches full page content."""
        cql_query = "type=page AND space=TEST"

        tool_result = await confluence_search(cql_query=cql_query, include_body=True)

        content, structured_content = tool_result.to_mcp_result()
        assert json.loads(content[0].text)["count"] == 2
        # Verify body field is added with full content
        assert structured_content["data"][0]["body"] == "<p>Full body content from page fetch</p>"
        # excerpt should still be there
        assert structured_content["data"][0]["excerpt"] == "<p>Content</p>"

    @pytest.mark.asyncio
    async def test_confluence_search_without_include_body(
        self,
        get_atlassian_access_token_mock: None,
        confluence_client_search_mock: list[ContentSearchResult],
    ) -> None:
        """Confluence search without include_body does not have body field."""
        cql_query = "type=page AND space=TEST"

        tool_result = await confluence_search(cql_query=cql_query, include_body=False)

        content, structured_content = tool_result.to_mcp_result()
        # body field should NOT be present when include_body=False
        assert "body" not in structured_content["data"][0]
        # excerpt should be there
        assert structured_content["data"][0]["excerpt"] == "<p>Content</p>"


@pytest.fixture
def confluence_client_update_page_mock(
    get_atlassian_access_token_mock: None,
) -> Iterator[ConfluencePage]:
    """Mock ConfluenceClient.update_page method."""
    mock_page = ConfluencePage(
        page_id="12345",
        title="Updated Page",
        space_id="67890",
        space_key="TEST",
        body="<p>Updated content</p>",
        version=6,
    )
    with patch(
        "datarobot_genai.drmcp.tools.confluence.tools.ConfluenceClient.update_page",
        return_value=mock_page,
    ):
        yield mock_page


@pytest.fixture
def confluence_client_update_page_error_mock(
    get_atlassian_access_token_mock: None,
) -> Iterator[None]:
    """Mock ConfluenceClient.update_page method to raise ConfluenceError."""
    with patch(
        "datarobot_genai.drmcp.tools.confluence.tools.ConfluenceClient.update_page",
        side_effect=ConfluenceError("Page not found", status_code=404),
    ):
        yield


class TestConfluenceUpdatePage:
    """Confluence update page tool tests."""

    @pytest.mark.asyncio
    async def test_confluence_update_page_happy_path(
        self,
        get_atlassian_access_token_mock: None,
        confluence_client_update_page_mock: ConfluencePage,
    ) -> None:
        """Confluence update page -- happy path."""
        page_id = "12345"
        new_body_content = "<p>Updated content</p>"
        version_number = 5

        tool_result = await confluence_update_page(
            page_id=page_id,
            new_body_content=new_body_content,
            version_number=version_number,
        )

        content, structured_content = tool_result.to_mcp_result()
        expected = {"updated_page_id": "12345", "new_version": 6}
        assert json.loads(content[0].text) == expected
        assert structured_content == expected

    @pytest.mark.asyncio
    async def test_confluence_update_page_empty_page_id(
        self,
        get_atlassian_access_token_mock: None,
    ) -> None:
        """Confluence update page with empty page_id -- should raise error."""
        page_id = ""
        new_body_content = "<p>Updated content</p>"
        version_number = 5

        with pytest.raises(ToolError, match="page_id.*cannot be empty"):
            await confluence_update_page(
                page_id=page_id,
                new_body_content=new_body_content,
                version_number=version_number,
            )

    @pytest.mark.asyncio
    async def test_confluence_update_page_empty_body(
        self,
        get_atlassian_access_token_mock: None,
    ) -> None:
        """Confluence update page with empty body -- should raise error."""
        page_id = "12345"
        new_body_content = ""
        version_number = 5

        with pytest.raises(ToolError, match="new_body_content.*cannot be empty"):
            await confluence_update_page(
                page_id=page_id,
                new_body_content=new_body_content,
                version_number=version_number,
            )

    @pytest.mark.asyncio
    async def test_confluence_update_page_invalid_version(
        self,
        get_atlassian_access_token_mock: None,
    ) -> None:
        """Confluence update page with invalid version -- should raise error."""
        page_id = "12345"
        new_body_content = "<p>Updated content</p>"
        version_number = 0

        with pytest.raises(ToolError, match="version_number.*must be a positive integer"):
            await confluence_update_page(
                page_id=page_id,
                new_body_content=new_body_content,
                version_number=version_number,
            )

    @pytest.mark.asyncio
    async def test_confluence_update_page_when_error_in_client(
        self,
        get_atlassian_access_token_mock: None,
        confluence_client_update_page_error_mock: None,
    ) -> None:
        """Confluence update page -- error in client."""
        page_id = "12345"
        new_body_content = "<p>Updated content</p>"
        version_number = 5

        with pytest.raises(ToolError, match="Page not found"):
            await confluence_update_page(
                page_id=page_id,
                new_body_content=new_body_content,
                version_number=version_number,
            )
