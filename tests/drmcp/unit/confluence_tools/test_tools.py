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
from datarobot_genai.drmcp.tools.clients.confluence import ConfluenceComment
from datarobot_genai.drmcp.tools.clients.confluence import ConfluenceError
from datarobot_genai.drmcp.tools.clients.confluence import ConfluencePage
from datarobot_genai.drmcp.tools.confluence.tools import confluence_add_comment
from datarobot_genai.drmcp.tools.confluence.tools import confluence_get_page


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
        assert content[0].text == "Successfully retrieved page 'Test Page'."
        assert structured_content == {
            "page_id": "12345",
            "title": "Test Page",
            "space_id": "67890",
            "space_key": "TEST",
            "body": "<p>Test content</p>",
        }

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
        assert content[0].text == "Successfully retrieved page 'Test Page'."
        assert structured_content["title"] == "Test Page"
        assert structured_content["space_key"] == "TEST"

    @pytest.mark.asyncio
    async def test_confluence_get_page_by_title_without_space_key(
        self,
        get_atlassian_access_token_mock: None,
    ) -> None:
        """Confluence get page by title without space_key -- should raise error."""
        page_title = "Test Page"

        with pytest.raises(MCPError, match="space_key.*required"):
            await confluence_get_page(page_id_or_title=page_title)

    @pytest.mark.asyncio
    async def test_confluence_get_page_when_error_in_client(
        self,
        get_atlassian_access_token_mock: None,
        confluence_client_get_page_error_mock: None,
    ) -> None:
        """Confluence get page -- error in client."""
        page_id = "12345"

        with pytest.raises(MCPError, match="Page not found"):
            await confluence_get_page(page_id_or_title=page_id)

    @pytest.mark.asyncio
    async def test_confluence_get_page_empty_page_id(
        self,
        get_atlassian_access_token_mock: None,
    ) -> None:
        """Confluence get page with empty page_id -- should raise error."""
        page_id = ""

        with pytest.raises(MCPError, match="cannot be empty"):
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
        assert content[0].text == "Comment added successfully to page ID 12345."
        assert structured_content == {
            "comment_id": "98765",
            "page_id": "12345",
        }

    @pytest.mark.asyncio
    async def test_confluence_add_comment_empty_page_id(
        self,
        get_atlassian_access_token_mock: None,
    ) -> None:
        """Confluence add comment with empty page_id -- should raise error."""
        page_id = ""
        comment_body = "<p>Test comment</p>"

        with pytest.raises(MCPError, match="page_id.*cannot be empty"):
            await confluence_add_comment(page_id=page_id, comment_body=comment_body)

    @pytest.mark.asyncio
    async def test_confluence_add_comment_empty_comment_body(
        self,
        get_atlassian_access_token_mock: None,
    ) -> None:
        """Confluence add comment with empty comment_body -- should raise error."""
        page_id = "12345"
        comment_body = ""

        with pytest.raises(MCPError, match="comment_body.*cannot be empty"):
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

        with pytest.raises(MCPError, match="Page not found"):
            await confluence_add_comment(page_id=page_id, comment_body=comment_body)
