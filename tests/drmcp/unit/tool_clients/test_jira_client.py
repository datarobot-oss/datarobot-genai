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

from datarobot_genai.drmcp.tools.clients.jira import JiraClient, Issue


def make_response(status_code: int, json_data: dict, cloud_id: str) -> httpx.Response:
    """Create a mock httpx.Response with a request attached."""
    request = httpx.Request("GET", f"https://api.atlassian.com/ex/jira/{cloud_id}")
    return httpx.Response(status_code, json=json_data, request=request)


class TestJiraClient:
    """Test JiraClient class."""

    @pytest.fixture
    def mock_access_token(self) -> str:
        """Mock access token."""
        return "test_access_token_456"

    @pytest.fixture
    def mock_cloud_id(self) -> str:
        """Mock cloud ID."""
        return "test-cloud-id-123"

    @pytest.fixture
    def mock_issue_response(self) -> dict:
        """Mock Jira REST API issue response (simplified)."""
        return {
            "id": "123",
            "key": "PROJ-123",
            "fields": {
                "summary": "Dummy summary",
                "status": {
                    "name": "In Progress",
                },
                "updated": "2025-12-15T07:47:19.176-0500",
                "created": "2025-12-11T09:01:58.944-0500",
                "reporter": {"emailAddress": "dummy@reporter.com"},
                "assignee": {"emailAddress": "dummy@assignee.com"},
            },
        }

    @pytest.mark.asyncio
    async def test_get_issue_by_id_success(
        self, mock_access_token: str, mock_cloud_id: str, mock_issue_response: dict
    ) -> None:
        """Test successfully getting an issue by ID."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.jira.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with JiraClient(mock_access_token) as client:

                async def mock_get(url: str, params: dict | None = None) -> httpx.Response:
                    return make_response(200, mock_issue_response, mock_cloud_id)

                client._client.get = mock_get

                result = await client.get_jira_issue("PROJ-123")

                assert result == Issue(**mock_issue_response)

    @pytest.mark.asyncio
    async def test_get_page_by_id_not_found(
        self, mock_access_token: str, mock_cloud_id: str
    ) -> None:
        """Test getting a page that doesn't exist."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.jira.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with JiraClient(mock_access_token) as client:

                async def mock_get(url: str, params: dict | None = None) -> httpx.Response:
                    return make_response(404, {"message": "Not found"}, mock_cloud_id)

                client._client.get = mock_get

                with pytest.raises(ValueError, match="not found"):
                    await client.get_jira_issue("not existing")
