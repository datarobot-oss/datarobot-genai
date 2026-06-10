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

from datarobot_genai.drtools.core.clients.atlassian import AtlassianAuth
from datarobot_genai.drtools.core.clients.jira import Issue
from datarobot_genai.drtools.core.clients.jira import JiraClient


def make_response(status_code: int, json_data: dict | None, cloud_id: str) -> httpx.Response:
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

    @pytest.fixture
    def mock_issue_type_response(self) -> dict:
        """Mock Jira REST API issue response (simplified)."""
        return {
            "startAt": 0,
            "maxResults": 50,
            "total": 10,
            "issueTypes": [
                {
                    "id": "1",
                    "untranslatedName": "Bug",
                },
                {
                    "id": "2",
                    "untranslatedName": "Story",
                },
            ],
        }

    @pytest.fixture
    def mock_issue_create_response(self) -> dict:
        return {"id": "625846", "key": "PROJ-123"}

    @pytest.mark.asyncio
    async def test_search_issues_success(
        self, mock_access_token: str, mock_cloud_id: str, mock_issue_response: dict
    ) -> None:
        """Test successfully searching issues."""
        with patch(
            "datarobot_genai.drtools.core.clients.jira.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with JiraClient(mock_access_token) as client:

                async def mock_post(url: str, json: dict | None = None) -> httpx.Response:
                    return make_response(200, {"issues": [mock_issue_response]}, mock_cloud_id)

                client._client.post = mock_post

                result = await client.search_jira_issues(
                    jql_query="issuetype = Story AND project = PROJ AND summary ~ Dummy",
                    max_results=50,
                )

                assert result == [Issue(**mock_issue_response)]

    @pytest.mark.asyncio
    async def test_get_issue_success(
        self, mock_access_token: str, mock_cloud_id: str, mock_issue_response: dict
    ) -> None:
        """Test successfully getting an issue."""
        with patch(
            "datarobot_genai.drtools.core.clients.jira.get_atlassian_cloud_id",
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
    async def test_get_issue_not_found(self, mock_access_token: str, mock_cloud_id: str) -> None:
        """Test getting an issue that doesn't exist."""
        with patch(
            "datarobot_genai.drtools.core.clients.jira.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with JiraClient(mock_access_token) as client:

                async def mock_get(url: str, params: dict | None = None) -> httpx.Response:
                    return make_response(404, {"message": "Not found"}, mock_cloud_id)

                client._client.get = mock_get

                with pytest.raises(ValueError, match="not found"):
                    await client.get_jira_issue("not existing")

    @pytest.mark.asyncio
    async def test_get_issue_type_success(
        self, mock_access_token: str, mock_cloud_id: str, mock_issue_type_response: dict
    ) -> None:
        """Test successfully getting an issue type."""
        with patch(
            "datarobot_genai.drtools.core.clients.jira.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with JiraClient(mock_access_token) as client:

                async def mock_get(url: str, params: dict | None = None) -> httpx.Response:
                    return make_response(200, mock_issue_type_response, mock_cloud_id)

                client._client.get = mock_get

                result = await client.get_jira_issue_types("PROJ")

                assert result == {"Bug": "1", "Story": "2"}

    @pytest.mark.asyncio
    async def test_create_issue_success(
        self, mock_access_token: str, mock_cloud_id: str, mock_issue_create_response: dict
    ) -> None:
        """Test successfully creating an issue."""
        with patch(
            "datarobot_genai.drtools.core.clients.jira.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with JiraClient(mock_access_token) as client:

                async def mock_post(url: str, json: dict | None = None) -> httpx.Response:
                    return make_response(200, mock_issue_create_response, mock_cloud_id)

                client._client.post = mock_post

                result = await client.create_jira_issue(
                    project_key="PROJ",
                    summary="Dummy summary",
                    issue_type_id="2",
                    description="Dummy issue description",
                )

                assert result == "PROJ-123"

    @pytest.mark.asyncio
    async def test_update_issue_success(self, mock_access_token: str, mock_cloud_id: str) -> None:
        """Test successfully updating an issue."""
        with patch(
            "datarobot_genai.drtools.core.clients.jira.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with JiraClient(mock_access_token) as client:

                async def mock_put(url: str, json: dict) -> httpx.Response:
                    return make_response(204, None, mock_cloud_id)

                client._client.put = mock_put

                result = await client.update_jira_issue(
                    issue_key="PROJ-123", fields={"summary": "Dummy summary"}
                )

                assert result == ["summary"]

    @pytest.mark.asyncio
    async def test_api_token_basic_auth_headers_and_cloud_id_path(
        self, mock_cloud_id: str, mock_issue_response: dict
    ) -> None:
        """API token Basic auth uses tenant_info cloud ID and Basic Authorization header."""
        import base64

        auth = AtlassianAuth.api_token_basic(
            email="user@example.com",
            api_token="api-token-secret",
            site_url="https://acme.atlassian.net",
        )
        expected_basic = base64.b64encode(b"user@example.com:api-token-secret").decode("ascii")

        with patch(
            "datarobot_genai.drtools.core.clients.jira.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ) as mock_get_cloud_id:
            async with JiraClient(auth) as client:
                assert client._client.headers["Authorization"] == f"Basic {expected_basic}"

                async def mock_get(url: str, params: dict | None = None) -> httpx.Response:
                    return make_response(200, mock_issue_response, mock_cloud_id)

                client._client.get = mock_get
                await client.get_jira_issue("PROJ-123")

                mock_get_cloud_id.assert_awaited_once()
                assert mock_get_cloud_id.call_args.kwargs["auth"] == auth


class TestIssueModel:
    def _base_fields(self, **overrides: object) -> dict[str, object]:
        fields: dict[str, object] = {
            "summary": "Dummy summary",
            "status": {"name": "In Progress"},
            "updated": "2025-12-15T07:47:19.176-0500",
            "created": "2025-12-11T09:01:58.944-0500",
            "reporter": {"emailAddress": "dummy@reporter.com"},
            "assignee": {"emailAddress": "dummy@assignee.com"},
        }
        fields.update(overrides)
        return fields

    def test_as_flat_dict_uses_display_name_when_email_missing(self) -> None:
        issue = Issue(
            id="123",
            key="PROJ-123",
            fields=self._base_fields(
                reporter={"displayName": "Reporter User"},
                assignee={"displayName": "Assignee User"},
            ),
        )

        flat = issue.as_flat_dict()
        assert flat["reporterEmailAddress"] == "Reporter User"
        assert flat["assigneeEmailAddress"] == "Assignee User"

    def test_as_flat_dict_uses_account_id_when_email_and_display_name_missing(self) -> None:
        issue = Issue(
            id="123",
            key="PROJ-123",
            fields=self._base_fields(
                reporter={"accountId": "reporter-account-id"},
                assignee={"accountId": "assignee-account-id"},
            ),
        )

        flat = issue.as_flat_dict()
        assert flat["reporterEmailAddress"] == "reporter-account-id"
        assert flat["assigneeEmailAddress"] == "assignee-account-id"

    def test_as_flat_dict_allows_unassigned_issue(self) -> None:
        issue = Issue(
            id="123",
            key="PROJ-123",
            fields=self._base_fields(assignee=None),
        )

        flat = issue.as_flat_dict()
        assert flat["assigneeEmailAddress"] is None
