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

import logging
from http import HTTPStatus
from typing import Any

import httpx
from pydantic import BaseModel
from pydantic import Field

from .atlassian import ATLASSIAN_API_BASE
from .atlassian import get_atlassian_cloud_id

logger = logging.getLogger(__name__)

RESPONSE_JIRA_ISSUE_FIELDS = {
    "id",
    "key",
    "summary",
    "status",
    "reporter",
    "assignee",
    "created",
    "updated",
}
RESPONSE_JIRA_ISSUE_FIELDS_STR = ",".join(RESPONSE_JIRA_ISSUE_FIELDS)


class _IssuePerson(BaseModel):
    email_address: str = Field(alias="emailAddress")


class _IssueStatus(BaseModel):
    name: str


class _IssueFields(BaseModel):
    summary: str
    status: _IssueStatus
    reporter: _IssuePerson
    assignee: _IssuePerson
    created: str
    updated: str


class Issue(BaseModel):
    id: str
    key: str
    fields: _IssueFields

    def as_flat_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "key": self.key,
            "summary": self.fields.summary,
            "reporterEmailAddress": self.fields.reporter.email_address,
            "assigneeEmailAddress": self.fields.assignee.email_address,
            "created": self.fields.created,
            "updated": self.fields.updated,
            "status": self.fields.status.name,
        }


class JiraClient:
    """
    Client for interacting with Jira API using OAuth access token.

    At the moment of creating this client, official Jira SDK is not supporting async.
    """

    def __init__(self, access_token: str) -> None:
        """
        Initialize Jira client with access token.

        Args:
            access_token: OAuth access token for Atlassian API
        """
        self.access_token = access_token
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        self._cloud_id: str | None = None

    async def _get_cloud_id(self) -> str:
        """
        Get the cloud ID for the authenticated Atlassian Jira instance.

        According to Atlassian OAuth 2.0 documentation, API calls should use:
        https://api.atlassian.com/ex/jira/{cloudId}/rest/api/3/...

        Returns
        -------
            Cloud ID string

        Raises
        ------
            ValueError: If cloud ID cannot be retrieved
        """
        if self._cloud_id:
            return self._cloud_id

        self._cloud_id = await get_atlassian_cloud_id(self._client, service_type="jira")
        return self._cloud_id

    async def _get_full_url(self, path: str) -> str:
        """Return URL for Jira API."""
        cloud_id = await self._get_cloud_id()
        return f"{ATLASSIAN_API_BASE}/ex/jira/{cloud_id}/rest/api/3/{path}"

    async def search_jira_issues(self, jql_query: str, max_results: int) -> list[Issue]:
        """
        Search Jira issues using JQL (Jira Query Language).

        Args:
            jql_query: JQL Query
            max_results: Maximum number of issues to return

        Returns
        -------
            List of Jira issues

        Raises
        ------
            httpx.HTTPStatusError: If the API request fails
        """
        url = await self._get_full_url("search/jql")
        response = await self._client.post(
            url,
            json={
                "jql": jql_query,
                "fields": list(RESPONSE_JIRA_ISSUE_FIELDS),
                "maxResults": max_results,
            },
        )

        response.raise_for_status()
        raw_issues = response.json().get("issues", [])
        issues = [Issue(**issue) for issue in raw_issues]
        return issues

    async def get_jira_issue(self, issue_key: str) -> Issue:
        """
        Get a Jira issue by its key.

        Args:
            issue_key: The key of the Jira issue, e.g., 'PROJ-123'

        Returns
        -------
            Jira issue

        Raises
        ------
            httpx.HTTPStatusError: If the API request fails
        """
        url = await self._get_full_url(f"issue/{issue_key}")
        response = await self._client.get(url, params={"fields": RESPONSE_JIRA_ISSUE_FIELDS_STR})

        if response.status_code == HTTPStatus.NOT_FOUND:
            raise ValueError(f"{issue_key} not found")

        response.raise_for_status()
        issue = Issue(**response.json())
        return issue

    async def get_jira_issue_types(self, project_key: str) -> dict[str, str]:
        """
        Get Jira issue types possible for given project.

        Args:
            project_key: The key of the Jira project, e.g., 'PROJ'

        Returns
        -------
            Dictionary where key is the issue type name and value is the issue type ID

        Raises
        ------
            httpx.HTTPStatusError: If the API request fails
        """
        url = await self._get_full_url(f"issue/createmeta/{project_key}/issuetypes")
        response = await self._client.get(url)

        response.raise_for_status()
        jsoned = response.json()
        issue_types = {
            issue_type["untranslatedName"]: issue_type["id"]
            for issue_type in jsoned.get("issueTypes", [])
        }
        return issue_types

    async def create_jira_issue(
        self, project_key: str, summary: str, issue_type_id: str, description: str | None
    ) -> str:
        """
        Create Jira issue.

        Args:
            project_key: The key of the Jira project, e.g., 'PROJ'
            summary: Summary of Jira issue (title), e.g., 'Fix bug abc'
            issue_type_id: ID type of Jira issue, e.g., "1"
            description: Optional description of Jira issue

        Returns
        -------
            Jira issue key

        Raises
        ------
            httpx.HTTPStatusError: If the API request fails
        """
        url = await self._get_full_url("issue")
        payload = {
            "fields": {
                "project": {"key": project_key},
                "summary": summary,
                "issuetype": {"id": issue_type_id},
            }
        }

        if description:
            payload["fields"]["description"] = {
                "content": [
                    {"content": [{"text": description, "type": "text"}], "type": "paragraph"}
                ],
                "type": "doc",
                "version": 1,
            }

        response = await self._client.post(url, json=payload)

        response.raise_for_status()
        jsoned = response.json()
        return jsoned["key"]

    async def update_jira_issue(self, issue_key: str, fields: dict[str, Any]) -> list[str]:
        """
        Update Jira issue.

        Args:
            issue_key: The key of the Jira issue, e.g., 'PROJ-123'
            fields: A dictionary of field names and their new values
                e.g., {'description': 'New content'}

        Returns
        -------
            List of updated fields

        Raises
        ------
            httpx.HTTPStatusError: If the API request fails
        """
        url = await self._get_full_url(f"issue/{issue_key}")
        payload = {"fields": fields}

        response = await self._client.put(url, json=payload)

        response.raise_for_status()
        return list(fields.keys())

    async def get_available_jira_transitions(self, issue_key: str) -> dict[str, str]:
        """
        Get Available Jira Transitions.

        Args:
            issue_key: The key of the Jira issue, e.g., 'PROJ-123'

        Returns
        -------
            Dictionary where key is the transition name and value is the transition ID

        Raises
        ------
            httpx.HTTPStatusError: If the API request fails
        """
        url = await self._get_full_url(f"issue/{issue_key}/transitions")
        response = await self._client.get(url)
        response.raise_for_status()
        jsoned = response.json()
        transitions = {
            transition["name"]: transition["id"] for transition in jsoned.get("transitions", [])
        }
        return transitions

    async def transition_jira_issue(self, issue_key: str, transition_id: str) -> None:
        """
        Transition Jira issue.

        Args:
            issue_key: The key of the Jira issue, e.g., 'PROJ-123'
            transition_id: Id of target transitionm e.g. '123'.
                Can be obtained from `get_available_jira_transitions`.

        Returns
        -------
            Nothing

        Raises
        ------
            httpx.HTTPStatusError: If the API request fails
        """
        url = await self._get_full_url(f"issue/{issue_key}")
        payload = {"transition": {"id": transition_id}}

        response = await self._client.post(url, json=payload)

        response.raise_for_status()

    async def __aenter__(self) -> "JiraClient":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """Async context manager exit."""
        await self._client.aclose()
