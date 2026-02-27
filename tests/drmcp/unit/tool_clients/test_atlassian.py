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

"""Tests for Atlassian API client utilities."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
import pytest
from datarobot.auth.datarobot.exceptions import OAuthServiceClientErr
from fastmcp.exceptions import ToolError

from datarobot_genai.drtools.clients.atlassian import ATLASSIAN_API_BASE
from datarobot_genai.drtools.clients.atlassian import OAUTH_ACCESSIBLE_RESOURCES_PATH
from datarobot_genai.drtools.clients.atlassian import _find_first_resource_with_id
from datarobot_genai.drtools.clients.atlassian import _find_resource_by_service
from datarobot_genai.drtools.clients.atlassian import get_atlassian_access_token
from datarobot_genai.drtools.clients.atlassian import get_atlassian_cloud_id


class TestGetAtlassianAccessToken:
    """Test get_atlassian_access_token function."""

    @pytest.mark.asyncio
    async def test_get_access_token_success(self) -> None:
        """Test successful access token retrieval."""
        mock_token = "test_access_token_123"
        with patch(
            "datarobot_genai.drtools.clients.atlassian.get_access_token",
            new_callable=AsyncMock,
            return_value=mock_token,
        ):
            result = await get_atlassian_access_token()
            assert result == mock_token
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_get_access_token_empty_token(self) -> None:
        """Test handling of empty access token."""
        with patch(
            "datarobot_genai.drtools.clients.atlassian.get_access_token",
            new_callable=AsyncMock,
            return_value="",
        ):
            result = await get_atlassian_access_token()
            assert isinstance(result, ToolError)
            assert "empty access token" in str(result).lower()

    @pytest.mark.asyncio
    async def test_get_access_token_oauth_error(self) -> None:
        """Test handling of OAuthServiceClientErr."""
        oauth_error = OAuthServiceClientErr("OAuth error occurred")
        with patch(
            "datarobot_genai.drtools.clients.atlassian.get_access_token",
            new_callable=AsyncMock,
            side_effect=oauth_error,
        ):
            result = await get_atlassian_access_token()
            assert isinstance(result, ToolError)
            assert "Could not obtain access token" in str(result)

    @pytest.mark.asyncio
    async def test_get_access_token_unexpected_error(self) -> None:
        """Test handling of unexpected exceptions."""
        unexpected_error = ValueError("Unexpected error")
        with patch(
            "datarobot_genai.drtools.clients.atlassian.get_access_token",
            new_callable=AsyncMock,
            side_effect=unexpected_error,
        ):
            result = await get_atlassian_access_token()
            assert isinstance(result, ToolError)
            assert "unexpected error" in str(result).lower()


class TestFindResourceByService:
    """Test _find_resource_by_service helper function."""

    def test_find_jira_resource(self) -> None:
        """Test finding Jira resource by service type."""
        resources = [
            {"id": "cloud-123", "scopes": ["read:jira-work", "write:jira-work"]},
            {"id": "cloud-456", "scopes": ["read:confluence-content"]},
        ]
        result = _find_resource_by_service(resources, "jira")
        assert result is not None
        assert result["id"] == "cloud-123"

    def test_find_confluence_resource(self) -> None:
        """Test finding Confluence resource by service type."""
        resources = [
            {"id": "cloud-123", "scopes": ["read:jira-work"]},
            {"id": "cloud-456", "scopes": ["read:confluence-content", "write:confluence-content"]},
        ]
        result = _find_resource_by_service(resources, "confluence")
        assert result is not None
        assert result["id"] == "cloud-456"

    def test_find_resource_case_insensitive(self) -> None:
        """Test that service type matching is case insensitive."""
        resources = [
            {"id": "cloud-123", "scopes": ["read:JIRA-work"]},
        ]
        result = _find_resource_by_service(resources, "jira")
        assert result is not None
        assert result["id"] == "cloud-123"

    def test_find_resource_no_match(self) -> None:
        """Test when no matching resource is found."""
        resources = [
            {"id": "cloud-123", "scopes": ["read:jira-work"]},
            {"id": "cloud-456", "scopes": ["read:confluence-content"]},
        ]
        result = _find_resource_by_service(resources, "bitbucket")
        assert result is None

    def test_find_resource_no_id(self) -> None:
        """Test that resources without ID are skipped."""
        resources = [
            {"scopes": ["read:jira-work"]},  # No ID
            {"id": "cloud-123", "scopes": ["read:jira-work"]},
        ]
        result = _find_resource_by_service(resources, "jira")
        assert result is not None
        assert result["id"] == "cloud-123"

    def test_find_resource_empty_scopes(self) -> None:
        """Test handling of resources with empty scopes."""
        resources = [
            {"id": "cloud-123", "scopes": []},
            {"id": "cloud-456", "scopes": ["read:jira-work"]},
        ]
        result = _find_resource_by_service(resources, "jira")
        assert result is not None
        assert result["id"] == "cloud-456"


class TestFindFirstResourceWithId:
    """Test _find_first_resource_with_id helper function."""

    def test_find_first_resource_with_id(self) -> None:
        """Test finding first resource with ID."""
        resources = [
            {"scopes": ["read:jira-work"]},  # No ID
            {"id": "cloud-123", "scopes": ["read:jira-work"]},
            {"id": "cloud-456", "scopes": ["read:confluence-content"]},
        ]
        result = _find_first_resource_with_id(resources)
        assert result is not None
        assert result["id"] == "cloud-123"

    def test_find_first_resource_no_id(self) -> None:
        """Test when no resources have ID."""
        resources = [
            {"scopes": ["read:jira-work"]},
            {"scopes": ["read:confluence-content"]},
        ]
        result = _find_first_resource_with_id(resources)
        assert result is None

    def test_find_first_resource_empty_list(self) -> None:
        """Test with empty resource list."""
        resources = []
        result = _find_first_resource_with_id(resources)
        assert result is None


class TestGetAtlassianCloudId:
    """Test get_atlassian_cloud_id function."""

    @pytest.mark.asyncio
    async def test_get_cloud_id_with_service_type(self) -> None:
        """Test getting cloud ID with service type filter."""
        mock_resources = [
            {"id": "cloud-jira-123", "scopes": ["read:jira-work"]},
            {"id": "cloud-confluence-456", "scopes": ["read:confluence-content"]},
        ]
        mock_response = MagicMock()
        mock_response.json.return_value = mock_resources
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)

        cloud_id = await get_atlassian_cloud_id(mock_client, service_type="jira")
        assert cloud_id == "cloud-jira-123"
        mock_client.get.assert_awaited_once()
        assert ATLASSIAN_API_BASE in str(mock_client.get.call_args)

    @pytest.mark.asyncio
    async def test_get_cloud_id_without_service_type(self) -> None:
        """Test getting cloud ID without service type (fallback)."""
        mock_resources = [
            {"id": "cloud-123", "scopes": ["read:jira-work"]},
        ]
        mock_response = MagicMock()
        mock_response.json.return_value = mock_resources
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)

        cloud_id = await get_atlassian_cloud_id(mock_client)
        assert cloud_id == "cloud-123"

    @pytest.mark.asyncio
    async def test_get_cloud_id_service_type_fallback(self) -> None:
        """Test fallback when service type not found."""
        mock_resources = [
            {"id": "cloud-123", "scopes": ["read:other-service"]},
        ]
        mock_response = MagicMock()
        mock_response.json.return_value = mock_resources
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)

        cloud_id = await get_atlassian_cloud_id(mock_client, service_type="jira")
        assert cloud_id == "cloud-123"  # Falls back to first resource with ID

    @pytest.mark.asyncio
    async def test_get_cloud_id_no_resources(self) -> None:
        """Test error when no resources are returned."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)

        with pytest.raises(ValueError, match="No accessible resources found"):
            await get_atlassian_cloud_id(mock_client)

    @pytest.mark.asyncio
    async def test_get_cloud_id_no_cloud_id_in_resources(self) -> None:
        """Test error when no resource has an ID."""
        mock_resources = [
            {"scopes": ["read:jira-work"]},  # No ID
        ]
        mock_response = MagicMock()
        mock_response.json.return_value = mock_resources
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)

        with pytest.raises(ValueError, match="No cloud ID found"):
            await get_atlassian_cloud_id(mock_client)

    @pytest.mark.asyncio
    async def test_get_cloud_id_401_error(self) -> None:
        """Test handling of 401 authentication error."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=MagicMock(), response=mock_response
        )

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)

        with pytest.raises(ValueError, match="Authentication failed"):
            await get_atlassian_cloud_id(mock_client)

    @pytest.mark.asyncio
    async def test_get_cloud_id_other_http_error(self) -> None:
        """Test handling of other HTTP errors."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Internal Server Error", request=MagicMock(), response=mock_response
        )

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)

        with pytest.raises(ValueError, match="Failed to get Atlassian cloud ID"):
            await get_atlassian_cloud_id(mock_client)

    @pytest.mark.asyncio
    async def test_get_cloud_id_network_error(self) -> None:
        """Test handling of network/request errors."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(
            side_effect=httpx.RequestError("Network error", request=MagicMock())
        )

        with pytest.raises(ValueError, match="Network error"):
            await get_atlassian_cloud_id(mock_client)

    @pytest.mark.asyncio
    async def test_get_cloud_id_url_construction(self) -> None:
        """Test that the correct URL is constructed."""
        mock_resources = [{"id": "cloud-123", "scopes": ["read:jira-work"]}]
        mock_response = MagicMock()
        mock_response.json.return_value = mock_resources
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)

        await get_atlassian_cloud_id(mock_client)

        # Verify the URL was constructed correctly
        mock_client.get.assert_awaited_once()
        call_args = mock_client.get.call_args
        assert call_args is not None
        # httpx.AsyncClient.get() is called with URL as first positional argument
        url = call_args[0][0] if call_args[0] else None
        expected_url = f"{ATLASSIAN_API_BASE}{OAUTH_ACCESSIBLE_RESOURCES_PATH}"
        assert url == expected_url
