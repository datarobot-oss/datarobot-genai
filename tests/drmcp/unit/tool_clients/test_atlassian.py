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

import base64
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
import pytest
from datarobot.auth.datarobot.exceptions import OAuthServiceClientErr

from datarobot_genai.drmcputils.auth import set_request_headers
from datarobot_genai.drmcputils.credentials import AuthResolutionStrategy
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.core.clients.atlassian import ATLASSIAN_API_BASE
from datarobot_genai.drtools.core.clients.atlassian import OAUTH_ACCESSIBLE_RESOURCES_PATH
from datarobot_genai.drtools.core.clients.atlassian import TENANT_INFO_PATH
from datarobot_genai.drtools.core.clients.atlassian import AtlassianAuth
from datarobot_genai.drtools.core.clients.atlassian import AtlassianAuthMode
from datarobot_genai.drtools.core.clients.atlassian import _find_first_resource_with_id
from datarobot_genai.drtools.core.clients.atlassian import _find_resource_by_service
from datarobot_genai.drtools.core.clients.atlassian import get_atlassian_cloud_id
from datarobot_genai.drtools.core.clients.atlassian import get_atlassian_cloud_id_from_site
from datarobot_genai.drtools.core.clients.atlassian import get_confluence_access_token
from datarobot_genai.drtools.core.clients.atlassian import get_jira_access_token
from datarobot_genai.drtools.core.clients.atlassian import normalize_atlassian_site_url
from datarobot_genai.drtools.core.clients.atlassian import resolve_config_atlassian_auth


def _mock_creds(
    monkeypatch: pytest.MonkeyPatch,
    *,
    strategy: AuthResolutionStrategy = AuthResolutionStrategy.HTTP,
    api_key: str = "",
    email: str = "",
    site_url: str = "",
) -> MagicMock:
    mock_creds = MagicMock()
    mock_creds.auth_resolution_strategy = strategy
    mock_creds.atlassian_api_token = api_key
    mock_creds.atlassian_email = email
    mock_creds.atlassian_site_url = site_url
    monkeypatch.setattr(
        "datarobot_genai.drtools.core.clients.atlassian.get_credentials",
        lambda: mock_creds,
    )
    return mock_creds


@pytest.mark.parametrize(
    ("get_token", "provider_type", "access_token_header"),
    [
        pytest.param(get_jira_access_token, "jira", "x-datarobot-jira-access-token", id="jira"),
        pytest.param(
            get_confluence_access_token,
            "confluence",
            "x-datarobot-confluence-access-token",
            id="confluence",
        ),
    ],
)
class TestGetAtlassianServiceAccessToken:
    """Test per-service Atlassian access token helpers."""

    @pytest.fixture(autouse=True)
    def _clear_header_ctx(self) -> None:
        set_request_headers({})
        yield
        set_request_headers({})

    @pytest.mark.asyncio
    async def test_get_access_token_success(
        self,
        get_token: object,
        provider_type: str,
        access_token_header: str,
    ) -> None:
        """Test successful OAuth Bearer access token retrieval."""
        mock_token = "test_access_token_123"
        mock_get_access_token = AsyncMock(return_value=mock_token)
        with patch(
            "datarobot_genai.drmcputils.auth.get_access_token",
            mock_get_access_token,
        ):
            result = await get_token()
        assert isinstance(result, AtlassianAuth)
        assert result.mode == AtlassianAuthMode.OAUTH_BEARER
        assert result.token == mock_token
        assert result.authorization_header() == f"Bearer {mock_token}"
        mock_get_access_token.assert_awaited_once_with(provider_type)

    @pytest.mark.asyncio
    async def test_get_access_token_empty_token(
        self,
        get_token: object,
        provider_type: str,
        access_token_header: str,
    ) -> None:
        """Test handling of empty access token without header fallback."""
        with patch(
            "datarobot_genai.drmcputils.auth.get_access_token",
            new_callable=AsyncMock,
            return_value="",
        ):
            with patch("datarobot_genai.drmcputils.auth.get_request_headers", return_value={}):
                result = await get_token()
        assert isinstance(result, ToolError)
        assert result.kind == ToolErrorKind.AUTHENTICATION
        assert "empty access token" in str(result).lower()
        assert access_token_header in str(result)

    @pytest.mark.asyncio
    async def test_get_access_token_oauth_error(
        self,
        get_token: object,
        provider_type: str,
        access_token_header: str,
    ) -> None:
        """Test handling of OAuthServiceClientErr without header fallback."""
        oauth_error = OAuthServiceClientErr("OAuth error occurred")
        with patch(
            "datarobot_genai.drmcputils.auth.get_access_token",
            new_callable=AsyncMock,
            side_effect=oauth_error,
        ):
            with patch("datarobot_genai.drmcputils.auth.get_request_headers", return_value={}):
                result = await get_token()
        assert isinstance(result, ToolError)
        assert "Could not obtain access token" in str(result)
        assert access_token_header in str(result)

    @pytest.mark.asyncio
    async def test_get_access_token_unexpected_error(
        self,
        get_token: object,
        provider_type: str,
        access_token_header: str,
    ) -> None:
        """Test handling of unexpected exceptions."""
        unexpected_error = ValueError("Unexpected error")
        with patch(
            "datarobot_genai.drmcputils.auth.get_access_token",
            new_callable=AsyncMock,
            side_effect=unexpected_error,
        ):
            result = await get_token()
        assert isinstance(result, ToolError)
        assert result.kind == ToolErrorKind.INTERNAL
        assert "unexpected error" in str(result).lower()

    @pytest.mark.asyncio
    async def test_header_fallback_when_oauth_raises(
        self,
        get_token: object,
        provider_type: str,
        access_token_header: str,
    ) -> None:
        """OBO failure is satisfied by the service-specific access-token header."""
        with patch(
            "datarobot_genai.drmcputils.auth.get_access_token",
            new_callable=AsyncMock,
            side_effect=RuntimeError("no auth ctx"),
        ):
            set_request_headers({access_token_header: "from-header"})
            result = await get_token()
        assert isinstance(result, AtlassianAuth)
        assert result.mode == AtlassianAuthMode.OAUTH_BEARER
        assert result.token == "from-header"


class TestResolveConfigAtlassianAuth:
    def test_api_token_basic_when_email_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_creds(
            monkeypatch,
            strategy=AuthResolutionStrategy.CONFIG,
            api_key="atlassian-api-token",
            email="user@example.com",
            site_url="acme.atlassian.net",
        )
        auth = resolve_config_atlassian_auth()
        assert isinstance(auth, AtlassianAuth)
        assert auth.mode == AtlassianAuthMode.API_TOKEN_BASIC
        assert auth.email == "user@example.com"
        assert auth.token == "atlassian-api-token"
        assert auth.site_url == "https://acme.atlassian.net"

    def test_oauth_bearer_when_email_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_creds(
            monkeypatch,
            strategy=AuthResolutionStrategy.CONFIG,
            api_key="oauth-access-token",
        )
        auth = resolve_config_atlassian_auth()
        assert isinstance(auth, AtlassianAuth)
        assert auth.mode == AtlassianAuthMode.OAUTH_BEARER
        assert auth.token == "oauth-access-token"

    def test_missing_api_key_returns_tool_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_creds(monkeypatch, strategy=AuthResolutionStrategy.CONFIG)
        result = resolve_config_atlassian_auth()
        assert isinstance(result, ToolError)
        assert result.kind == ToolErrorKind.AUTHENTICATION

    def test_email_without_site_url_returns_tool_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_creds(
            monkeypatch,
            strategy=AuthResolutionStrategy.CONFIG,
            api_key="atlassian-api-token",
            email="user@example.com",
        )
        result = resolve_config_atlassian_auth()
        assert isinstance(result, ToolError)
        assert "ATLASSIAN_SITE_URL" in str(result)

    @pytest.mark.asyncio
    async def test_config_strategy_uses_api_token_basic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_creds(
            monkeypatch,
            strategy=AuthResolutionStrategy.CONFIG,
            api_key="atlassian-api-token",
            email="user@example.com",
            site_url="https://acme.atlassian.net",
        )
        auth = await get_jira_access_token()
        assert isinstance(auth, AtlassianAuth)
        assert auth.mode == AtlassianAuthMode.API_TOKEN_BASIC
        expected = base64.b64encode(b"user@example.com:atlassian-api-token").decode("ascii")
        assert auth.authorization_header() == f"Basic {expected}"


class TestAtlassianAuthHelpers:
    def test_normalize_site_url_adds_https(self) -> None:
        assert normalize_atlassian_site_url("acme.atlassian.net") == "https://acme.atlassian.net"

    def test_normalize_site_url_strips_trailing_slash(self) -> None:
        assert (
            normalize_atlassian_site_url("https://acme.atlassian.net/")
            == "https://acme.atlassian.net"
        )


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


class TestGetAtlassianCloudIdFromSite:
    @pytest.mark.asyncio
    async def test_resolves_cloud_id_from_tenant_info(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"cloudId": "tenant-cloud-789"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)

        cloud_id = await get_atlassian_cloud_id_from_site(mock_client, "https://acme.atlassian.net")
        assert cloud_id == "tenant-cloud-789"
        mock_client.get.assert_awaited_once_with(f"https://acme.atlassian.net{TENANT_INFO_PATH}")

    @pytest.mark.asyncio
    async def test_tenant_info_401(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=MagicMock(), response=mock_response
        )

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)

        with pytest.raises(ValueError, match="Authentication failed for Atlassian API token"):
            await get_atlassian_cloud_id_from_site(mock_client, "https://acme.atlassian.net")


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
    async def test_get_cloud_id_api_token_uses_tenant_info(self) -> None:
        auth = AtlassianAuth.api_token_basic(
            email="user@example.com",
            api_token="api-token",
            site_url="https://acme.atlassian.net",
        )
        mock_response = MagicMock()
        mock_response.json.return_value = {"cloudId": "tenant-cloud-999"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)

        cloud_id = await get_atlassian_cloud_id(mock_client, service_type="jira", auth=auth)
        assert cloud_id == "tenant-cloud-999"
        mock_client.get.assert_awaited_once_with(f"https://acme.atlassian.net{TENANT_INFO_PATH}")

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

        mock_client.get.assert_awaited_once()
        call_args = mock_client.get.call_args
        assert call_args is not None
        url = call_args[0][0] if call_args[0] else None
        expected_url = f"{ATLASSIAN_API_BASE}{OAUTH_ACCESSIBLE_RESOURCES_PATH}"
        assert url == expected_url
