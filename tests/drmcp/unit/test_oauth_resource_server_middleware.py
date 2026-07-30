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
from http import HTTPStatus
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from datarobot_genai.drmcp.core.constants import MCP_OAUTH_REALM
from datarobot_genai.drmcp.core.constants import MCP_PATH_ENDPOINT
from datarobot_genai.drmcp.core.constants import OAUTH_PROTECTED_RESOURCE_METADATA_ENDPOINT
from datarobot_genai.drmcp.core.oauth_resource_server_middleware import (
    MCPOAuthResourceServerMiddleware,
)
from datarobot_genai.drmcp.core.oauth_resource_server_middleware import _extract_bearer_token
from datarobot_genai.drmcp.core.oauth_resource_server_middleware import (
    create_oauth_resource_server_middleware,
)
from datarobot_genai.drmcp.core.routes_utils import build_oauth_protected_resource_metadata_url
from datarobot_genai.drmcp.core.routes_utils import oauth_protected_resource_metadata_path
from datarobot_genai.drmcp.core.routes_utils import prefix_mount_path


@pytest.fixture
def mock_module_under_test() -> str:
    return "datarobot_genai.drmcp.core.oauth_resource_server_middleware"


@pytest.fixture
def mock_mount_path() -> Iterator[Mock]:
    with patch("datarobot_genai.drmcp.core.routes_utils.get_config") as mock_get_config:
        mock_config = Mock()
        mock_config.mount_path = "/api"
        mock_get_config.return_value = mock_config
        yield mock_config


@pytest.fixture
def oauth_middleware_app(mock_mount_path: Mock) -> Starlette:
    del mock_mount_path

    async def mcp_endpoint(_: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    async def metadata_endpoint(_: Request) -> PlainTextResponse:
        return PlainTextResponse("metadata")

    async def other_endpoint(_: Request) -> PlainTextResponse:
        return PlainTextResponse("other")

    mcp_path = prefix_mount_path(MCP_PATH_ENDPOINT)
    metadata_path = oauth_protected_resource_metadata_path()

    return Starlette(
        routes=[
            Route(mcp_path, mcp_endpoint, methods=["POST", "GET"]),
            Route(metadata_path, metadata_endpoint, methods=["GET"]),
            Route("/api/metadata", other_endpoint, methods=["GET"]),
        ],
        middleware=[Middleware(MCPOAuthResourceServerMiddleware)],
    )


class TestExtractBearerToken:
    @pytest.mark.parametrize(
        ("authorization_header", "expected"),
        [
            ("Bearer token-123", "token-123"),
            ("bearer token-123", "token-123"),
            ("Bearer   token-123", "token-123"),
            ("Basic abc", None),
            ("Bearer", None),
            ("Bearer   ", None),
            ("", None),
        ],
    )
    def test_extract_bearer_token(self, authorization_header: str, expected: str | None) -> None:
        assert _extract_bearer_token(authorization_header) == expected


class TestOAuthProtectedResourceMetadataUrl:
    def test_build_url_uses_prefix_mount_path(self, mock_mount_path: Mock) -> None:
        del mock_mount_path
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": prefix_mount_path(MCP_PATH_ENDPOINT),
                "headers": [],
                "scheme": "https",
                "server": ("example.com", 443),
            }
        )

        expected_path = prefix_mount_path(OAUTH_PROTECTED_RESOURCE_METADATA_ENDPOINT)
        assert build_oauth_protected_resource_metadata_url(request) == (
            f"https://example.com{expected_path}"
        )


class TestMCPOAuthResourceServerMiddleware:
    def test_mcp_request_without_authorization_returns_401(
        self,
        oauth_middleware_app: Starlette,
        mock_mount_path: Mock,
    ) -> None:
        del mock_mount_path
        client = TestClient(oauth_middleware_app)
        mcp_path = prefix_mount_path(MCP_PATH_ENDPOINT)
        metadata_path = prefix_mount_path(OAUTH_PROTECTED_RESOURCE_METADATA_ENDPOINT)

        response = client.post(mcp_path)

        assert response.status_code == HTTPStatus.UNAUTHORIZED
        assert response.json() == {
            "error": "unauthorized",
            "error_description": "Bearer token required",
        }
        assert response.headers["www-authenticate"] == (
            f'Bearer realm="{MCP_OAUTH_REALM}", '
            f'resource_metadata="http://testserver{metadata_path}"'
        )

    def test_mcp_request_with_bearer_token_is_allowed(
        self,
        oauth_middleware_app: Starlette,
        mock_mount_path: Mock,
    ) -> None:
        del mock_mount_path
        client = TestClient(oauth_middleware_app)
        mcp_path = prefix_mount_path(MCP_PATH_ENDPOINT)

        response = client.post(mcp_path, headers={"Authorization": "Bearer token-123"})

        assert response.status_code == HTTPStatus.OK
        assert response.text == "ok"

    def test_non_mcp_request_is_not_challenged(
        self,
        oauth_middleware_app: Starlette,
        mock_mount_path: Mock,
    ) -> None:
        del mock_mount_path
        client = TestClient(oauth_middleware_app)

        response = client.get("/api/metadata")

        assert response.status_code == HTTPStatus.OK
        assert response.text == "other"

    def test_oauth_metadata_request_is_not_challenged(
        self,
        oauth_middleware_app: Starlette,
        mock_mount_path: Mock,
    ) -> None:
        del mock_mount_path
        client = TestClient(oauth_middleware_app)
        metadata_path = oauth_protected_resource_metadata_path()

        response = client.get(metadata_path)

        assert response.status_code == HTTPStatus.OK
        assert response.text == "metadata"


class TestCreateOAuthResourceServerMiddleware:
    def test_returns_none_when_flag_disabled(
        self,
        mock_module_under_test: str,
    ) -> None:
        with patch(f"{mock_module_under_test}.get_config") as mock_get_config:
            mock_get_config.return_value = Mock(
                mcp_oauth_metadata="resource: https://example.com",
                mcp_enable_unauthenticated_well_known_route=False,
            )
            assert create_oauth_resource_server_middleware() is None

    def test_returns_none_when_metadata_not_configured(
        self,
        mock_module_under_test: str,
    ) -> None:
        with patch(f"{mock_module_under_test}.get_config") as mock_get_config:
            mock_get_config.return_value = Mock(
                mcp_oauth_metadata=None,
                mcp_enable_unauthenticated_well_known_route=True,
            )
            with patch(
                f"{mock_module_under_test}.MCPOAuthProtectedResourceMetadataManager"
            ) as mock_manager_cls:
                mock_manager_cls.return_value.get_protected_resource_metadata.return_value = None
                assert create_oauth_resource_server_middleware() is None

    def test_returns_middleware_class_when_metadata_configured(
        self,
        mock_module_under_test: str,
    ) -> None:
        with patch(f"{mock_module_under_test}.get_config") as mock_get_config:
            mock_get_config.return_value = Mock(
                mcp_oauth_metadata="resource: https://example.com",
                mcp_enable_unauthenticated_well_known_route=True,
            )
            with patch(
                f"{mock_module_under_test}.MCPOAuthProtectedResourceMetadataManager"
            ) as mock_manager_cls:
                mock_manager_cls.return_value.get_protected_resource_metadata.return_value = Mock()
                assert create_oauth_resource_server_middleware() is MCPOAuthResourceServerMiddleware
