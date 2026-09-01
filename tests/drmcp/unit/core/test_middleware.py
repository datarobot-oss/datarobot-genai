# Copyright 2025 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from collections.abc import Iterator
from http import HTTPStatus
from typing import Any
from unittest.mock import ANY
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from datarobot_genai.drmcp.core.middleware import GeneralOAuthClaimValidationMiddleware
from datarobot_genai.drmcp.core.middleware import OAuthJWTTokenHandlerMiddleware
from datarobot_genai.drmcp.core.middleware import OAuthMCPToolCallScopeValidationMiddleware
from datarobot_genai.drmcp.core.middleware import build_http_response_from_auth_error
from datarobot_genai.drmcp.core.middleware import is_path_exempt_from_oauth_validation
from datarobot_genai.drmcpbase.auth.exceptions import AudienceClaimValidationError
from datarobot_genai.drmcpbase.auth.exceptions import MCPToolScopeClaimValidationError
from datarobot_genai.drmcpbase.auth.jwt import JWTTokenClaimsValidator
from datarobot_genai.drmcpbase.auth.jwt import JWTTokenHandler
from datarobot_genai.drmcpbase.oauth_protected_resource_metadata.entities import AuthErrorResponse
from datarobot_genai.drmcpbase.oauth_protected_resource_metadata.entities import (
    ErrorCodeInAuthErrorResponse,
)


async def _ok_response(_request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


def mock_app() -> Starlette:
    """Build a Starlette app mirroring selected routes in routes.py."""
    return Starlette(
        routes=[
            Route("/", _ok_response, methods=["GET"]),
            Route("/mcp", _ok_response, methods=["POST"]),
        ],
        middleware=[
            Middleware(OAuthJWTTokenHandlerMiddleware),
            Middleware(GeneralOAuthClaimValidationMiddleware),
        ],
    )


@pytest.fixture
def module_under_test() -> str:
    return "datarobot_genai.drmcp.core.middleware"


@pytest.fixture
def mock_get_user_from_request_scope(module_under_test: str) -> Iterator[Mock]:
    with patch(f"{module_under_test}.get_user_from_request_scope") as mock_func:
        yield mock_func


@pytest.fixture
def mock_jwt_token_claims_validator_cls(module_under_test: str) -> Iterator[Mock]:
    with patch(f"{module_under_test}.JWTTokenClaimsValidator") as mock_cls:
        yield mock_cls


class TestAuthErrorHTTPResponseGeneration:
    @pytest.fixture
    def mock_json_response_cls(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.JSONResponse") as mock_cls:
            yield mock_cls

    def test_build_http_response_from_auth_error(self, mock_json_response_cls: Mock) -> None:
        mock_status_code = Mock()
        mock_error_response = Mock()
        output = build_http_response_from_auth_error(mock_status_code, mock_error_response)

        expected_header_name = mock_error_response.get_header_name.return_value
        expected_header_value = mock_error_response.to_header_value.return_value
        mock_json_response_cls.assert_called_once_with(
            status_code=mock_status_code,
            content=mock_error_response.to_response_content.return_value,
            headers={expected_header_name: expected_header_value},
        )
        assert output == mock_json_response_cls.return_value


class TestPathExemptFromOAuthValidation:
    def test_is_path_exempt_from_oauth_validation_returns_true_if_health_check(self) -> None:
        assert is_path_exempt_from_oauth_validation("/")

    @pytest.mark.parametrize(
        "well_known_sub_path",
        ["oauth-protected-resource", "other-sub-route"],
        ids=str,
    )
    def test_is_path_exempt_from_oauth_validation_returns_true_if_well_known_paths(
        self,
        well_known_sub_path: str,
    ) -> None:
        assert is_path_exempt_from_oauth_validation(f"/.well-known/{well_known_sub_path}")


class TestOAuthJWTTokenHandlerMiddleware:
    @pytest.fixture
    def mock_get_config(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.get_config") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_is_path_exempt_from_oauth_validation(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.is_path_exempt_from_oauth_validation") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_parse_to_access_token(self) -> Iterator[Mock]:
        with patch.object(JWTTokenHandler, "parse_to_access_token") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_to_run_jwt_token_handling(self, module_under_test: str) -> Iterator[Mock]:
        with patch.object(OAuthJWTTokenHandlerMiddleware, "to_run_jwt_token_handling") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_to_run_jwt_token_handling_returns_true(self, module_under_test: str) -> Iterator[Mock]:
        with patch.object(OAuthJWTTokenHandlerMiddleware, "to_run_jwt_token_handling") as mock_func:
            mock_func.return_value = True
            yield mock_func

    @pytest.fixture
    def mock_update_scope_with_auth_credentials(self) -> Iterator[Mock]:
        with patch.object(
            OAuthJWTTokenHandlerMiddleware,
            "update_scope_with_auth_credentials",
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_update_scope_with_authenticated_user(self) -> Iterator[Mock]:
        with patch.object(
            OAuthJWTTokenHandlerMiddleware,
            "update_scope_with_authenticated_user",
        ) as mock_func:
            yield mock_func

    @pytest.mark.parametrize(
        "is_path_exempt_from_validation, is_oauth_validation_enabled, to_run_jwt_token_handling",
        [(True, True, False), (True, False, False), (False, False, False), (False, True, True)],
        ids=str,
    )
    def test_to_run_jwt_token_handling(
        self,
        is_path_exempt_from_validation: bool,
        is_oauth_validation_enabled: bool,
        to_run_jwt_token_handling: bool,
        mock_get_config: Mock,
        mock_is_path_exempt_from_oauth_validation: Mock,
    ) -> None:
        mock_is_path_exempt_from_oauth_validation.return_value = is_path_exempt_from_validation
        mock_config = mock_get_config.return_value
        mock_config.oauth_claim_validation = is_oauth_validation_enabled

        mock_request = Mock()
        output = OAuthJWTTokenHandlerMiddleware.to_run_jwt_token_handling(mock_request)

        assert output is to_run_jwt_token_handling

    def test_bypass_jwt_token_handling(
        self,
        mock_to_run_jwt_token_handling: Mock,
        mock_parse_to_access_token: Mock,
        mock_update_scope_with_auth_credentials: Mock,
        mock_update_scope_with_authenticated_user: Mock,
    ) -> None:
        mock_to_run_jwt_token_handling.return_value = False

        client = TestClient(mock_app())
        response = client.get("/")

        assert response.status_code == HTTPStatus.OK
        mock_parse_to_access_token.assert_not_called()
        mock_update_scope_with_auth_credentials.assert_not_called()
        mock_update_scope_with_authenticated_user.assert_not_called()

    @pytest.mark.usefixtures("mock_to_run_jwt_token_handling_returns_true")
    async def test_run_jwt_token_handling(
        self,
        mock_parse_to_access_token: Mock,
        mock_update_scope_with_auth_credentials: Mock,
        mock_update_scope_with_authenticated_user: Mock,
    ) -> None:
        request = Mock()
        mock_call_next = AsyncMock()
        middleware = OAuthJWTTokenHandlerMiddleware(app=Mock())

        await middleware.dispatch(request, mock_call_next)

        mock_access_token = mock_parse_to_access_token.return_value
        mock_update_scope_with_auth_credentials.assert_called_once_with(
            request.scope, mock_access_token
        )
        mock_update_scope_with_authenticated_user.assert_called_once_with(
            request.scope, mock_access_token
        )
        mock_call_next.assert_called_once_with(request)

    @pytest.mark.usefixtures("mock_to_run_jwt_token_handling_returns_true")
    def test_return_error_when_there_is_no_valid_jwt_token(
        self,
        mock_parse_to_access_token: Mock,
        mock_update_scope_with_auth_credentials: Mock,
        mock_update_scope_with_authenticated_user: Mock,
    ) -> None:
        mock_parse_to_access_token.return_value = None

        client = TestClient(mock_app())
        response = client.get("/")

        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
        mock_parse_to_access_token.assert_called_once_with(
            OAuthJWTTokenHandlerMiddleware.HTTP_HEADER_TO_VALIDATE,
            ANY,
        )
        mock_update_scope_with_auth_credentials.assert_not_called()
        mock_update_scope_with_authenticated_user.assert_not_called()


class TestGeneralOAuthClaimValidationMiddleware:
    @pytest.fixture
    def mock_get_config(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.get_config") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_validate_audience_claim(self) -> Iterator[Mock]:
        with patch.object(JWTTokenClaimsValidator, "validate_audience_claim") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_expected_audience_claim(self) -> Iterator[Mock]:
        with patch.object(
            GeneralOAuthClaimValidationMiddleware,
            "get_expected_audience_claim",
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_build_well_known_protected_resource_url(
        self, module_under_test: str
    ) -> Iterator[Mock]:
        with patch(f"{module_under_test}.build_well_known_protected_resource_url") as mock_func:
            mock_func.return_value = "https://mcp.example.com/.well-known/oauth-protected-resource"
            yield mock_func

    def test_get_expected_audience_claim(self, mock_get_config: Mock) -> None:
        output = GeneralOAuthClaimValidationMiddleware.get_expected_audience_claim()

        mock_get_config.assert_called_once_with()
        assert output == mock_get_config.return_value.mcp_xaa_token_audience

    async def test_skip_claim_validation_if_no_user_to_validate_in_inbound_request(
        self,
        mock_get_user_from_request_scope: Mock,
        mock_jwt_token_claims_validator_cls: Mock,
    ) -> None:
        mock_get_user_from_request_scope.return_value = None

        request = Mock()
        mock_call_next = AsyncMock()
        middleware = GeneralOAuthClaimValidationMiddleware(app=Mock())
        await middleware.dispatch(request, mock_call_next)

        mock_jwt_token_claims_validator_cls.assert_not_called()
        mock_call_next.assert_called_once_with(request)

    async def test_run_claim_validation_succeeds(
        self,
        mock_get_expected_audience_claim: Mock,
        mock_get_user_from_request_scope: Mock,
        mock_jwt_token_claims_validator_cls: Mock,
    ) -> None:
        request = Mock()
        mock_call_next = AsyncMock()
        middleware = GeneralOAuthClaimValidationMiddleware(app=Mock())

        await middleware.dispatch(request, mock_call_next)

        mock_get_user_from_request_scope.assert_called_once_with(request)
        mock_user = mock_get_user_from_request_scope.return_value
        mock_jwt_token_claims_validator_cls.assert_called_once_with(mock_user)
        claim_validator = mock_jwt_token_claims_validator_cls.return_value
        mock_get_expected_audience_claim.assert_called_once_with()
        mock_claims = mock_get_expected_audience_claim.return_value
        claim_validator.validate_audience_claim.assert_called_once_with(mock_claims)
        mock_call_next.assert_called_once_with(request)

    @pytest.mark.usefixtures(
        "mock_get_expected_audience_claim",
        "mock_get_user_from_request_scope",
        "mock_build_well_known_protected_resource_url",
    )
    async def test_run_claim_validation_fails(
        self,
        mock_jwt_token_claims_validator_cls: Mock,
    ) -> None:
        claim_validator = mock_jwt_token_claims_validator_cls.return_value
        error_message = "dsfadaa"
        claim_validator.validate_audience_claim.side_effect = AudienceClaimValidationError(
            error_message
        )

        request = Mock()
        mock_call_next = AsyncMock()
        middleware = GeneralOAuthClaimValidationMiddleware(app=Mock())

        output = await middleware.dispatch(request, mock_call_next)
        assert isinstance(output, JSONResponse)
        assert output.status_code == HTTPStatus.FORBIDDEN
        assert json.loads(output.body) == {
            "error": "invalid_token",
            "error_description": error_message,
        }
        assert output.headers["www-authenticate"] == (
            "Bearer "
            'resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource", '
            'error="invalid_token", '
            f'error_description="{error_message}"'
        )
        mock_call_next.assert_not_called()


class TestOAuthMCPToolCallScopeValidationMiddleware:
    @pytest.fixture
    def mock_json_loads(self) -> Iterator[Mock]:
        with patch.object(json, "loads") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_mcp_tool_name_in_request(self) -> Iterator[AsyncMock]:
        with patch.object(
            OAuthMCPToolCallScopeValidationMiddleware,
            "get_mcp_tool_name_in_request",
        ) as mock_func:
            mock_func.return_value = "mcp_tool_name"
            yield mock_func

    @pytest.fixture
    def mock_declared_scopes_for_one_tool(self, module_under_test: str) -> Iterator[AsyncMock]:
        with patch(f"{module_under_test}.declared_scopes_for_one_tool") as mock_func:
            mock_func.return_value = {"mcp:tools:write"}
            yield mock_func

    @pytest.fixture
    def mock_build_http_response_from_auth_error(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.build_http_response_from_auth_error") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_build_well_known_protected_resource_url(
        self, module_under_test: str
    ) -> Iterator[Mock]:
        with patch(f"{module_under_test}.build_well_known_protected_resource_url") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_request_not_post_method(self) -> Iterator[Mock]:
        yield Mock()

    @pytest.fixture
    def mock_request_body(self) -> dict[str, Any]:
        return {
            "method": "tools/call",
            "params": {"name": "tool_name"},
        }

    @pytest.fixture
    def mock_request(self, mock_request_body: dict[str, Any]) -> Iterator[Mock]:
        _request = Mock()
        _request.method = "POST"
        _request.body = AsyncMock(return_value=mock_request_body)

        yield _request

    async def test_get_mcp_tool_name(
        self,
        mock_request: Mock,
        mock_request_body: dict[str, Any],
        mock_json_loads: Mock,
    ) -> None:
        mock_json_loads.return_value = mock_request_body

        output = await OAuthMCPToolCallScopeValidationMiddleware.get_mcp_tool_name_in_request(
            mock_request
        )

        mock_request.body.assert_called_once_with()
        mock_json_loads.assert_called_once_with(mock_request.body.return_value)
        assert output == mock_request_body["params"]["name"]

    async def test_get_mcp_tool_name_return_none_if_not_a_post_method(
        self,
        mock_request_not_post_method: Mock,
    ) -> None:
        output = await OAuthMCPToolCallScopeValidationMiddleware.get_mcp_tool_name_in_request(
            mock_request_not_post_method
        )

        assert output is None

    @pytest.mark.parametrize(
        "raised_error",
        [
            json.JSONDecodeError("Expecting value", "", 0),
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
        ],
        ids=["JSONDecodeError", "UnicodeDecodeError"],
    )
    async def test_get_mcp_tool_name_return_none_if_request_payload_invalid(
        self,
        raised_error: json.JSONDecodeError | UnicodeDecodeError,
        mock_request: Mock,
        mock_json_loads: Mock,
    ) -> None:
        mock_json_loads.side_effect = raised_error

        output = await OAuthMCPToolCallScopeValidationMiddleware.get_mcp_tool_name_in_request(
            mock_request
        )

        mock_request.body.assert_called_once_with()
        assert output is None

    @pytest.mark.parametrize(
        "json_loads_output",
        [None, {"method": "tools/not_a_call"}, {"not_a_method": "dafa"}],
        ids=str,
    )
    async def test_get_mcp_tool_name_return_none_if_not_mcp_tool_call(
        self,
        json_loads_output: dict[str, Any],
        mock_request: Mock,
        mock_json_loads: Mock,
    ) -> None:
        mock_json_loads.return_value = json_loads_output

        output = await OAuthMCPToolCallScopeValidationMiddleware.get_mcp_tool_name_in_request(
            mock_request,
        )

        assert output is None

    @pytest.mark.parametrize(
        "params_in_request",
        [None, {"not_a_name": "dafa"}],
        ids=str,
    )
    async def test_get_mcp_tool_name_return_none_if_no_name_in_request_payload(
        self,
        params_in_request: dict[str, Any],
        mock_request: Mock,
        mock_json_loads: Mock,
    ) -> None:
        mock_json_loads.return_value = {"params": params_in_request}

        output = await OAuthMCPToolCallScopeValidationMiddleware.get_mcp_tool_name_in_request(
            mock_request,
        )

        assert output is None

    async def test_run_claim_validation_succeeds(
        self,
        mock_get_user_from_request_scope: Mock,
        mock_jwt_token_claims_validator_cls: Mock,
        mock_build_well_known_protected_resource_url: Mock,
        mock_declared_scopes_for_one_tool: Mock,
        mock_get_mcp_tool_name_in_request: Mock,
        mock_request: Mock,
    ) -> None:
        mock_call_next = AsyncMock()
        middleware = OAuthMCPToolCallScopeValidationMiddleware(app=Mock())

        await middleware.dispatch(mock_request, mock_call_next)

        mock_get_user_from_request_scope.assert_called_once_with(mock_request)
        mock_declared_scopes_for_one_tool.assert_called_once_with(
            mock_request.app.state.fastmcp_server,
            mock_get_mcp_tool_name_in_request.return_value,
        )
        mock_get_user_from_request_scope.assert_called_once_with(mock_request)
        mock_user = mock_get_user_from_request_scope.return_value
        mock_jwt_token_claims_validator_cls.assert_called_once_with(mock_user)
        claim_validator = mock_jwt_token_claims_validator_cls.return_value
        claim_validator.validate_mcp_tool_scope_claims.assert_called_once_with(
            mock_declared_scopes_for_one_tool.return_value,
        )

        mock_call_next.assert_called_once_with(mock_request)

    async def test_skip_validation_if_not_a_valid_mcp_tool_call(
        self,
        mock_get_mcp_tool_name_in_request: AsyncMock,
    ) -> None:
        mock_get_mcp_tool_name_in_request.return_value = None

        request = Mock()
        mock_call_next = AsyncMock()
        middleware = OAuthMCPToolCallScopeValidationMiddleware(app=Mock())
        await middleware.dispatch(request, mock_call_next)

        mock_get_mcp_tool_name_in_request.assert_called_once_with(request)
        mock_call_next.assert_called_once_with(request)

    async def test_skip_validation_if_no_registered_scope_found_in_mcp_tool(
        self,
        mock_get_mcp_tool_name_in_request: AsyncMock,
        mock_declared_scopes_for_one_tool: AsyncMock,
    ) -> None:
        mock_declared_scopes_for_one_tool.return_value = None

        request = Mock()
        mock_call_next = AsyncMock()
        middleware = OAuthMCPToolCallScopeValidationMiddleware(app=Mock())
        await middleware.dispatch(request, mock_call_next)

        mock_declared_scopes_for_one_tool.assert_called_once_with(
            request.app.state.fastmcp_server,
            mock_get_mcp_tool_name_in_request.return_value,
        )
        mock_call_next.assert_called_once_with(request)

    @pytest.mark.usefixtures(
        "mock_get_mcp_tool_name_in_request",
        "mock_declared_scopes_for_one_tool",
    )
    async def test_skip_validation_if_no_user_found_in_request_scope(
        self,
        mock_get_user_from_request_scope: Mock,
    ) -> None:
        mock_get_user_from_request_scope.return_value = None

        request = Mock()
        mock_call_next = AsyncMock()
        middleware = OAuthMCPToolCallScopeValidationMiddleware(app=Mock())

        await middleware.dispatch(request, mock_call_next)

        mock_call_next.assert_called_once_with(request)

    @pytest.mark.usefixtures(
        "mock_get_mcp_tool_name_in_request",
    )
    async def test_run_claim_validation_fails(
        self,
        mock_get_user_from_request_scope: Mock,
        mock_jwt_token_claims_validator_cls: Mock,
        mock_build_http_response_from_auth_error: Mock,
        mock_build_well_known_protected_resource_url: Mock,
        mock_declared_scopes_for_one_tool: AsyncMock,
        mock_request: Mock,
    ) -> None:
        claim_validator = mock_jwt_token_claims_validator_cls.return_value
        expected_error_message = "dasfdaw"
        claim_validator.validate_mcp_tool_scope_claims.side_effect = (
            MCPToolScopeClaimValidationError(expected_error_message)
        )
        expected_scopes = ["scope"]
        mock_declared_scopes_for_one_tool.return_value = expected_scopes

        mock_call_next = AsyncMock()
        middleware = OAuthMCPToolCallScopeValidationMiddleware(app=Mock())
        await middleware.dispatch(mock_request, mock_call_next)

        mock_get_user_from_request_scope.assert_called_once_with(mock_request)
        mock_user = mock_get_user_from_request_scope.return_value
        mock_jwt_token_claims_validator_cls.assert_called_once_with(mock_user)
        mock_build_http_response_from_auth_error.assert_called_once_with(
            status_code=HTTPStatus.FORBIDDEN,
            auth_error_response=AuthErrorResponse(
                resource_metadata=mock_build_well_known_protected_resource_url.return_value,
                error_code=ErrorCodeInAuthErrorResponse.INSUFFICIENT_SCOPE,
                error_description=expected_error_message,
                scopes=sorted(expected_scopes),
            ),
        )

        mock_call_next.assert_not_called()
