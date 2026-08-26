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

from datarobot_genai.drmcp.core.middleware import ErrorResponse
from datarobot_genai.drmcp.core.middleware import GeneralOAuthClaimValidationMiddleware
from datarobot_genai.drmcp.core.middleware import OAuthJWTTokenHandlerMiddleware
from datarobot_genai.drmcp.core.middleware import is_path_exempt_from_oauth_validation
from datarobot_genai.drmcpbase.auth.exceptions import AudienceClaimValidationError
from datarobot_genai.drmcpbase.auth.jwt import JWTTokenClaimsValidator
from datarobot_genai.drmcpbase.auth.jwt import JWTTokenHandler


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


class TestErrorResponse:
    def test_unauthorized(self) -> None:
        expected = JSONResponse(
            status_code=HTTPStatus.UNAUTHORIZED,
            content={"detail": "Audience claim validation failed."},
        )
        actual = ErrorResponse.INVALID_OAUTH_AUDIENCE_CLAIM.to_starlette_response()
        assert (expected.body, expected.status_code) == (actual.body, actual.status_code)


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
    def mock_jwt_token_claims_validator_cls(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.JWTTokenClaimsValidator") as mock_cls:
            yield mock_cls

    @pytest.fixture
    def mock_validate_audience_claim(self) -> Iterator[Mock]:
        with patch.object(JWTTokenClaimsValidator, "validate_audience_claim") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_user_from_request_scope(self) -> Iterator[Mock]:
        with patch.object(
            GeneralOAuthClaimValidationMiddleware,
            "get_user_from_request_scope",
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_expected_audience_claim(self) -> Iterator[Mock]:
        with patch.object(
            GeneralOAuthClaimValidationMiddleware,
            "get_expected_audience_claim",
        ) as mock_func:
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
        assert output.status_code == HTTPStatus.UNAUTHORIZED
        assert json.loads(output.body) == {"detail": error_message}
        mock_call_next.assert_not_called()
