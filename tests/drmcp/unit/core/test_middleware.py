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
from collections.abc import Iterator
from http import HTTPStatus
from unittest.mock import ANY
from unittest.mock import Mock
from unittest.mock import patch

import jwt
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
from datarobot_genai.drmcp.core.middleware import is_exempt_from_validation
from datarobot_genai.drmcputils.auth import JWTTokenClaimsValidator
from datarobot_genai.drmcputils.exceptions import AudienceClaimValidationError


async def _ok_response(_request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


def mock_app() -> Starlette:
    """Build a Starlette app mirroring selected routes in routes.py."""
    return Starlette(
        routes=[
            Route("/", _ok_response, methods=["GET"]),
            Route("/mcp", _ok_response, methods=["POST"]),
        ],
        middleware=[Middleware(GeneralOAuthClaimValidationMiddleware)],
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


class TestGeneralOAuthClaimValidationMiddleware:
    @pytest.fixture
    def mock_get_config(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.get_config") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_is_exempt_from_validation(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.is_exempt_from_validation") as mock_func:
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

    def test_is_exempt_from_validation_returns_true_if_health_check(self) -> None:
        assert is_exempt_from_validation("/")

    @pytest.mark.parametrize(
        "well_known_sub_path",
        ["oauth-protected-resource", "other-sub-route"],
        ids=str,
    )
    def test_is_exempt_from_validation_returns_true_if_well_known_paths(
        self,
        well_known_sub_path: str,
    ) -> None:
        assert is_exempt_from_validation(f"/.well-known/{well_known_sub_path}")

    def test_bypasses_validation_if_route_is_exempted(
        self,
        mock_is_exempt_from_validation: Mock,
        mock_validate_audience_claim: Mock,
    ) -> None:
        mock_is_exempt_from_validation.return_value = True

        client = TestClient(mock_app())
        response = client.get("/")

        assert response.status_code == HTTPStatus.OK
        mock_validate_audience_claim.assert_not_called()

    def test_run_validation_succeeds(
        self,
        mock_get_expected_audience_claim: Mock,
        mock_jwt_token_claims_validator_cls: Mock,
    ) -> None:
        header_name = "x-datarobot-external-access-token"
        mock_headers = {header_name: "Beaer dsafafd"}
        client = TestClient(mock_app())
        response = client.post("/mcp", headers=mock_headers)

        mock_jwt_token_claims_validator_cls.assert_called_once_with(header_name, ANY)
        mock_validator = mock_jwt_token_claims_validator_cls.return_value
        mock_get_expected_audience_claim.assert_called_once_with()
        mock_validator.validate_audience_claim.assert_called_once_with(
            mock_get_expected_audience_claim.return_value
        )
        assert response.status_code == HTTPStatus.OK

    def test_run_validation_fails_due_to_invalid_aud_claims(
        self,
        mock_get_expected_audience_claim: Mock,
        mock_validate_audience_claim: Mock,
        mock_jwt_token_claims_validator_cls: Mock,
    ) -> None:
        mock_validator = mock_jwt_token_claims_validator_cls.return_value
        mock_validator.validate_audience_claim.side_effect = AudienceClaimValidationError()

        header_name = "x-datarobot-external-access-token"
        mock_headers = {header_name: "Beaer dsafafd"}
        client = TestClient(mock_app())
        response = client.post("/mcp", headers=mock_headers)

        mock_jwt_token_claims_validator_cls.assert_called_once_with(header_name, ANY)
        mock_get_expected_audience_claim.assert_called_once_with()
        mock_validator.validate_audience_claim.assert_called_once_with(
            mock_get_expected_audience_claim.return_value
        )
        assert response.status_code == HTTPStatus.UNAUTHORIZED

    def test_run_validation_fails_due_to_invalid_jwt_token(
        self,
        mock_jwt_token_claims_validator_cls: Mock,
    ) -> None:
        mock_jwt_token_claims_validator_cls.side_effect = jwt.exceptions.PyJWTError()

        header_name = "x-datarobot-external-access-token"
        mock_headers = {header_name: "Beaer invalid_jwt"}
        client = TestClient(mock_app())
        response = client.post("/mcp", headers=mock_headers)

        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
