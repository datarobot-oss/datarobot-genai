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
from unittest.mock import ANY
from unittest.mock import Mock
from unittest.mock import patch

import jwt
import pytest
from fastmcp.server.auth import AccessToken

from datarobot_genai.drmcpbase.auth.exceptions import AudienceClaimValidationError
from datarobot_genai.drmcpbase.auth.exceptions import MCPToolScopeClaimValidationError
from datarobot_genai.drmcpbase.auth.jwt import AuthorizationClaims
from datarobot_genai.drmcpbase.auth.jwt import JWTTokenClaimsValidator
from datarobot_genai.drmcpbase.auth.jwt import JWTTokenHandler


@pytest.fixture
def module_under_test() -> str:
    return "datarobot_genai.drmcpbase.auth.jwt"


class TestAuthorizationClaims:
    def test_from_access_token(self) -> None:
        access_token = AccessToken(
            token="token",
            client_id="client_id",
            scopes=["scope"],
            claims={"aud": "audience"},
        )
        claims = AuthorizationClaims.from_access_token(access_token)
        assert claims.audience == "audience"
        assert claims.scopes == frozenset(["scope"])

    @pytest.mark.parametrize(
        "claim, is_none",
        [({"aud": "dafas"}, False), ({"aud": None}, True), ({}, True)],
        ids=str,
    )
    def test_audience_is_none(self, claim: dict[str, str | None], is_none: bool) -> None:
        access_token = AccessToken(
            token="token",
            client_id="client_id",
            scopes=["scope"],
            claims=claim,
        )
        claims = AuthorizationClaims.from_access_token(access_token)
        assert claims.audience_is_none() is is_none

    @pytest.mark.parametrize(
        "aud_value, is_list",
        [
            (["aud"], True),
            (["aud_1", "aud_2"], True),
            ([None], True),
            ("aud", False),
            (None, False),
        ],
        ids=str,
    )
    def test_audience_is_a_list(self, aud_value: str | list[str] | None, is_list: bool) -> None:
        access_token = AccessToken(
            token="token",
            client_id="client_id",
            scopes=[""],
            claims={"aud": aud_value},
        )
        claims = AuthorizationClaims.from_access_token(access_token)
        assert claims.audience_is_a_list() is is_list

    @pytest.mark.parametrize(
        "aud_value, contain_expected_audience",
        [
            (["expected_aud"], True),
            (["unexpected_aud"], False),
            ("expected_aud", True),
            ("unexpected_aud", False),
        ],
        ids=str,
    )
    def test_contain_expected_audience(
        self, aud_value: str | list[str], contain_expected_audience: bool
    ) -> None:
        access_token = AccessToken(
            token="token",
            client_id="client_id",
            scopes=[""],
            claims={"aud": aud_value},
        )
        claims = AuthorizationClaims.from_access_token(access_token)
        assert claims.contain_expected_audience("expected_aud") is contain_expected_audience

    @pytest.mark.parametrize(
        "scopes, has_scope",
        [(["dsafa"], True), ([], False)],
        ids=str,
    )
    def test_has_scope_claims(self, scopes: list[str], has_scope: bool) -> None:
        access_token = AccessToken(
            token="token",
            client_id="client_id",
            scopes=scopes,
            claims={},
        )
        claims = AuthorizationClaims.from_access_token(access_token)
        assert claims.has_scope_claims() is has_scope

    @pytest.mark.parametrize(
        "expected_scopes, is_subset_of_expected_scopes",
        [
            (frozenset(["expected_scope_one"]), True),
            (frozenset(["expected_scope_one", "expected_scope_two"]), True),
            (frozenset([]), True),
            (frozenset(["expected_scope_unknown"]), False),
        ],
        ids=str,
    )
    def test_are_expected_scopes_subset_of_scope_claims(
        self, expected_scopes: frozenset[str], is_subset_of_expected_scopes
    ) -> None:
        scope_claims = ["expected_scope_one", "expected_scope_two"]
        authorization_claims = AuthorizationClaims(scopes=frozenset(scope_claims))
        assert (
            authorization_claims.are_expected_scopes_subset_of_scope_claims(expected_scopes)
            is is_subset_of_expected_scopes
        )


class TestJWTTokenHandler:
    @pytest.fixture
    def mock_get_bearer_token_header(self) -> Iterator[Mock]:
        with patch.object(JWTTokenHandler, "get_bearer_token_header") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_bearer_token_value(self) -> Iterator[Mock]:
        with patch.object(JWTTokenHandler, "get_bearer_token_value") as mock_func:
            mock_func.return_value = "token"
            yield mock_func

    @pytest.fixture
    def mock_is_jwt_decode(self) -> Iterator[Mock]:
        with patch.object(JWTTokenHandler, "is_jwt_token") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_access_token_cls(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.AccessToken") as mock_cls:
            yield mock_cls

    @pytest.fixture
    def mock_get_jwt_payload_without_signature_verification(
        self,
    ) -> Iterator[Mock]:
        with patch.object(
            JWTTokenHandler, "get_jwt_payload_without_signature_verification"
        ) as mock_func:
            mock_func.return_value = {}
            yield mock_func

    @pytest.fixture
    def mock_extract_scopes(self) -> Iterator[Mock]:
        with patch.object(JWTTokenHandler, "extract_scopes") as mock_func:
            mock_func.return_value = ["scope"]
            yield mock_func

    @pytest.fixture
    def mock_extract_exp(self) -> Iterator[Mock]:
        with patch.object(JWTTokenHandler, "extract_exp") as mock_func:
            mock_func.return_value = 1
            yield mock_func

    @pytest.fixture
    def mock_extract_client_id(self) -> Iterator[Mock]:
        with patch.object(JWTTokenHandler, "extract_client_id") as mock_func:
            mock_func.return_value = "client_id"
            yield mock_func

    @pytest.fixture
    def mock_jwt_get_unverified_header(self) -> Iterator[Mock]:
        with patch.object(jwt, "get_unverified_header") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_jwt_decode(self) -> Iterator[Mock]:
        with patch.object(jwt, "decode") as mock_func:
            yield mock_func

    @pytest.mark.parametrize(
        "token_header_schema",
        ["bearer", "Bearer"],
        ids=str,
    )
    def test_get_bearer_token_value(self, token_header_schema: str) -> None:
        expected_token_value = "adfad adfada"
        token_value = JWTTokenHandler.get_bearer_token_value(
            f"{token_header_schema} {expected_token_value}"
        )
        assert token_value == expected_token_value

    def test_get_bearer_token_value_if_without_bearer_header_schema(self) -> None:
        token_without_schema = "afdsafsa"
        assert JWTTokenHandler.get_bearer_token_value(token_without_schema) == token_without_schema

    @pytest.mark.parametrize(
        "invalid_token_header_schema",
        ["_earer", "adfafd"],
        ids=str,
    )
    def test_get_bearer_token_value_returns_null_if_bearer_header_schema_invalid(
        self, invalid_token_header_schema: str
    ) -> None:
        assert not JWTTokenHandler.get_bearer_token_value(
            f"{invalid_token_header_schema} afdsafsa",
            bearer_token_header_can_be_missing=False,
        )

    def test_is_jwt_token(
        self,
        mock_jwt_get_unverified_header: Mock,
    ) -> None:
        mock_token = Mock()
        output = JWTTokenHandler.is_jwt_token(mock_token)

        mock_jwt_get_unverified_header.assert_called_once_with(mock_token)
        assert output is True

    def test_is_jwt_token_fails(
        self,
        mock_jwt_get_unverified_header: Mock,
    ) -> None:
        mock_jwt_get_unverified_header.side_effect = jwt.exceptions.DecodeError

        mock_token = Mock()
        assert JWTTokenHandler.is_jwt_token(mock_token) is False

    @pytest.mark.parametrize(
        "header_name",
        ["authorization", "Authorization", "AUTHORIZATION"],
        ids=str,
    )
    def test_get_bearer_token_header_header_name_arg_is_case_insensitive(
        self, header_name: str
    ) -> None:
        token_value = "afdaf"
        headers_with_lower_case_key = {"authorization": token_value}
        assert (
            JWTTokenHandler.get_bearer_token_header(header_name, headers_with_lower_case_key)
            == token_value
        )

    def test_get_bearer_token_header_returns_null_if_absent(self) -> None:
        assert JWTTokenHandler.get_bearer_token_header(Mock(), {}) is None

    def test_get_jwt_payload_without_signature_verification(self, mock_jwt_decode: Mock) -> None:
        mock_jwt_token_value = Mock()
        output = JWTTokenHandler.get_jwt_payload_without_signature_verification(
            mock_jwt_token_value
        )

        mock_jwt_decode.assert_called_once_with(
            mock_jwt_token_value,
            options={"verify_signature": False},
        )
        assert output == mock_jwt_decode.return_value

    @pytest.mark.parametrize(
        "input_claims,output_scopes",
        [
            ({"scp": "scope_value"}, ["scope_value"]),
            ({"scp": ["scope_value"]}, ["scope_value"]),
        ],
        ids=str,
    )
    def test_extract_scopes_from_scope_claim(
        self,
        input_claims: dict[str, str | list[str]],
        output_scopes: list[str],
    ) -> None:
        output = JWTTokenHandler.extract_scopes(input_claims)
        assert output == output_scopes

    @pytest.mark.parametrize(
        "input_claims,output_scopes",
        [
            ({"scope": "scope_value"}, ["scope_value"]),
            ({"scope": ["scope_value"]}, ["scope_value"]),
        ],
        ids=str,
    )
    def test_extract_scopes_from_scp_claim(
        self,
        input_claims: dict[str, str | list[str]],
        output_scopes: list[str],
    ) -> None:
        output = JWTTokenHandler.extract_scopes(input_claims)
        assert output == output_scopes

    def test_extract_scopes_returns_null(self) -> None:
        assert JWTTokenHandler.extract_scopes({}) == []

    @pytest.mark.parametrize(
        "input_claims,output_exp",
        [
            ({"exp": "1"}, 1),
            ({"exp": "1.0"}, 1.0),
            ({"exp": None}, None),
            ({}, None),
        ],
        ids=str,
    )
    def test_extract_exp(
        self,
        input_claims: dict[str, str | list[str]],
        output_exp: list[str],
    ) -> None:
        output = JWTTokenHandler.extract_exp(input_claims)
        assert output == output_exp

    @pytest.mark.parametrize(
        "input_claims,output_client_id",
        [
            ({"client_id": "client_id"}, "client_id"),
            ({"azp": "client_id"}, "client_id"),
            ({"sub": "client_id"}, "client_id"),
            ({}, "unknown"),
        ],
        ids=str,
    )
    def test_extract_client_id(
        self,
        input_claims: dict[str, str],
        output_client_id: str,
    ) -> None:
        output = JWTTokenHandler.extract_client_id(input_claims)
        assert output == output_client_id

    def test_parse_to_access_token(
        self,
        mock_extract_client_id: Mock,
        mock_extract_exp: Mock,
        mock_extract_scopes: Mock,
        mock_get_bearer_token_header: Mock,
        mock_get_bearer_token_value: Mock,
        mock_is_jwt_decode: Mock,
        mock_get_jwt_payload_without_signature_verification: Mock,
    ) -> None:
        mock_header_name = Mock()
        mock_header = Mock()
        access_token = JWTTokenHandler.parse_to_access_token(mock_header_name, mock_header)

        mock_get_bearer_token_header.assert_called_once_with(mock_header_name, mock_header)
        mock_get_bearer_token_value.assert_called_once_with(
            mock_get_bearer_token_header.return_value,
        )
        mock_is_jwt_decode.assert_called_once_with(
            mock_get_bearer_token_value.return_value,
        )
        mock_get_jwt_payload_without_signature_verification.assert_called_once_with(
            mock_get_bearer_token_value.return_value,
        )
        mock_jwt_payload = mock_get_jwt_payload_without_signature_verification.return_value
        mock_extract_client_id.assert_called_once_with(mock_jwt_payload)
        mock_extract_scopes.assert_called_once_with(mock_jwt_payload)
        mock_extract_exp.assert_called_once_with(mock_jwt_payload)
        assert isinstance(access_token, AccessToken)
        assert access_token.token == mock_get_bearer_token_value.return_value
        assert access_token.client_id == mock_extract_client_id.return_value
        assert access_token.scopes == mock_extract_scopes.return_value
        assert access_token.expires_at == mock_extract_exp.return_value
        assert (
            access_token.claims == mock_get_jwt_payload_without_signature_verification.return_value
        )

    def test_parse_to_access_token_returns_null_if_no_jwt_bearer_token_header_found(
        self,
        mock_get_bearer_token_header: Mock,
    ) -> None:
        mock_get_bearer_token_header.return_value = None

        assert JWTTokenHandler.parse_to_access_token(Mock(), Mock()) is None

    @pytest.mark.usefixtures("mock_get_bearer_token_header")
    def test_parse_to_access_token_returns_null_if_jwt_bearer_token_header_value_is_invalid(
        self,
        mock_get_bearer_token_value: Mock,
    ) -> None:
        mock_get_bearer_token_value.return_value = None

        assert JWTTokenHandler.parse_to_access_token(Mock(), Mock()) is None

    @pytest.mark.usefixtures(
        "mock_get_bearer_token_header",
        "mock_get_bearer_token_value",
    )
    def test_parse_to_access_token_returns_null_if_jwt_content_is_malformed(
        self,
        mock_is_jwt_decode: Mock,
    ) -> None:
        mock_is_jwt_decode.return_value = False

        assert JWTTokenHandler.parse_to_access_token(Mock(), Mock()) is None

    @pytest.mark.usefixtures(
        "mock_get_bearer_token_header",
        "mock_get_bearer_token_value",
        "mock_is_jwt_decode",
    )
    def test_parse_to_access_token_returns_null_if_jwt_decode_failed(
        self,
        mock_get_jwt_payload_without_signature_verification: Mock,
    ) -> None:
        mock_get_jwt_payload_without_signature_verification.side_effect = (
            jwt.exceptions.DecodeError()
        )

        mock_header_name = Mock()
        mock_header = Mock()
        assert JWTTokenHandler.parse_to_access_token(mock_header_name, mock_header) is None

    @pytest.mark.usefixtures(
        "mock_get_bearer_token_header",
        "mock_get_bearer_token_value",
        "mock_get_jwt_payload_without_signature_verification",
        "mock_is_jwt_decode",
    )
    @pytest.mark.parametrize("raised_error", [ValueError, TypeError], ids=str)
    def test_parse_to_access_token_returns_null_if_access_token_init_failed(
        self,
        raised_error: ValueError | TypeError,
        mock_access_token_cls: Mock,
    ) -> None:
        mock_access_token_cls.side_effect = raised_error

        assert JWTTokenHandler.parse_to_access_token(Mock(), Mock()) is None


class TestJWTTokenClaimsValidator:
    @pytest.fixture
    def mock_logger(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.logger") as mock_logger:
            yield mock_logger

    @pytest.fixture
    def mock_authorization_claims_from_access_token(self) -> Iterator[Mock]:
        with patch.object(AuthorizationClaims, "from_access_token") as mock_func:
            yield mock_func

    def test_init(self, mock_authorization_claims_from_access_token: Mock) -> None:
        mock_authenticated_user = Mock()
        validator = JWTTokenClaimsValidator(mock_authenticated_user)

        mock_authorization_claims_from_access_token.assert_called_once_with(
            mock_authenticated_user.access_token,
        )
        assert validator.claims == mock_authorization_claims_from_access_token.return_value

    def test_validate_audience_claim_passes_when_audience_matches(
        self,
        mock_authorization_claims_from_access_token: Mock,
        mock_logger: Mock,
    ) -> None:
        audience_value = "expected-aud"
        mock_authorization_claims_from_access_token.return_value = AuthorizationClaims(
            scopes=frozenset(),
            audience=audience_value,
        )

        validator = JWTTokenClaimsValidator(Mock())
        validator.validate_audience_claim(audience_value)
        mock_logger.info.assert_called_once_with("Audience claim validation succeeded.")

    def test_validate_audience_claim_ignored_when_no_expected_audience_is_provided(
        self,
        mock_authorization_claims_from_access_token: Mock,
        mock_logger: Mock,
    ) -> None:
        mock_authorization_claims_from_access_token.return_value = AuthorizationClaims(
            scopes=frozenset(),
            audience="expected-aud",
        )

        validator = JWTTokenClaimsValidator(Mock())
        validator.validate_audience_claim(None)
        mock_logger.info.assert_called_once_with(
            "Authorization audience claim validation is not executed. "
            "There is no expected audience claim provided."
        )

    def test_validate_audience_claim_fails_when_audience_mismatches(
        self,
        mock_authorization_claims_from_access_token: Mock,
    ) -> None:
        mock_authorization_claims_from_access_token.return_value = AuthorizationClaims(
            scopes=frozenset(),
            audience="other-aud",
        )

        validator = JWTTokenClaimsValidator(Mock())
        with pytest.raises(AudienceClaimValidationError):
            validator.validate_audience_claim("expected-aud")

    def test_validate_audience_claim_fails_if_no_valid_claim(
        self,
        mock_authorization_claims_from_access_token: Mock,
    ) -> None:
        mock_authorization_claims_from_access_token.return_value = AuthorizationClaims(
            scopes=frozenset(),
            audience=None,
        )

        validator = JWTTokenClaimsValidator(Mock())
        with pytest.raises(AudienceClaimValidationError):
            validator.validate_audience_claim(ANY)

    def test_validate_mcp_tool_scope_claim_passes(
        self,
        mock_authorization_claims_from_access_token: Mock,
        mock_logger: Mock,
    ) -> None:
        expected_scope_value = frozenset(["expected-scope"])
        mock_authorization_claims_from_access_token.return_value = AuthorizationClaims(
            scopes=expected_scope_value,
        )

        validator = JWTTokenClaimsValidator(Mock())
        validator.validate_mcp_tool_scope_claims(expected_scope_value)
        mock_logger.info.assert_called_once_with("MCP tool scope claim validation succeeded.")

    @pytest.mark.parametrize("expected_scopes", [frozenset([]), None], ids=str)
    def test_validate_mcp_tool_scope_claim_ignored_when_no_expected_scope_is_provided(
        self,
        expected_scopes: frozenset[str] | None,
        mock_authorization_claims_from_access_token: Mock,
        mock_logger: Mock,
    ) -> None:
        mock_authorization_claims_from_access_token.return_value = AuthorizationClaims(
            scopes=frozenset(["scope-to-validate"]),
        )

        validator = JWTTokenClaimsValidator(Mock())
        validator.validate_mcp_tool_scope_claims(expected_scopes)
        mock_logger.info.assert_called_once_with(
            "Authorization MCP tool scope claim validation is not executed. "
            "There is no expected scope claim provided."
        )

    def test_validate_mcp_tool_scope_claim_fails_when_scope_mismatches(
        self,
        mock_authorization_claims_from_access_token: Mock,
        mock_logger: Mock,
    ) -> None:
        expected_scope_value = frozenset(["expected-scope"])
        mock_authorization_claims_from_access_token.return_value = AuthorizationClaims(
            scopes=expected_scope_value,
        )

        validator = JWTTokenClaimsValidator(Mock())
        with pytest.raises(MCPToolScopeClaimValidationError):
            scope_to_validate = frozenset(["not-expected-scope"])
            validator.validate_mcp_tool_scope_claims(scope_to_validate)

    def test_validate_mcp_tool_scope_claim_fails_if_no_valid_claim(
        self,
        mock_authorization_claims_from_access_token: Mock,
    ) -> None:
        mock_authorization_claims_from_access_token.return_value = AuthorizationClaims(
            scopes=frozenset(),
            audience=None,
        )

        validator = JWTTokenClaimsValidator(Mock())
        with pytest.raises(MCPToolScopeClaimValidationError):
            validator.validate_mcp_tool_scope_claims(frozenset(["scope-to-validate"]))
