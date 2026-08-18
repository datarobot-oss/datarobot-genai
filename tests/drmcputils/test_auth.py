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
from unittest.mock import Mock
from unittest.mock import patch

import jwt
import pytest

from datarobot_genai.drmcputils.auth import AuthorizationClaims
from datarobot_genai.drmcputils.auth import JWTTokenClaimsValidator
from datarobot_genai.drmcputils.exceptions import AudienceClaimValidationError


class TestAuthorizationClaims:
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
        claims = AuthorizationClaims.from_jwt_payload_partition({"aud": aud_value})
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
        claims = AuthorizationClaims.from_jwt_payload_partition({"aud": aud_value})
        assert claims.contain_expected_audience("expected_aud") is contain_expected_audience


class TestJWTTokenClaimsValidator:
    @pytest.fixture
    def module_under_test(self) -> str:
        return "datarobot_genai.drmcputils.auth"

    @pytest.fixture
    def mock_logger(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.logger") as mock_logger:
            yield mock_logger

    @pytest.fixture
    def mock_claims_validator_get_claims_from_jwt_token(self) -> Iterator[Mock]:
        with patch.object(
            JWTTokenClaimsValidator,
            "get_claims_from_jwt_token",
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_claims_validator_get_bearer_token_header(self) -> Iterator[Mock]:
        with patch.object(
            JWTTokenClaimsValidator,
            "get_bearer_token_header",
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_claims_validator_get_bearer_token_value(self) -> Iterator[Mock]:
        with patch.object(
            JWTTokenClaimsValidator,
            "get_bearer_token_value",
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_claims_validator_is_jwt_decode(self) -> Iterator[Mock]:
        with patch.object(JWTTokenClaimsValidator, "is_jwt_token") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_claims_validator_get_jwt_payload_without_signature_verification(
        self,
    ) -> Iterator[Mock]:
        with patch.object(
            JWTTokenClaimsValidator, "get_jwt_payload_without_signature_verification"
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_claims_validator_has_valid_claims(self) -> Iterator[Mock]:
        with patch.object(JWTTokenClaimsValidator, "has_valid_claims") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_jwt_get_unverified_header(self) -> Iterator[Mock]:
        with patch.object(jwt, "get_unverified_header") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_jwt_decode(self) -> Iterator[Mock]:
        with patch.object(jwt, "decode") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_from_jwt_payload_partition(self) -> Iterator[Mock]:
        with patch.object(AuthorizationClaims, "from_jwt_payload_partition") as mock_func:
            yield mock_func

    def test_init(self, mock_claims_validator_get_claims_from_jwt_token: Mock) -> None:
        mock_http_request_header_name = Mock()
        mock_http_request_header = Mock()
        validator = JWTTokenClaimsValidator(mock_http_request_header_name, mock_http_request_header)

        assert validator.http_header_name_of_jwt_token == mock_http_request_header_name
        mock_claims_validator_get_claims_from_jwt_token.assert_called_once_with(
            mock_http_request_header_name, mock_http_request_header
        )
        assert validator.claims == mock_claims_validator_get_claims_from_jwt_token.return_value

    @pytest.mark.parametrize(
        "get_claims_result,has_valid_claims",
        [(Mock(), True), (None, False)],
        ids=str,
    )
    def test_check_validity_of_claims(
        self,
        get_claims_result: Mock | None,
        has_valid_claims: bool,
        mock_claims_validator_get_claims_from_jwt_token: Mock,
    ) -> None:
        mock_claims_validator_get_claims_from_jwt_token.return_value = get_claims_result

        mock_http_request_header = Mock()
        validator = JWTTokenClaimsValidator(Mock(), mock_http_request_header)

        assert validator.has_valid_claims() is has_valid_claims

    @pytest.mark.parametrize(
        "token_header_schema",
        ["bearer", "Bearer"],
        ids=str,
    )
    def test_get_bearer_token_value(self, token_header_schema: str) -> None:
        expected_token_value = "adfad adfada"
        token_value = JWTTokenClaimsValidator.get_bearer_token_value(
            f"{token_header_schema} {expected_token_value}"
        )
        assert token_value == expected_token_value

    def test_get_bearer_token_value_if_without_bearer_header_schema(self) -> None:
        token_without_schema = "afdsafsa"
        assert (
            JWTTokenClaimsValidator.get_bearer_token_value(token_without_schema)
            == token_without_schema
        )

    @pytest.mark.parametrize(
        "invalid_token_header_schema",
        ["_earer", "adfafd"],
        ids=str,
    )
    def test_get_bearer_token_value_returns_null_if_bearer_header_schema_invalid(
        self, invalid_token_header_schema: str
    ) -> None:
        assert not JWTTokenClaimsValidator.get_bearer_token_value(
            f"{invalid_token_header_schema} afdsafsa",
            bearer_token_header_can_be_missing=False,
        )

    def test_is_jwt_token(
        self,
        mock_jwt_get_unverified_header: Mock,
    ) -> None:
        mock_token = Mock()
        output = JWTTokenClaimsValidator.is_jwt_token(mock_token)

        mock_jwt_get_unverified_header.assert_called_once_with(mock_token)
        assert output is True

    def test_is_jwt_token_fails(
        self,
        mock_jwt_get_unverified_header: Mock,
    ) -> None:
        mock_jwt_get_unverified_header.side_effect = jwt.exceptions.DecodeError

        mock_token = Mock()
        assert JWTTokenClaimsValidator.is_jwt_token(mock_token) is False

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
            JWTTokenClaimsValidator.get_bearer_token_header(
                header_name, headers_with_lower_case_key
            )
            == token_value
        )

    def test_get_bearer_token_header_returns_null_if_absent(self) -> None:
        assert JWTTokenClaimsValidator.get_bearer_token_header(Mock(), {}) is None

    def test_get_jwt_payload_without_signature_verification(self, mock_jwt_decode: Mock) -> None:
        mock_jwt_token_value = Mock()
        output = JWTTokenClaimsValidator.get_jwt_payload_without_signature_verification(
            mock_jwt_token_value
        )

        mock_jwt_decode.assert_called_once_with(
            mock_jwt_token_value,
            options={"verify_signature": False},
        )
        assert output == mock_jwt_decode.return_value

    def test_get_claims(
        self,
        mock_from_jwt_payload_partition: Mock,
        mock_claims_validator_get_bearer_token_header: Mock,
        mock_claims_validator_get_bearer_token_value: Mock,
        mock_claims_validator_is_jwt_decode: Mock,
        mock_claims_validator_get_jwt_payload_without_signature_verification: Mock,
    ) -> None:
        mock_header_name = Mock()
        mock_header = Mock()
        validator = JWTTokenClaimsValidator(mock_header_name, mock_header)

        mock_claims_validator_get_bearer_token_header.assert_called_once_with(
            mock_header_name, mock_header
        )
        mock_claims_validator_get_bearer_token_value.assert_called_once_with(
            mock_claims_validator_get_bearer_token_header.return_value,
        )
        mock_claims_validator_is_jwt_decode.assert_called_once_with(
            mock_claims_validator_get_bearer_token_value.return_value,
        )
        mock_claims_validator_get_jwt_payload_without_signature_verification.assert_called_once_with(
            mock_claims_validator_get_bearer_token_value.return_value,
        )
        mock_from_jwt_payload_partition.assert_called_once_with(
            mock_claims_validator_get_jwt_payload_without_signature_verification.return_value,
        )
        assert validator.claims is mock_from_jwt_payload_partition.return_value

    def test_get_claims_returns_null_if_no_jwt_bearer_token_header_found(
        self,
        mock_claims_validator_get_bearer_token_header: Mock,
    ) -> None:
        mock_claims_validator_get_bearer_token_header.return_value = None

        assert JWTTokenClaimsValidator.get_claims_from_jwt_token(Mock(), Mock()) is None

    def test_get_claims_returns_null_if_jwt_bearer_token_header_value_is_invalid(
        self,
        mock_claims_validator_get_bearer_token_header: Mock,
        mock_claims_validator_get_bearer_token_value: Mock,
    ) -> None:
        mock_claims_validator_get_bearer_token_value.return_value = None

        mock_header_name = Mock()
        mock_header = Mock()
        assert (
            JWTTokenClaimsValidator.get_claims_from_jwt_token(mock_header_name, mock_header) is None
        )

        mock_claims_validator_get_bearer_token_header.assert_called_once_with(
            mock_header_name, mock_header
        )
        mock_claims_validator_get_bearer_token_value.assert_called_once_with(
            mock_claims_validator_get_bearer_token_header.return_value,
        )

    def test_get_claims_returns_null_if_jwt_content_is_malformed(
        self,
        mock_claims_validator_get_bearer_token_header: Mock,
        mock_claims_validator_get_bearer_token_value: Mock,
        mock_claims_validator_is_jwt_decode: Mock,
    ) -> None:
        mock_claims_validator_is_jwt_decode.return_value = False

        mock_header_name = Mock()
        mock_header = Mock()
        assert (
            JWTTokenClaimsValidator.get_claims_from_jwt_token(mock_header_name, mock_header) is None
        )

        mock_claims_validator_get_bearer_token_header.assert_called_once_with(
            mock_header_name, mock_header
        )
        mock_claims_validator_get_bearer_token_value.assert_called_once_with(
            mock_claims_validator_get_bearer_token_header.return_value,
        )
        mock_claims_validator_is_jwt_decode.assert_called_once_with(
            mock_claims_validator_get_bearer_token_value.return_value,
        )

    def test_validate_audience_claim_passes_when_audience_matches(
        self,
        mock_claims_validator_get_claims_from_jwt_token: Mock,
        mock_logger: Mock,
    ) -> None:
        mock_claims_validator_get_claims_from_jwt_token.return_value = AuthorizationClaims(
            audience="expected-aud",
        )

        validator = JWTTokenClaimsValidator(Mock(), Mock())
        validator.validate_audience_claim("expected-aud")
        mock_logger.info.assert_called_once_with("Audience claim validation succeeded.")

    def test_validate_audience_claim_ignored_when_no_expected_audience_is_provided(
        self,
        mock_claims_validator_get_claims_from_jwt_token: Mock,
        mock_logger: Mock,
    ) -> None:
        mock_claims_validator_get_claims_from_jwt_token.return_value = AuthorizationClaims(
            audience="expected-aud",
        )

        validator = JWTTokenClaimsValidator(Mock(), Mock())
        validator.validate_audience_claim(None)
        mock_logger.info.assert_called_once_with(
            "Authorization audience claim validation is not executed. "
            "There is no expected audience claim provided."
        )

    def test_validate_audience_claim_fails_when_audience_mismatches(
        self,
        mock_claims_validator_get_claims_from_jwt_token: Mock,
    ) -> None:
        mock_claims_validator_get_claims_from_jwt_token.return_value = AuthorizationClaims(
            audience="other-aud",
        )

        validator = JWTTokenClaimsValidator(Mock(), Mock())
        with pytest.raises(AudienceClaimValidationError):
            validator.validate_audience_claim("expected-aud")

    def test_validate_audience_claim_fails_if_no_valid_claim(
        self,
        mock_claims_validator_get_claims_from_jwt_token: Mock,
    ) -> None:
        mock_claims_validator_get_claims_from_jwt_token.return_value = None

        validator = JWTTokenClaimsValidator(Mock(), Mock())
        with pytest.raises(AudienceClaimValidationError):
            validator.validate_audience_claim("expected-aud")
