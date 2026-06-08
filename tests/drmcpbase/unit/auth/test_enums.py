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

import pytest

from datarobot_genai.drmcpbase.auth.enums import DataRobotBearerHeaderEnum
from datarobot_genai.drmcpbase.auth.exceptions import InvalidBearerTokenError
from datarobot_genai.drmcpbase.auth.exceptions import (
    NoDataRobotBearerTokenFoundInRequestContextError,
)
from datarobot_genai.drmcpbase.auth.exceptions import NoHeadersFoundInRequestContextError


@pytest.fixture
def module_under_test() -> str:
    return "datarobot_genai.drmcpbase.auth.enums"


class TestDataRobotBearerHeaderEnum:
    @pytest.fixture
    def mock_get_http_headers(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.get_http_headers") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_extract_bearer_token_value(self) -> Iterator[Mock]:
        with patch.object(DataRobotBearerHeaderEnum, "extract_bearer_token_value") as mock_func:
            yield mock_func

    @pytest.mark.parametrize(
        "bearer_header_enum",
        [bearer_header_enum for bearer_header_enum in DataRobotBearerHeaderEnum],
    )
    def test_get_header_key_value(self, bearer_header_enum: DataRobotBearerHeaderEnum) -> None:
        bearer_header_enum_and_value_map = {
            DataRobotBearerHeaderEnum.X_DATAROBOT_AUTHORIZATION: "x-datarobot-authorization",
            DataRobotBearerHeaderEnum.AUTHORIZATION: "authorization",
        }
        assert (
            bearer_header_enum_and_value_map[bearer_header_enum]
            == bearer_header_enum.get_normalized_header_key()
        )

    def test_extract_bearer_token_value(self) -> None:
        mock_token_value = "dafdaf"
        mock_bearer_token_string = f"Bearer {mock_token_value}"
        output = DataRobotBearerHeaderEnum.extract_bearer_token_value(mock_bearer_token_string)

        assert output == mock_token_value

    def test_extract_bearer_token_value_raise_error(self) -> None:
        with pytest.raises(InvalidBearerTokenError):
            invalid_bearer_token_string = "sdfafasfa"
            DataRobotBearerHeaderEnum.extract_bearer_token_value(invalid_bearer_token_string)

    def test_get_from_mcp_request(
        self,
        mock_get_http_headers: Mock,
    ) -> None:
        expected_token_value = "dsafsafas"
        token_string = f"Bearer {expected_token_value}"
        mock_get_http_headers.return_value = {
            DataRobotBearerHeaderEnum.AUTHORIZATION.get_normalized_header_key(): token_string,
        }

        output = DataRobotBearerHeaderEnum.AUTHORIZATION.get_from_mcp_request()
        mock_get_http_headers.assert_called_once_with(include_all=True)
        assert output == expected_token_value

    def test_get_from_mcp_request_raise_invalid_bearer_token_error(
        self,
        mock_get_http_headers: Mock,
    ) -> None:
        invalid_token_string = "dafadfs"
        mock_get_http_headers.return_value = {
            DataRobotBearerHeaderEnum.AUTHORIZATION.get_normalized_header_key(): invalid_token_string,  # noqa: E501
        }

        with pytest.raises(InvalidBearerTokenError):
            DataRobotBearerHeaderEnum.AUTHORIZATION.get_from_mcp_request()

    def test_get_from_mcp_request_raise_no_header_found_error(
        self,
        mock_get_http_headers: Mock,
    ) -> None:
        mock_get_http_headers.return_value = None

        with pytest.raises(NoHeadersFoundInRequestContextError):
            DataRobotBearerHeaderEnum.AUTHORIZATION.get_from_mcp_request()

    def test_get_from_mcp_request_raise_no_datarobot_bearer_token_found_error(
        self,
        mock_get_http_headers: Mock,
    ) -> None:
        mock_get_http_headers.return_value = {Mock(): Mock()}

        with pytest.raises(NoDataRobotBearerTokenFoundInRequestContextError):
            DataRobotBearerHeaderEnum.AUTHORIZATION.get_from_mcp_request()
