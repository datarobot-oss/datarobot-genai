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
from http import HTTPMethod
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from datarobot_genai.drtools.core.clients.utils import get_async_https_retry_client
from datarobot_genai.drtools.core.clients.utils import get_async_https_session
from datarobot_genai.drtools.core.clients.utils import get_ssl_context_from_ca_file


@pytest.fixture
def module_under_test() -> str:
    return "datarobot_genai.drtools.core.clients.utils"


class TestConnectionSetupRelated:
    @pytest.fixture
    def mock_create_default_ssl_context(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.create_default_ssl_context") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_tcp_connector_cls(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.TCPConnector") as mock_cls:
            yield mock_cls

    @pytest.fixture
    def mock_client_timeout_cls(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.ClientTimeout") as mock_cls:
            yield mock_cls

    @pytest.fixture
    def mock_client_session_cls(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.ClientSession") as mock_cls:
            yield mock_cls

    @pytest.fixture
    def mock_exponential_retry_cls(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.ExponentialRetry") as mock_cls:
            yield mock_cls

    @pytest.fixture
    def mock_retry_client_cls(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.RetryClient") as mock_cls:
            yield mock_cls

    @pytest.fixture
    def mock_get_ssl_context_from_ca_file(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.get_ssl_context_from_ca_file") as mock_func:
            yield mock_func

    def test_ssl_context_from_ca_file(self, mock_create_default_ssl_context: Mock) -> None:
        mock_ca_path = Mock()
        output = get_ssl_context_from_ca_file(mock_ca_path)

        mock_create_default_ssl_context.assert_called_once_with()
        mock_ssl_context = mock_create_default_ssl_context.return_value
        mock_ssl_context.load_verify_locations.assert_called_once_with(cafile=mock_ca_path)
        assert output == mock_create_default_ssl_context.return_value

    def test_get_async_https_session(
        self,
        mock_tcp_connector_cls: Mock,
        mock_client_timeout_cls: Mock,
        mock_client_session_cls: Mock,
        mock_get_ssl_context_from_ca_file: Mock,
    ) -> None:
        mock_ca = Mock()
        output = get_async_https_session(mock_ca)

        mock_get_ssl_context_from_ca_file.assert_called_once_with(mock_ca)
        mock_ssl_context = mock_get_ssl_context_from_ca_file.return_value
        mock_tcp_connector_cls.assert_called_once_with(ssl=mock_ssl_context)
        mock_client_timeout_cls.assert_called_once_with(connect=30, sock_read=60)
        expected_headers = {"User-Agent": "global-mcp"}
        mock_client_session_cls.assert_called_once_with(
            headers=expected_headers,
            connector=mock_tcp_connector_cls.return_value,
            timeout=mock_client_timeout_cls.return_value,
        )
        assert output == mock_client_session_cls.return_value

    def test_get_async_https_retry_client(
        self,
        mock_exponential_retry_cls: Mock,
        mock_retry_client_cls: Mock,
    ) -> None:
        mock_session = Mock()
        output = get_async_https_retry_client(mock_session)

        mock_exponential_retry_cls.assert_called_once_with(
            attempts=3,
            start_timeout=0.1,
            methods=[HTTPMethod.GET, HTTPMethod.POST, HTTPMethod.PUT],
        )
        mock_retry = mock_exponential_retry_cls.return_value
        mock_retry_client_cls.assert_called_once_with(
            client_session=mock_session,
            retry_options=mock_retry,
        )
        assert output == mock_retry_client_cls.return_value
