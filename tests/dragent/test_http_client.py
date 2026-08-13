# Copyright 2026 DataRobot, Inc. and its affiliates.
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
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from datarobot_genai.dragent.http_client import AsyncClientConfig
from datarobot_genai.dragent.http_client import get_retriable_async_http_client


@pytest.fixture
def module_under_test() -> str:
    return "datarobot_genai.dragent.http_client"


class TestAsyncClientConfig:
    @pytest.fixture
    def mock_retry_cls(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.Retry") as mock_cls:
            yield mock_cls

    @pytest.fixture
    def mock_retry_transport_cls(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.RetryTransport") as mock_cls:
            yield mock_cls

    @pytest.fixture
    def mock_timeout_cls(
        self,
        module_under_test: str,
    ) -> Iterator[Mock]:
        with patch(f"{module_under_test}.Timeout") as mock_cls:
            yield mock_cls

    @pytest.fixture
    def mock_create_retry_policy(self, module_under_test: str) -> Iterator[Mock]:
        with patch.object(AsyncClientConfig, "create_retry_policy") as mock_func:
            yield mock_func

    def test_create_retry_policy(self, mock_retry_cls: Mock) -> None:
        config = AsyncClientConfig()
        output = config.create_retry_policy()

        mock_retry_cls.assert_called_once_with(
            total=config.max_retries,
            backoff_factor=config.retry_backoff,
        )
        assert output == mock_retry_cls.return_value

    def test_create_httpx_retry_transport(
        self,
        mock_create_retry_policy: Mock,
        mock_retry_transport_cls: Mock,
    ) -> None:
        config = AsyncClientConfig()
        output = config.create_httpx_retry_transport()

        mock_retry_transport_cls.assert_called_once_with(
            retry=mock_create_retry_policy.return_value
        )
        assert output == mock_retry_transport_cls.return_value

    def test_create_httpx_timeout(
        self,
        mock_timeout_cls: Mock,
    ) -> None:
        config = AsyncClientConfig()
        output = config.create_httpx_timeout()

        mock_timeout_cls.assert_called_once_with(config.timeout_in_seconds)
        assert output == mock_timeout_cls.return_value


class TestAsyncHttpClient:
    @pytest.fixture
    def mock_async_client_cls(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.AsyncClient") as mock_cls:
            yield mock_cls

    def test_get_retriable_async_http_client(self, mock_async_client_cls: Mock) -> None:
        mock_auth_with_headers = Mock()
        mock_async_client_config = Mock()
        output = get_retriable_async_http_client(mock_auth_with_headers, mock_async_client_config)

        mock_async_client_cls.assert_called_once_with(
            auth=mock_auth_with_headers,
            timeout=mock_async_client_config.create_httpx_timeout(),
            transport=mock_async_client_config.create_httpx_retry_transport(),
        )
        assert output == mock_async_client_cls.return_value
