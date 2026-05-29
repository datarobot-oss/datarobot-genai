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
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from datarobot_genai.drtools.core.rest_client import request_user_dr_client
from datarobot_genai.drtools.core.rest_client import resolve_request_user_token

_MODULE = "datarobot_genai.drtools.core.rest_client"


class TestResolveRequestUserToken:
    def test_uses_token_from_request_headers(self) -> None:
        with (
            patch(f"{_MODULE}.resolve_token_from_headers", return_value="user-tok"),
            patch(f"{_MODULE}.get_credentials"),
        ):
            assert resolve_request_user_token() == "user-tok"

    def test_falls_back_to_app_token_when_no_header(self) -> None:
        mock_creds = MagicMock()
        mock_creds.datarobot.application_api_token = "app-tok"
        with (
            patch(f"{_MODULE}.resolve_token_from_headers", return_value=None),
            patch(f"{_MODULE}.get_credentials", return_value=mock_creds),
        ):
            assert resolve_request_user_token() == "app-tok"

    def test_raises_when_headers_auth_only_and_no_token(self) -> None:
        with (
            patch(f"{_MODULE}.resolve_token_from_headers", return_value=None),
            patch(f"{_MODULE}.get_credentials"),
        ):
            with pytest.raises(ValueError, match="No API token found"):
                resolve_request_user_token(headers_auth_only=True)


class TestRequestUserDrClient:
    def test_scopes_client_to_resolved_token_and_yields_it(self) -> None:
        mock_creds = MagicMock()
        mock_creds.datarobot.endpoint = "https://x.example/api/v2"
        with (
            patch(f"{_MODULE}.resolve_token_from_headers", return_value="user-tok"),
            patch(f"{_MODULE}.get_credentials", return_value=mock_creds),
            patch(f"{_MODULE}.client_configuration") as mock_cfg,
            patch(f"{_MODULE}.dr") as mock_dr,
        ):
            expected = mock_dr.client.get_client.return_value
            with request_user_dr_client() as client:
                # client_configuration is entered for the body's lifetime, scoped
                # to the resolved token + endpoint (ContextVar, not global dr.Client).
                mock_cfg.assert_called_once_with(
                    token="user-tok", endpoint="https://x.example/api/v2"
                )
                mock_cfg.return_value.__enter__.assert_called_once()
                assert client is expected
            # context exited -> scope released
            mock_cfg.return_value.__exit__.assert_called_once()

    def test_propagates_headers_auth_only(self) -> None:
        with (
            patch(f"{_MODULE}.resolve_token_from_headers", return_value=None),
            patch(f"{_MODULE}.get_credentials"),
            patch(f"{_MODULE}.client_configuration"),
            patch(f"{_MODULE}.dr"),
        ):
            with pytest.raises(ValueError, match="No API token found"):
                with request_user_dr_client(headers_auth_only=True):
                    pass
