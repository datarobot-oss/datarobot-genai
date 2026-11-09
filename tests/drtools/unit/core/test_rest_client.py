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

from datarobot_genai.drmcputils.auth import set_request_headers
from datarobot_genai.drmcputils.clients.datarobot import ThreadSafeDataRobotClient
from datarobot_genai.drmcputils.clients.datarobot import get_datarobot_access_token
from datarobot_genai.drmcputils.clients.datarobot import request_user_dr_client
from datarobot_genai.drmcputils.clients.datarobot import request_user_dr_sdk
from datarobot_genai.drmcputils.exceptions import ToolError

_MODULE = "datarobot_genai.drmcputils.clients.datarobot"


class TestGetDatarobotAccessTokenExtended:
    def test_uses_token_from_request_headers(self) -> None:
        with (
            patch(f"{_MODULE}.resolve_datarobot_token", return_value="user-tok"),
            patch(f"{_MODULE}.get_credentials"),
        ):
            assert get_datarobot_access_token() == "user-tok"

    def test_falls_back_to_app_token_when_headers_auth_only_false(self) -> None:
        mock_creds = MagicMock()
        mock_creds.datarobot.datarobot_api_token = "app-tok"
        with (
            patch(f"{_MODULE}.resolve_datarobot_token", return_value=None),
            patch(f"{_MODULE}.get_credentials", return_value=mock_creds),
        ):
            assert get_datarobot_access_token(headers_auth_only=False) == "app-tok"

    def test_raises_tool_error_when_headers_auth_only_and_no_token(self) -> None:
        with (
            patch(f"{_MODULE}.resolve_datarobot_token", return_value=None),
            patch(f"{_MODULE}.get_credentials"),
        ):
            with pytest.raises(ToolError, match="DataRobot API token not found"):
                get_datarobot_access_token(headers_auth_only=True)


class TestRequestUserDrClient:
    def test_scopes_client_to_resolved_token_and_yields_it(self) -> None:
        mock_creds = MagicMock()
        mock_creds.datarobot.datarobot_endpoint = "https://x.example/api/v2"
        with (
            patch(f"{_MODULE}.resolve_datarobot_token", return_value="user-tok"),
            patch(f"{_MODULE}.get_credentials", return_value=mock_creds),
            patch(f"{_MODULE}.client_configuration") as mock_cfg,
            patch(f"{_MODULE}.dr") as mock_dr,
            patch(f"{_MODULE}.DRContext"),
        ):
            expected = mock_dr.client.get_client.return_value
            with request_user_dr_client() as client:
                mock_cfg.assert_called_once_with(
                    token="user-tok", endpoint="https://x.example/api/v2", token_type=None
                )
                mock_cfg.return_value.__enter__.assert_called_once()
                assert client is expected
            mock_cfg.return_value.__exit__.assert_called_once()

    def test_raises_tool_error_when_headers_auth_only_and_no_token(self) -> None:
        with (
            patch(f"{_MODULE}.resolve_datarobot_token", return_value=None),
            patch(f"{_MODULE}.get_credentials"),
            patch(f"{_MODULE}.client_configuration"),
            patch(f"{_MODULE}.dr"),
        ):
            with pytest.raises(ToolError, match="DataRobot API token not found"):
                with request_user_dr_client(headers_auth_only=True):
                    pass


class TestRequestUserDrSdk:
    def test_scopes_sdk_via_client_configuration(self) -> None:
        mock_creds = MagicMock()
        mock_creds.datarobot.datarobot_endpoint = "https://x.example/api/v2"
        with (
            patch(f"{_MODULE}.resolve_datarobot_token", return_value="user-tok"),
            patch(f"{_MODULE}.get_credentials", return_value=mock_creds),
            patch(f"{_MODULE}.client_configuration") as mock_cfg,
            patch(f"{_MODULE}.DRContext"),
        ):
            with request_user_dr_sdk(headers_auth_only=False) as sdk:
                import datarobot as dr_mod

                assert sdk is dr_mod
            mock_cfg.assert_called_once_with(
                token="user-tok", endpoint="https://x.example/api/v2", token_type=None
            )


class TestThreadSafeDataRobotClientRequestUserClient:
    def test_delegates_to_same_client_configuration(self) -> None:
        mock_creds = MagicMock()
        mock_creds.datarobot.datarobot_endpoint = "https://x.example/api/v2"
        with (
            patch(f"{_MODULE}.resolve_datarobot_token", return_value="user-tok"),
            patch(f"{_MODULE}.get_credentials", return_value=mock_creds),
            patch(f"{_MODULE}.client_configuration") as mock_cfg,
            patch(f"{_MODULE}.dr") as mock_dr,
            patch(f"{_MODULE}.DRContext"),
        ):
            expected = mock_dr.client.get_client.return_value
            client_wrapper = ThreadSafeDataRobotClient()
            with client_wrapper.request_user_client() as client:
                mock_cfg.assert_called_once_with(
                    token="user-tok", endpoint="https://x.example/api/v2", token_type=None
                )
                assert client is expected


@pytest.mark.parametrize(
    "scoped_client",
    [
        request_user_dr_client,
        request_user_dr_sdk,
        lambda: ThreadSafeDataRobotClient().request_user_client(),
    ],
)
@pytest.mark.parametrize(
    ("headers", "token_type"),
    [
        ({"x-datarobot-external-access-token": "idp-token"}, "Bearer"),
        ({"x-datarobot-external-access-token": "Bearer idp-token"}, "Bearer"),
        ({"x-datarobot-external-access-token": "idp-token", "x-datarobot-api-key": "key"}, None),
    ],
)
def test_sends_the_callers_idp_token_as_bearer(scoped_client, headers, token_type) -> None:
    # GIVEN request headers that carry the caller's IdP token
    set_request_headers(headers)

    # WHEN a request-scoped client is configured
    with (
        patch(f"{_MODULE}.client_configuration") as mock_cfg,
        patch(f"{_MODULE}.dr"),
        patch(f"{_MODULE}.DRContext"),
    ):
        with scoped_client():
            pass

    # THEN the IdP token is sent as Bearer, unless an API key outranks it
    assert mock_cfg.call_args.kwargs["token_type"] == token_type
