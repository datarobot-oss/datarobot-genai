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
from unittest.mock import Mock
from unittest.mock import patch

import datarobot as dr
import pytest
from datarobot.auth.identity import Identity
from datarobot.auth.session import AuthCtx
from datarobot.auth.users import User

from datarobot_genai.drmcputils.auth import set_request_headers
from datarobot_genai.drmcputils.clients.datarobot import _suspend_default_use_case
from datarobot_genai.drmcputils.clients.datarobot import request_user_dr_sdk
from datarobot_genai.drmcputils.exceptions import ToolError

_MODULE = "datarobot_genai.drmcputils.clients.datarobot"


class TestRequestUserDrSdk:
    @patch(f"{_MODULE}.client_configuration")
    @patch(f"{_MODULE}.get_credentials")
    def test_uses_token_from_x_datarobot_authorization_header(
        self, mock_get_creds, mock_client_configuration
    ) -> None:
        set_request_headers({"x-datarobot-authorization": "Bearer header-token"})
        mock_creds = MagicMock()
        mock_creds.datarobot.datarobot_endpoint = "https://test.datarobot.com/api/v2"
        mock_get_creds.return_value = mock_creds

        with request_user_dr_sdk(headers_auth_only=False) as sdk:
            assert sdk is dr

        mock_client_configuration.assert_called_once_with(
            token="header-token", endpoint="https://test.datarobot.com/api/v2"
        )

    @patch(f"{_MODULE}.get_credentials")
    def test_raises_tool_error_when_no_token_in_headers(self, mock_get_creds) -> None:
        set_request_headers({})
        mock_creds = MagicMock()
        mock_creds.datarobot.datarobot_api_token = "credential-token"
        mock_get_creds.return_value = mock_creds

        with pytest.raises(ToolError, match="DataRobot API token not found"):
            with request_user_dr_sdk(headers_auth_only=True):
                pass

    @patch(f"{_MODULE}.client_configuration")
    @patch(f"{_MODULE}.get_credentials")
    def test_uses_datarobot_api_token_when_no_headers(
        self, mock_get_creds, mock_client_configuration
    ) -> None:
        set_request_headers({})
        mock_creds = MagicMock()
        mock_creds.datarobot.datarobot_api_token = "env-api-token"
        mock_creds.datarobot.datarobot_endpoint = "https://app.datarobot.com/api/v2"
        mock_get_creds.return_value = mock_creds

        with request_user_dr_sdk(headers_auth_only=False) as sdk:
            assert sdk is dr

        mock_client_configuration.assert_called_once_with(
            token="env-api-token", endpoint="https://app.datarobot.com/api/v2"
        )

    @patch(f"{_MODULE}.client_configuration")
    @patch(f"{_MODULE}.get_credentials")
    @patch(f"{_MODULE}.DRContext")
    def test_suspends_and_restores_dr_context_use_case(
        self, mock_dr_context, mock_get_creds, mock_client_configuration
    ) -> None:
        """GIVEN the application set a default Use Case on the global SDK context
        WHEN a request-scoped SDK client is used
        THEN the default is cleared only inside the block and restored on exit
        (regression: it was nulled permanently — DRContext is a process-global
        singleton, so concurrent users had their default clobbered).
        """
        with patch(f"{_MODULE}.resolve_datarobot_token", return_value="tok"):
            mock_creds = MagicMock()
            mock_creds.datarobot.datarobot_endpoint = "https://test.datarobot.com/api/v2"
            mock_get_creds.return_value = mock_creds
            mock_dr_context.use_case = "some-use-case"
            mock_dr_context._use_case = "some-use-case"

            with request_user_dr_sdk():
                assert mock_dr_context.use_case is None

            assert mock_dr_context.use_case == "some-use-case"

    @patch(f"{_MODULE}.client_configuration")
    @patch(f"{_MODULE}.get_credentials")
    @patch(f"{_MODULE}.DRContext")
    def test_restores_dr_context_use_case_on_error(
        self, mock_dr_context, mock_get_creds, mock_client_configuration
    ) -> None:
        """The saved default Use Case is restored even when the block raises."""
        with patch(f"{_MODULE}.resolve_datarobot_token", return_value="tok"):
            mock_creds = MagicMock()
            mock_creds.datarobot.datarobot_endpoint = "https://test.datarobot.com/api/v2"
            mock_get_creds.return_value = mock_creds
            mock_dr_context.use_case = "some-use-case"
            mock_dr_context._use_case = "some-use-case"

            with pytest.raises(RuntimeError, match="boom"):
                with request_user_dr_sdk():
                    raise RuntimeError("boom")

            assert mock_dr_context.use_case == "some-use-case"

    @patch(f"{_MODULE}.DRContext")
    def test_overlapping_suspends_restore_once(self, mock_dr_context) -> None:
        """GIVEN two request-scoped blocks that overlap (interleaved, not nested)
        WHEN the first block exits while the second is still active
        THEN the default stays cleared until the last block exits, and only the
        original default is restored
        (regression: the later entrant saved ``None`` and restored ``None`` over
        the earlier block's restore).
        """
        mock_dr_context.use_case = "some-use-case"
        mock_dr_context._use_case = "some-use-case"

        block_a = _suspend_default_use_case()
        block_a.__enter__()
        mock_dr_context._use_case = None  # mirror what the property setter stored
        block_b = _suspend_default_use_case()
        block_b.__enter__()

        block_a.__exit__(None, None, None)
        assert mock_dr_context.use_case is None  # block B is still active

        block_b.__exit__(None, None, None)
        assert mock_dr_context.use_case == "some-use-case"

    @patch(f"{_MODULE}.client_configuration")
    @patch(f"{_MODULE}.get_credentials")
    @patch("datarobot_genai.drmcputils.auth.AuthContextHeaderHandler")
    def test_extracts_token_from_auth_context_when_no_standard_headers(
        self, mock_handler_class, mock_get_creds, mock_client_configuration
    ) -> None:
        auth_ctx = AuthCtx(
            user=User(
                id="real-user-456",
                email="jane.smith@company.com",
                name="Jane Smith",
                phone_number=None,
                given_name="Jane",
                family_name="Smith",
                profile_picture_url=None,
                metadata={},
            ),
            identities=[
                Identity(
                    id="dr-identity-1",
                    type="datarobot",
                    provider_type="datarobot_ext_email",
                    provider_user_id="jane.smith@company.com",
                    provider_identity_id=None,
                ),
            ],
            metadata={
                "dr_ctx": {"email": "jane.smith@company.com", "api_key": "auth-context-api-key"}
            },
        )
        mock_handler = Mock()
        mock_handler.get_context.return_value = auth_ctx
        mock_handler_class.return_value = mock_handler
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
        set_request_headers({"x-datarobot-authorization-context": jwt})
        mock_creds = MagicMock()
        mock_creds.datarobot.datarobot_endpoint = "https://app.datarobot.com/api/v2"
        mock_get_creds.return_value = mock_creds

        with request_user_dr_sdk(headers_auth_only=False):
            pass

        mock_client_configuration.assert_called_once_with(
            token="auth-context-api-key", endpoint="https://app.datarobot.com/api/v2"
        )
