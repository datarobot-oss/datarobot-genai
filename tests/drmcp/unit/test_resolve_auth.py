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

"""Unit tests for drtools auth resolution (resolve_datarobot_token, resolve_secret)."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from datarobot_genai.drmcputils.auth import get_oauth_access_token_with_header_fallback
from datarobot_genai.drmcputils.auth import resolve_datarobot_token
from datarobot_genai.drmcputils.auth import resolve_secret
from datarobot_genai.drmcputils.auth import set_request_headers
from datarobot_genai.drmcputils.credentials import AuthResolutionStrategy
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind


def _mock_tools_credentials(
    *,
    token: str = "",
    strategy: AuthResolutionStrategy = AuthResolutionStrategy.HTTP,
) -> MagicMock:
    mock_creds = MagicMock()
    mock_creds.auth_resolution_strategy = strategy
    mock_creds.datarobot.datarobot_api_token = token
    return mock_creds


@pytest.fixture(autouse=True)
def clear_request_headers() -> None:
    set_request_headers({})
    yield
    set_request_headers({})


@pytest.fixture
def credentials_holder(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    holder: dict[str, MagicMock] = {"creds": _mock_tools_credentials()}

    def _get_credentials() -> MagicMock:
        return holder["creds"]

    monkeypatch.setattr(
        "datarobot_genai.drmcputils.auth.get_credentials",
        _get_credentials,
    )
    return holder


class TestResolveDatarobotToken:
    def test_http_uses_header(self, credentials_holder: dict[str, MagicMock]) -> None:
        credentials_holder["creds"] = _mock_tools_credentials(
            token="config-token",
            strategy=AuthResolutionStrategy.HTTP,
        )
        set_request_headers({"x-datarobot-api-token": "header-token"})

        assert resolve_datarobot_token() == "header-token"

    def test_http_ignores_config_when_header_missing(
        self, credentials_holder: dict[str, MagicMock]
    ) -> None:
        credentials_holder["creds"] = _mock_tools_credentials(
            token="config-token",
            strategy=AuthResolutionStrategy.HTTP,
        )

        assert resolve_datarobot_token() is None

    def test_config_uses_config_only(self, credentials_holder: dict[str, MagicMock]) -> None:
        credentials_holder["creds"] = _mock_tools_credentials(
            token="config-token",
            strategy=AuthResolutionStrategy.CONFIG,
        )
        set_request_headers({"x-datarobot-api-token": "header-token"})

        assert resolve_datarobot_token() == "config-token"

    @patch("datarobot_genai.drmcputils.auth.get_request_headers")
    def test_safe_request_headers_on_exception(
        self,
        mock_get_headers: MagicMock,
        credentials_holder: dict[str, MagicMock],
    ) -> None:
        credentials_holder["creds"] = _mock_tools_credentials(
            token="config-token",
            strategy=AuthResolutionStrategy.HTTP,
        )
        mock_get_headers.side_effect = RuntimeError("no context")

        assert resolve_datarobot_token() is None

    def test_raw_authorization_header_never_used(
        self, credentials_holder: dict[str, MagicMock]
    ) -> None:
        """GIVEN only a raw Authorization header (on OAuth-protected servers it
        carries the MCP access token, e.g. an Okta JWT — not a DataRobot key)
        WHEN the DataRobot token is resolved
        THEN it is never forwarded as a DR API key
        (regression: 'authorization' used to be the last token candidate).
        """
        credentials_holder["creds"] = _mock_tools_credentials(
            strategy=AuthResolutionStrategy.HTTP,
        )
        set_request_headers({"authorization": "Bearer okta-mcp-access-token"})

        assert resolve_datarobot_token() is None

    def test_datarobot_headers_used_alongside_raw_authorization(
        self, credentials_holder: dict[str, MagicMock]
    ) -> None:
        """The DataRobot-specific headers keep working when a raw Authorization
        header is also present.
        """
        credentials_holder["creds"] = _mock_tools_credentials(
            strategy=AuthResolutionStrategy.HTTP,
        )
        set_request_headers(
            {
                "authorization": "Bearer okta-mcp-access-token",
                "x-datarobot-api-token": "dr-api-token",
            }
        )

        assert resolve_datarobot_token() == "dr-api-token"


class TestResolveSecret:
    def test_http_uses_header_only(self, credentials_holder: dict[str, MagicMock]) -> None:
        credentials_holder["creds"] = _mock_tools_credentials(strategy=AuthResolutionStrategy.HTTP)
        set_request_headers({"x-tavily-api-key": "header-key"})

        assert resolve_secret("x-tavily-api-key", "config-key") == "header-key"

    def test_http_ignores_config_when_header_missing(
        self, credentials_holder: dict[str, MagicMock]
    ) -> None:
        credentials_holder["creds"] = _mock_tools_credentials(strategy=AuthResolutionStrategy.HTTP)

        assert resolve_secret("x-tavily-api-key", "config-key") is None

    def test_config_uses_config_only(self, credentials_holder: dict[str, MagicMock]) -> None:
        credentials_holder["creds"] = _mock_tools_credentials(
            strategy=AuthResolutionStrategy.CONFIG,
        )
        set_request_headers({"x-tavily-api-key": "header-key"})

        assert resolve_secret("x-tavily-api-key", "config-key") == "config-key"

    def test_empty_config_value_treated_as_missing(
        self, credentials_holder: dict[str, MagicMock]
    ) -> None:
        credentials_holder["creds"] = _mock_tools_credentials(
            strategy=AuthResolutionStrategy.CONFIG
        )

        assert resolve_secret("x-tavily-api-key", "") is None

    @patch("datarobot_genai.drmcputils.auth.get_request_headers")
    def test_safe_request_headers_on_exception(
        self,
        mock_get_headers: MagicMock,
        credentials_holder: dict[str, MagicMock],
    ) -> None:
        credentials_holder["creds"] = _mock_tools_credentials(strategy=AuthResolutionStrategy.HTTP)
        mock_get_headers.side_effect = RuntimeError("no context")

        assert resolve_secret("x-tavily-api-key", "config-key") is None


class TestGetOauthAccessTokenWithHeaderFallbackStrategy:
    @pytest.mark.asyncio
    async def test_config_strategy_returns_tool_error(
        self, credentials_holder: dict[str, MagicMock]
    ) -> None:
        credentials_holder["creds"] = _mock_tools_credentials(
            strategy=AuthResolutionStrategy.CONFIG,
        )

        out = await get_oauth_access_token_with_header_fallback(
            "google",
            display_name="Google",
            access_token_header_segment="google-drive",
        )
        assert isinstance(out, ToolError)
        assert out.kind == ToolErrorKind.AUTHENTICATION
        assert "does not support auth_resolution_strategy=config" in str(out)
        assert "x-datarobot-google-drive-access-token" in str(out)

    @pytest.mark.asyncio
    async def test_http_strategy_does_not_use_config_credentials(
        self, credentials_holder: dict[str, MagicMock]
    ) -> None:
        credentials_holder["creds"] = _mock_tools_credentials(strategy=AuthResolutionStrategy.HTTP)

        with patch(
            "datarobot_genai.drmcputils.auth.get_access_token",
            new_callable=AsyncMock,
            side_effect=RuntimeError("no context"),
        ):
            out = await get_oauth_access_token_with_header_fallback(
                "google",
                display_name="Google",
                access_token_header_segment="google-drive",
            )
        assert isinstance(out, ToolError)
        assert out.kind == ToolErrorKind.AUTHENTICATION
