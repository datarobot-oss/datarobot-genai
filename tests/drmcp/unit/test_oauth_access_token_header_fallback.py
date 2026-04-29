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

"""Unit tests for OAuth access-token header fallback helpers in drtools.core.auth."""

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from datarobot.auth.datarobot.exceptions import OAuthServiceClientErr

from datarobot_genai.drtools.core.auth import get_oauth_access_token_with_header_fallback
from datarobot_genai.drtools.core.auth import oauth_access_token_header_name
from datarobot_genai.drtools.core.auth import resolve_oauth_access_token_from_headers
from datarobot_genai.drtools.core.auth import set_request_headers_for_context
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind


@pytest.fixture(autouse=True)
def clear_request_headers_ctx() -> None:
    """Avoid leaking request header context between tests."""
    yield
    set_request_headers_for_context({})


class TestOauthAccessTokenHeaderName:
    def test_segment_lowercase_and_hyphens(self) -> None:
        assert (
            oauth_access_token_header_name("google-drive")
            == "x-datarobot-google-drive-access-token"
        )

    def test_underscores_normalized(self) -> None:
        assert (
            oauth_access_token_header_name("microsoft_graph")
            == "x-datarobot-microsoft-graph-access-token"
        )

    def test_whitespace_trimmed(self) -> None:
        assert (
            oauth_access_token_header_name("  Atlassian  ") == "x-datarobot-atlassian-access-token"
        )


class TestResolveOauthAccessTokenFromHeaders:
    def test_reads_from_framework_headers(self) -> None:
        with patch(
            "datarobot_genai.drtools.core.auth._get_http_headers",
            return_value={"x-datarobot-google-drive-access-token": "tok-from-fw"},
        ):
            assert resolve_oauth_access_token_from_headers("google-drive") == "tok-from-fw"

    def test_bearer_prefix_stripped(self) -> None:
        with patch(
            "datarobot_genai.drtools.core.auth._get_http_headers",
            return_value={"x-datarobot-google-drive-access-token": "Bearer abc.def"},
        ):
            assert resolve_oauth_access_token_from_headers("google-drive") == "abc.def"

    def test_mixed_case_header_key(self) -> None:
        with patch(
            "datarobot_genai.drtools.core.auth._get_http_headers",
            return_value={"X-DataRobot-Google-Drive-Access-Token": "plain"},
        ):
            assert resolve_oauth_access_token_from_headers("google-drive") == "plain"

    def test_framework_headers_take_priority_over_context(self) -> None:
        set_request_headers_for_context(
            {"x-datarobot-google-drive-access-token": "from-ctx"},
        )
        with patch(
            "datarobot_genai.drtools.core.auth._get_http_headers",
            return_value={"x-datarobot-google-drive-access-token": "from-fw"},
        ):
            assert resolve_oauth_access_token_from_headers("google-drive") == "from-fw"

    def test_falls_back_to_request_headers_context(self) -> None:
        with patch("datarobot_genai.drtools.core.auth._get_http_headers", return_value={}):
            set_request_headers_for_context(
                {"x-datarobot-microsoft-graph-access-token": "ctx-token"},
            )
            assert resolve_oauth_access_token_from_headers("microsoft-graph") == "ctx-token"

    def test_returns_none_when_missing(self) -> None:
        with patch("datarobot_genai.drtools.core.auth._get_http_headers", return_value={}):
            set_request_headers_for_context({})
            assert resolve_oauth_access_token_from_headers("google-drive") is None

    def test_get_http_headers_exception_falls_back_to_context(self) -> None:
        with patch(
            "datarobot_genai.drtools.core.auth._get_http_headers",
            side_effect=RuntimeError("no http"),
        ):
            set_request_headers_for_context(
                {"x-datarobot-atlassian-access-token": "atlas"},
            )
            assert resolve_oauth_access_token_from_headers("atlassian") == "atlas"


class TestGetOauthAccessTokenWithHeaderFallback:
    @pytest.mark.asyncio
    async def test_returns_obo_token_when_oauth_succeeds(self) -> None:
        with patch(
            "datarobot_genai.drtools.core.auth.get_access_token",
            new_callable=AsyncMock,
            return_value="obo-secret",
        ):
            out = await get_oauth_access_token_with_header_fallback(
                "google",
                display_name="Google",
                access_token_header_segment="google-drive",
            )
        assert out == "obo-secret"

    @pytest.mark.asyncio
    async def test_returns_header_token_when_runtime_error(self) -> None:
        with patch(
            "datarobot_genai.drtools.core.auth.get_access_token",
            new_callable=AsyncMock,
            side_effect=RuntimeError("no context"),
        ):
            set_request_headers_for_context(
                {"x-datarobot-google-drive-access-token": "fallback"},
            )
            out = await get_oauth_access_token_with_header_fallback(
                "google",
                display_name="Google",
                access_token_header_segment="google-drive",
            )
        assert out == "fallback"

    @pytest.mark.asyncio
    async def test_returns_header_token_when_oauth_service_client_err(self) -> None:
        with patch(
            "datarobot_genai.drtools.core.auth.get_access_token",
            new_callable=AsyncMock,
            side_effect=OAuthServiceClientErr("not granted"),
        ):
            with patch(
                "datarobot_genai.drtools.core.auth._get_http_headers",
                return_value={"x-datarobot-google-drive-access-token": "hdr"},
            ):
                out = await get_oauth_access_token_with_header_fallback(
                    "google",
                    display_name="Google",
                    access_token_header_segment="google-drive",
                )
        assert out == "hdr"

    @pytest.mark.asyncio
    async def test_returns_header_when_oauth_returns_empty_string(self) -> None:
        with patch(
            "datarobot_genai.drtools.core.auth.get_access_token",
            new_callable=AsyncMock,
            return_value="",
        ):
            set_request_headers_for_context(
                {"x-datarobot-atlassian-access-token": "atok"},
            )
            out = await get_oauth_access_token_with_header_fallback(
                "atlassian",
                display_name="Atlassian",
                access_token_header_segment="atlassian",
            )
        assert out == "atok"

    @pytest.mark.asyncio
    async def test_tool_error_when_oauth_fails_and_no_header(self) -> None:
        with patch(
            "datarobot_genai.drtools.core.auth.get_access_token",
            new_callable=AsyncMock,
            side_effect=OAuthServiceClientErr("denied"),
        ):
            with patch("datarobot_genai.drtools.core.auth._get_http_headers", return_value={}):
                out = await get_oauth_access_token_with_header_fallback(
                    "microsoft",
                    display_name="Microsoft",
                    access_token_header_segment="microsoft-graph",
                )
        assert isinstance(out, ToolError)
        assert out.kind == ToolErrorKind.AUTHENTICATION
        assert "x-datarobot-microsoft-graph-access-token" in str(out)

    @pytest.mark.asyncio
    async def test_tool_error_internal_on_unexpected_exception(self) -> None:
        with patch(
            "datarobot_genai.drtools.core.auth.get_access_token",
            new_callable=AsyncMock,
            side_effect=ValueError("boom"),
        ):
            out = await get_oauth_access_token_with_header_fallback(
                "google",
                display_name="Google",
                access_token_header_segment="google-drive",
            )
        assert isinstance(out, ToolError)
        assert out.kind == ToolErrorKind.INTERNAL
