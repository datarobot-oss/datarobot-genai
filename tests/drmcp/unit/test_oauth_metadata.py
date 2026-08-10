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
"""Assembling the published protected-resource metadata from the server's settings."""

import pytest

from datarobot_genai.drmcp.core.config import MCPServerConfig
from datarobot_genai.drmcp.core.oauth_metadata import build_protected_resource_metadata_config


def _config(**overrides: str | None) -> MCPServerConfig:
    """Build a config with every metadata field pinned, so ambient env cannot leak in."""
    fields: dict[str, str | None] = {
        "mcp_oauth_resource": None,
        "mcp_oauth_authorization_servers": None,
        "mcp_oauth_scopes_supported": None,
        "mcp_xaa_trusted_issuer": None,
        "mcp_xaa_exchange_audience": None,
        "mcp_xaa_token_url": None,
        "mcp_xaa_token_audience": None,
        "mcp_xaa_scopes": None,
        "mcp_xaa_token_endpoint_auth_method": None,
    }
    fields.update(overrides)
    return MCPServerConfig(**fields)  # type: ignore[arg-type]


class TestBuildProtectedResourceMetadataConfig:
    """One wiring shared by the well-known route and the startup validation pass."""

    def test_reads_every_setting_off_the_server_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(
            mcp_oauth_resource="https://mcp.example.com/",
            mcp_oauth_authorization_servers="https://as.example.com",
            mcp_oauth_scopes_supported="read,write",
            mcp_xaa_trusted_issuer="https://issuer.example.com",
            mcp_xaa_exchange_audience="https://audience.example.com",
            mcp_xaa_token_url="https://issuer.example.com/token",
            mcp_xaa_token_audience="https://internal.example.com",
            mcp_xaa_scopes="dr.impersonation",
            mcp_xaa_token_endpoint_auth_method="client_secret_jwt",
        )
        monkeypatch.setattr("datarobot_genai.drmcp.core.oauth_metadata.get_config", lambda: config)

        built = build_protected_resource_metadata_config()

        assert built.resource == "https://mcp.example.com/"
        assert built.authorization_servers == ["https://as.example.com"]
        assert built.scopes_supported == ["read", "write"]
        xaa = built.cross_application_access
        assert xaa is not None
        assert xaa.token_exchange.trusted_issuer == "https://issuer.example.com"
        assert xaa.token_exchange.audience == "https://audience.example.com"
        assert xaa.token_request.token_url == "https://issuer.example.com/token"
        assert xaa.token_request.audience == "https://internal.example.com"
        assert xaa.token_request.scopes == ["dr.impersonation"]
        assert xaa.token_endpoint_auth_method == "client_secret_jwt"

    def test_a_partial_xaa_config_warns_at_build_time(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The server builds this once at startup, so the warning lands in the boot log."""
        config = _config(mcp_xaa_trusted_issuer="https://issuer.example.com")
        monkeypatch.setattr("datarobot_genai.drmcp.core.oauth_metadata.get_config", lambda: config)

        built = build_protected_resource_metadata_config()

        assert built.cross_application_access is None
        assert "MCP_XAA_EXCHANGE_AUDIENCE" in caplog.text
        assert "MCP_XAA_TOKEN_URL" in caplog.text
        assert "MCP_XAA_SCOPES" in caplog.text
        assert "MCP_XAA_TRUSTED_ISSUER" not in caplog.text
