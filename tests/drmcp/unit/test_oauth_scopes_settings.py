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
"""Reading scope settings out of this server's configuration."""

import json
import logging

import pytest

from datarobot_genai.drmcp.core.config import MCPServerConfig
from datarobot_genai.drmcp.core.oauth_scopes import build_scope_settings
from datarobot_genai.drmcp.core.oauth_scopes import read_tag_scopes
from datarobot_genai.drmcp.core.oauth_scopes import split_setting

DB_WRITE = "mcp:tools:database:write"
EXECUTE = "mcp:tools:execute"


def envelope(payload: str) -> str:
    """Wrap a value the way the platform actually delivers a runtime parameter."""
    return json.dumps({"type": "string", "payload": payload})


class TestSplitSetting:
    def test_splits_on_commas_and_trims(self) -> None:
        assert split_setting(f" {EXECUTE} , {DB_WRITE} ") == [EXECUTE, DB_WRITE]

    def test_blank_and_unset_are_alike(self) -> None:
        assert split_setting(None) == []
        assert split_setting("   ") == []

    def test_empty_entries_are_dropped(self) -> None:
        assert split_setting(f"{EXECUTE},,") == [EXECUTE]


class TestReadTagScopes:
    """One variable per tag, rather than one packed variable holding every tag."""

    def test_collects_one_variable_per_tag(self) -> None:
        environ = {
            "MCP_OAUTH_TAG_SCOPES_DATABASE": f"{EXECUTE},{DB_WRITE}",
            "MCP_OAUTH_TAG_SCOPES_READONLY": "mcp:tools:read",
        }

        assert read_tag_scopes(environ) == {
            "DATABASE": [EXECUTE, DB_WRITE],
            "READONLY": ["mcp:tools:read"],
        }

    def test_unrelated_variables_are_ignored(self) -> None:
        environ = {"MCP_OAUTH_RESOURCE": "https://x", "PATH": "/usr/bin"}

        assert read_tag_scopes(environ) == {}

    def test_a_blanked_out_variable_turns_the_guard_off(self) -> None:
        """GIVEN an empty value, THEN the tag guards nothing rather than guarding nothing-ness."""
        environ = {"MCP_OAUTH_TAG_SCOPES_DATABASE": "  "}

        assert read_tag_scopes(environ) == {}

    def test_datarobot_runtime_parameter_spelling_is_accepted(self) -> None:
        """The platform exposes runtime parameters with its own prefix — and an envelope.

        The serverless path never delivers the bare value: the env var holds
        ``{"type": "string", "payload": ...}``, exactly what
        ``datarobot.core.config.getenv`` unwraps for declared settings fields.
        """
        environ = {
            "MLOPS_RUNTIME_PARAM_MCP_OAUTH_TAG_SCOPES_DATABASE": envelope(f"{EXECUTE},{DB_WRITE}")
        }

        assert read_tag_scopes(environ) == {"DATABASE": [EXECUTE, DB_WRITE]}

    def test_a_bare_runtime_parameter_value_still_works(self) -> None:
        """No platform contract says the envelope is forever; a bare value must not break."""
        environ = {"MLOPS_RUNTIME_PARAM_MCP_OAUTH_TAG_SCOPES_DATABASE": DB_WRITE}

        assert read_tag_scopes(environ) == {"DATABASE": [DB_WRITE]}

    def test_an_envelope_holding_nothing_turns_the_guard_off(self) -> None:
        environ = {"MLOPS_RUNTIME_PARAM_MCP_OAUTH_TAG_SCOPES_DATABASE": envelope("")}

        assert read_tag_scopes(environ) == {}

    def test_a_direct_variable_is_never_unwrapped(self) -> None:
        """Only the runtime-parameter spelling implies the platform's envelope."""
        environ = {"MCP_OAUTH_TAG_SCOPES_DATABASE": envelope(DB_WRITE)}

        assert read_tag_scopes(environ) != {"DATABASE": [DB_WRITE]}

    def test_the_suffix_is_the_tag(self) -> None:
        environ = {"MCP_OAUTH_TAG_SCOPES_READ_ONLY": "mcp:tools:read"}

        assert list(read_tag_scopes(environ)) == ["READ_ONLY"]


def _config(**overrides: str) -> MCPServerConfig:
    """Build a config with every scope-relevant field pinned, so ambient env cannot leak in."""
    fields: dict[str, str | None] = {
        "mcp_oauth_authorization_servers": None,
        "mcp_oauth_resource": None,
        "mcp_oauth_audience": None,
        "mcp_oauth_jwks_uri": None,
        "mcp_oauth_scope_source": None,
    }
    fields.update(overrides)
    return MCPServerConfig(**fields)  # type: ignore[arg-type]


class TestBuildScopeSettings:
    """The audience follows the same chain the published ``resource`` does."""

    @pytest.fixture(autouse=True)
    def _off_platform(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Start every test outside a DataRobot runtime; opt back in per test."""
        for name in (
            "MLOPS_DEPLOYMENT_ID",
            "WORKLOAD_ID",
            "DATAROBOT_PUBLIC_API_ENDPOINT",
            "DATAROBOT_ENDPOINT",
        ):
            monkeypatch.delenv(name, raising=False)

    def test_an_explicit_audience_wins(self) -> None:
        settings = build_scope_settings(
            _config(mcp_oauth_audience="https://aud", mcp_oauth_resource="https://res")
        )

        assert settings.audience == "https://aud"

    def test_the_resource_is_the_default_audience(self) -> None:
        settings = build_scope_settings(_config(mcp_oauth_resource="https://res"))

        assert settings.audience == "https://res"

    def test_the_runtime_resolved_url_is_the_last_resort(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN a deployment that lets `resource` resolve at runtime.

        The published document will carry the resolved URL as `resource`, so a
        discovering client mints its token with that `aud` — the server must
        check against the same value, or it publishes an identity it never
        verifies.
        """
        monkeypatch.setenv("MLOPS_DEPLOYMENT_ID", "abc123")
        monkeypatch.setenv("DATAROBOT_ENDPOINT", "https://dr.example.com/api/v2")

        settings = build_scope_settings(
            _config(mcp_oauth_authorization_servers="https://idp.example.com/oauth2/aus1")
        )

        assert settings.audience == (
            "https://dr.example.com/api/v2/deployments/abc123/directAccess/mcp"
        )
        assert settings.enforced, "setting the authorization server alone activates verification"

    def test_off_platform_the_audience_stays_unset(self) -> None:
        settings = build_scope_settings(
            _config(mcp_oauth_authorization_servers="https://idp.example.com/oauth2/aus1")
        )

        assert settings.audience is None
        assert not settings.enforced

    def test_the_first_authorization_server_is_the_issuer(self) -> None:
        settings = build_scope_settings(
            _config(mcp_oauth_authorization_servers="https://one,https://two")
        )

        assert settings.issuer == "https://one"

    def test_more_than_one_authorization_server_is_called_out(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Only the first can ever verify — one JWKS URI serves one issuer."""
        with caplog.at_level(logging.WARNING, logger="datarobot_genai.drmcp.core.oauth_scopes"):
            build_scope_settings(_config(mcp_oauth_authorization_servers="https://one,https://two"))

        assert "verified against the first" in caplog.text

    def test_a_single_authorization_server_is_not_warned_at(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="datarobot_genai.drmcp.core.oauth_scopes"):
            build_scope_settings(_config(mcp_oauth_authorization_servers="https://one"))

        assert caplog.text == ""
