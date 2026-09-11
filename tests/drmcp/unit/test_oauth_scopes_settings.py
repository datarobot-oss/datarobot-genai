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
from datarobot_genai.drmcpbase.oauth_scopes import ScopeSource

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
        "mcp_oauth_scope_source": None,
    }
    fields.update(overrides)
    return MCPServerConfig(**fields)  # type: ignore[arg-type]


class TestBuildScopeSettings:
    """Declarations only — which mechanism is read, and the tag-keyed rules."""

    def test_the_scope_source_is_read_off_the_config(self) -> None:
        settings = build_scope_settings(_config(mcp_oauth_scope_source="tags"))

        assert settings.source is ScopeSource.TAGS

    def test_the_source_defaults_to_both(self) -> None:
        settings = build_scope_settings(_config())

        assert settings.source is ScopeSource.BOTH

    def test_tag_scopes_are_read_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MCP_OAUTH_TAG_SCOPES_DATABASE", f"{EXECUTE},{DB_WRITE}")

        settings = build_scope_settings(_config())

        assert settings.tag_scopes == {"DATABASE": [EXECUTE, DB_WRITE]}


class TestWireScopesEnforcementWarning:
    """Declared scopes with the enforcement gate off are called out at startup."""

    _LOGGER = "datarobot_genai.drmcp.core.oauth_scopes"

    @pytest.fixture
    def guarded_server(self):  # type: ignore[no-untyped-def]
        from fastmcp import FastMCP

        from datarobot_genai.drmcpbase.oauth_scopes import required_scopes_check
        from datarobot_genai.drmcpbase.oauth_scopes import reset_scope_state

        mcp: FastMCP = FastMCP("gate-test")

        @mcp.tool(auth=required_scopes_check(EXECUTE))
        def guarded() -> str:
            """Declare a scope; what ``required_scopes=(EXECUTE,)`` attaches."""
            return "ok"

        yield mcp
        reset_scope_state()

    async def test_declared_scopes_with_the_gate_off_are_called_out(
        self,
        guarded_server,
        caplog: pytest.LogCaptureFixture,  # type: ignore[no-untyped-def]
    ) -> None:
        from datarobot_genai.drmcp.core.oauth_scopes import wire_scopes

        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            await wire_scopes(guarded_server, _config(mcp_enable_oauth_claim_validation=False))

        assert any(
            "MCP_ENABLE_OAUTH_CLAIM_VALIDATION is off" in r.message and EXECUTE in r.message
            for r in caplog.records
        )

    async def test_no_warning_when_the_gate_is_on(
        self,
        guarded_server,
        caplog: pytest.LogCaptureFixture,  # type: ignore[no-untyped-def]
    ) -> None:
        from datarobot_genai.drmcp.core.oauth_scopes import wire_scopes

        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            await wire_scopes(guarded_server, _config(mcp_enable_oauth_claim_validation=True))

        assert not [r for r in caplog.records if "MCP_ENABLE_OAUTH_CLAIM_VALIDATION" in r.message]

    async def test_no_warning_when_nothing_is_declared(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from fastmcp import FastMCP

        from datarobot_genai.drmcp.core.oauth_scopes import wire_scopes
        from datarobot_genai.drmcpbase.oauth_scopes import reset_scope_state

        try:
            with caplog.at_level(logging.WARNING, logger=self._LOGGER):
                await wire_scopes(
                    FastMCP("empty"), _config(mcp_enable_oauth_claim_validation=False)
                )
        finally:
            reset_scope_state()

        assert not [r for r in caplog.records if "MCP_ENABLE_OAUTH_CLAIM_VALIDATION" in r.message]
