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
"""Per-component OAuth scope declarations.

Declarations only: enforcement is the scope-validation middleware's job (see
``drmcp.core.middleware``), which checks each ``tools/call`` against
``declared_scopes_for_one_tool``. Nothing here may gate a caller at the tool
level — that is the removed behaviour these tests pin.
"""

import logging
from collections.abc import Iterator
from typing import Any

import pytest
from fastmcp import FastMCP
from fastmcp.server.auth import AuthContext

from datarobot_genai.drmcpbase.oauth_scopes import DECLARED_SCOPES_ATTR
from datarobot_genai.drmcpbase.oauth_scopes import TAG_APPLIED_ATTR
from datarobot_genai.drmcpbase.oauth_scopes import ScopeSettings
from datarobot_genai.drmcpbase.oauth_scopes import ScopeSource
from datarobot_genai.drmcpbase.oauth_scopes import collect_code_declared_scopes
from datarobot_genai.drmcpbase.oauth_scopes import configure_scopes
from datarobot_genai.drmcpbase.oauth_scopes import declared_scopes_for_one_tool
from datarobot_genai.drmcpbase.oauth_scopes import declared_scopes_of_component
from datarobot_genai.drmcpbase.oauth_scopes import derived_scopes
from datarobot_genai.drmcpbase.oauth_scopes import normalize_tag
from datarobot_genai.drmcpbase.oauth_scopes import report_foreign_auth_checks
from datarobot_genai.drmcpbase.oauth_scopes import required_scopes_check
from datarobot_genai.drmcpbase.oauth_scopes import reset_scope_state
from datarobot_genai.drmcpbase.oauth_scopes import restrict_tag_scopes
from datarobot_genai.drmcpbase.oauth_scopes import wire_scopes

EXECUTE = "mcp:tools:execute"
DB_WRITE = "mcp:tools:database:write"
READ = "mcp:tools:read"

SCOPES_LOGGER = "datarobot_genai.drmcpbase.oauth_scopes"


@pytest.fixture(autouse=True)
def _clean_scope_state() -> Iterator[None]:
    """GIVEN no scope settings left over from another test."""
    reset_scope_state()
    yield
    reset_scope_state()


@pytest.fixture
def mcp() -> FastMCP:
    """GIVEN a server with a code-declared tool, a tag-only tool, and an open one."""
    server: FastMCP = FastMCP(name="test")

    # What ``@dr_mcp_tool(required_scopes=(EXECUTE,))`` attaches after conversion.
    @server.tool(tags={"database"}, auth=required_scopes_check(EXECUTE))
    def run_sql() -> str:
        """Declare a scope in code, and carry a mappable tag as well."""
        return "ok"

    @server.tool(tags={"database"})
    def list_tables() -> str:
        """Carries the tag but declares nothing itself."""
        return "ok"

    @server.tool
    def harmless() -> str:
        """Guarded by nothing."""
        return "ok"

    return server


async def _visible(mcp: FastMCP) -> set[str]:
    return {tool.name for tool in await mcp.list_tools()}


async def _checks_on(mcp: FastMCP, name: str) -> list[Any]:
    tool = next(t for t in await mcp._list_tools() if t.name == name)
    auth = tool.auth or []
    return [auth] if callable(auth) else list(auth)


async def _context(mcp: FastMCP, name: str = "run_sql") -> AuthContext:
    tool = next(t for t in await mcp._list_tools() if t.name == name)
    return AuthContext(token=None, component=tool)


class TestNormalizeTag:
    @pytest.mark.parametrize(
        ("written", "expected"),
        [("read-only", "READ_ONLY"), ("Read_Only", "READ_ONLY"), ("  database ", "DATABASE")],
    )
    def test_folds_case_and_dashes(self, written: str, expected: str) -> None:
        assert normalize_tag(written) == expected

    def test_dash_and_underscore_spellings_are_the_same_tag(self) -> None:
        assert normalize_tag("read-only") == normalize_tag("read_only")


class TestScopeSource:
    def test_defaults_to_both(self) -> None:
        assert ScopeSource.parse(None) is ScopeSource.BOTH
        assert ScopeSource.parse("") is ScopeSource.BOTH

    @pytest.mark.parametrize(
        ("value", "member"),
        [("code", ScopeSource.CODE), ("TAGS", ScopeSource.TAGS), (" Both ", ScopeSource.BOTH)],
    )
    def test_accepts_known_values_case_insensitively(self, value: str, member: ScopeSource) -> None:
        assert ScopeSource.parse(value) is member

    def test_unknown_value_falls_back_rather_than_raising(self) -> None:
        assert ScopeSource.parse("everything") is ScopeSource.BOTH

    @pytest.mark.parametrize(
        ("source", "reads_code", "reads_tags"),
        [
            (ScopeSource.CODE, True, False),
            (ScopeSource.TAGS, False, True),
            (ScopeSource.BOTH, True, True),
        ],
    )
    def test_reads_flags(self, source: ScopeSource, reads_code: bool, reads_tags: bool) -> None:
        assert source.reads_code is reads_code
        assert source.reads_tags is reads_tags


class TestScopeSettings:
    def test_tag_keys_are_normalized_on_construction(self) -> None:
        settings = ScopeSettings(tag_scopes={"read-only": [READ]})
        assert settings.tag_scopes == {"READ_ONLY": [READ]}


class TestDeclarationOnlyChecks:
    """The attached checks record scopes but never gate a caller."""

    async def test_required_scopes_check_admits_every_caller(self, mcp: FastMCP) -> None:
        # GIVEN a caller presenting nothing at all (no auth provider, no token)
        ctx = await _context(mcp, "run_sql")
        # WHEN the code-declared check runs
        (check,) = await _checks_on(mcp, "run_sql")
        # THEN it admits the caller — enforcement is the middleware's job
        assert await check(ctx) is True

    def test_required_scopes_check_records_the_declared_names(self) -> None:
        check = required_scopes_check(EXECUTE, DB_WRITE)
        assert getattr(check, DECLARED_SCOPES_ATTR) == frozenset({EXECUTE, DB_WRITE})

    async def test_restrict_tag_scopes_admits_every_caller(self, mcp: FastMCP) -> None:
        await wire_scopes(mcp, ScopeSettings(tag_scopes={"database": [DB_WRITE]}))
        ctx = await _context(mcp, "list_tables")
        (check,) = await _checks_on(mcp, "list_tables")
        assert await check(ctx) is True

    def test_restrict_tag_scopes_is_marked_as_configuration(self) -> None:
        check = restrict_tag_scopes("database", [DB_WRITE])
        assert getattr(check, TAG_APPLIED_ATTR, False)
        assert getattr(check, DECLARED_SCOPES_ATTR) == frozenset({DB_WRITE})

    async def test_every_tool_stays_listed_whatever_it_declares(self, mcp: FastMCP) -> None:
        # GIVEN declarations from both mechanisms and a caller with no token
        await wire_scopes(mcp, ScopeSettings(tag_scopes={"database": [DB_WRITE]}))
        # THEN nothing is hidden at the tool level — there is no tool-level gate
        assert await _visible(mcp) == {"run_sql", "list_tables", "harmless"}


class TestForeignAuthChecks:
    """FastMCP-native (or custom) auth checks are reported: they gate at the tool level."""

    async def test_a_fastmcp_native_check_is_reported_by_name(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from fastmcp.server.auth import require_scopes as fastmcp_require_scopes

        server: FastMCP = FastMCP(name="foreign")

        @server.tool(auth=fastmcp_require_scopes(EXECUTE))
        def gated() -> str:
            """Guarded by FastMCP's own check — a tool-level gate on ctx.token."""
            return "ok"

        @server.tool(auth=required_scopes_check(EXECUTE))
        def declared() -> str:
            """Declare scopes our way — reported to the middleware, never hidden."""
            return "ok"

        with caplog.at_level(logging.WARNING, logger=SCOPES_LOGGER):
            affected = await report_foreign_auth_checks(server)
        assert affected == ["gated"]
        assert any("gated" in r.message and "hidden" in r.message for r in caplog.records)
        # And the behaviour the warning describes for the gate-off (default) shape:
        # no token-handler middleware ran, so FastMCP's ctx.token is None, its check
        # fails, and the tool is gone from tools/list for everyone.
        assert await _visible(server) == {"declared"}

    async def test_our_declarations_are_not_reported(self, mcp: FastMCP) -> None:
        await wire_scopes(mcp, ScopeSettings(tag_scopes={"database": [DB_WRITE]}))
        assert await report_foreign_auth_checks(mcp) == []


class TestCodeDeclaredScopes:
    async def test_scopes_are_readable_back_off_the_component(self, mcp: FastMCP) -> None:
        found = await collect_code_declared_scopes(mcp)
        assert found == {EXECUTE}

    async def test_tag_checks_do_not_masquerade_as_code_declarations(self, mcp: FastMCP) -> None:
        # GIVEN a tag rule active alongside the code declaration
        await wire_scopes(mcp, ScopeSettings(tag_scopes={"database": [DB_WRITE]}))
        # THEN the code collector reports only the in-code scope
        assert await collect_code_declared_scopes(mcp) == {EXECUTE}

    async def test_they_reach_the_published_list(self, mcp: FastMCP) -> None:
        await wire_scopes(mcp, ScopeSettings())
        assert derived_scopes() == [EXECUTE]


class TestTagScopes:
    async def test_declares_on_every_component_carrying_the_tag(self, mcp: FastMCP) -> None:
        await wire_scopes(mcp, ScopeSettings(tag_scopes={"database": [DB_WRITE]}))
        assert await declared_scopes_for_one_tool(mcp, "run_sql") == {EXECUTE, DB_WRITE}
        assert await declared_scopes_for_one_tool(mcp, "list_tables") == {DB_WRITE}

    async def test_untagged_components_are_unaffected(self, mcp: FastMCP) -> None:
        await wire_scopes(mcp, ScopeSettings(tag_scopes={"database": [DB_WRITE]}))
        assert await declared_scopes_for_one_tool(mcp, "harmless") == frozenset()
        assert await _checks_on(mcp, "harmless") == []

    async def test_env_var_spelling_matches_the_component_tag(self, mcp: FastMCP) -> None:
        # GIVEN the tag arrives in env-var spelling (upper case, underscores)
        await wire_scopes(mcp, ScopeSettings(tag_scopes={"DATABASE": [DB_WRITE]}))
        # THEN it matches the component's lower-case tag
        assert await declared_scopes_for_one_tool(mcp, "list_tables") == {DB_WRITE}

    async def test_tag_scopes_are_inert_under_code(self, mcp: FastMCP) -> None:
        await wire_scopes(
            mcp,
            ScopeSettings(source=ScopeSource.CODE, tag_scopes={"database": [DB_WRITE]}),
        )
        # No tag check is attached at all under source=code.
        assert await declared_scopes_for_one_tool(mcp, "list_tables") == frozenset()
        assert await _checks_on(mcp, "list_tables") == []

    async def test_tags_apply_under_the_default_source(self, mcp: FastMCP) -> None:
        await wire_scopes(mcp, ScopeSettings(tag_scopes={"database": [DB_WRITE]}))
        assert DB_WRITE in (await declared_scopes_for_one_tool(mcp, "list_tables") or set())

    async def test_rewiring_does_not_stack_duplicate_checks(self, mcp: FastMCP) -> None:
        settings = ScopeSettings(tag_scopes={"database": [DB_WRITE]})
        await wire_scopes(mcp, settings)
        await wire_scopes(mcp, settings)
        checks = await _checks_on(mcp, "list_tables")
        assert len(checks) == 1

    async def test_switching_away_from_tags_removes_the_tag_check(self, mcp: FastMCP) -> None:
        await wire_scopes(mcp, ScopeSettings(tag_scopes={"database": [DB_WRITE]}))
        await wire_scopes(
            mcp, ScopeSettings(source=ScopeSource.CODE, tag_scopes={"database": [DB_WRITE]})
        )
        assert await _checks_on(mcp, "list_tables") == []

    async def test_a_code_declaration_survives_tag_rewiring(self, mcp: FastMCP) -> None:
        await wire_scopes(mcp, ScopeSettings(tag_scopes={"database": [DB_WRITE]}))
        await wire_scopes(mcp, ScopeSettings(tag_scopes={"database": [DB_WRITE]}))
        checks = await _checks_on(mcp, "run_sql")
        declared = {
            scope for check in checks for scope in getattr(check, DECLARED_SCOPES_ATTR, frozenset())
        }
        assert EXECUTE in declared


class TestDeclaredScopesForOneTool:
    """The middleware's source of truth for what a ``tools/call`` must cover."""

    async def test_unknown_tool_is_none_not_empty(self, mcp: FastMCP) -> None:
        # None means "no such tool" — the middleware skips validation entirely;
        # an empty set would mean "exists and requires nothing".
        assert await declared_scopes_for_one_tool(mcp, "no_such_tool") is None

    async def test_an_undeclared_tool_requires_nothing(self, mcp: FastMCP) -> None:
        assert await declared_scopes_for_one_tool(mcp, "harmless") == frozenset()

    async def test_a_component_in_both_places_requires_the_union(self, mcp: FastMCP) -> None:
        await wire_scopes(mcp, ScopeSettings(tag_scopes={"database": [DB_WRITE]}))
        assert await declared_scopes_for_one_tool(mcp, "run_sql") == {EXECUTE, DB_WRITE}

    async def test_code_declarations_are_silenced_under_tags(self, mcp: FastMCP) -> None:
        # GIVEN source=tags, which silences in-code declarations
        await wire_scopes(
            mcp,
            ScopeSettings(source=ScopeSource.TAGS, tag_scopes={"database": [DB_WRITE]}),
        )
        # THEN the middleware is told about the tag rule only — a silenced
        # declaration must not be enforced either.
        assert await declared_scopes_for_one_tool(mcp, "run_sql") == {DB_WRITE}

    async def test_code_declarations_are_read_under_code(self, mcp: FastMCP) -> None:
        await wire_scopes(
            mcp,
            ScopeSettings(source=ScopeSource.CODE, tag_scopes={"database": [DB_WRITE]}),
        )
        assert await declared_scopes_for_one_tool(mcp, "run_sql") == {EXECUTE}


class TestDeclaredScopesOfComponent:
    """One reader for both consumers: the middleware and the static-catalog route."""

    async def test_reads_the_union_off_a_component(self, mcp: FastMCP) -> None:
        await wire_scopes(mcp, ScopeSettings(tag_scopes={"database": [DB_WRITE]}))
        tool = next(t for t in await mcp._list_tools() if t.name == "run_sql")
        assert declared_scopes_of_component(tool) == {EXECUTE, DB_WRITE}

    def test_a_component_with_no_auth_declares_nothing(self) -> None:
        # The static-catalog route hands it stand-ins for hosted tools too;
        # anything without checks (or without an auth attribute) reads as empty.
        assert declared_scopes_of_component(object()) == frozenset()

    async def test_agrees_with_the_middleware_lookup(self, mcp: FastMCP) -> None:
        # What the REST route reports per tool must be what the middleware
        # enforces — same function underneath, pinned here anyway.
        await wire_scopes(mcp, ScopeSettings(tag_scopes={"database": [DB_WRITE]}))
        for tool in await mcp._list_tools():
            assert declared_scopes_of_component(tool) == await declared_scopes_for_one_tool(
                mcp, tool.name
            )


class TestDerivedScopes:
    async def test_nothing_is_advertised_when_nothing_is_declared(self) -> None:
        configure_scopes(ScopeSettings())
        assert derived_scopes() == []

    async def test_sorted_and_deduplicated(self, mcp: FastMCP) -> None:
        await wire_scopes(
            mcp,
            ScopeSettings(tag_scopes={"database": [DB_WRITE, EXECUTE]}),
        )
        assert derived_scopes() == sorted({EXECUTE, DB_WRITE})

    async def test_published_list_is_the_union_of_both_sources(self, mcp: FastMCP) -> None:
        await wire_scopes(mcp, ScopeSettings(tag_scopes={"database": [DB_WRITE]}))
        assert derived_scopes() == sorted({EXECUTE, DB_WRITE})

    async def test_the_source_narrows_the_published_list(self, mcp: FastMCP) -> None:
        await wire_scopes(
            mcp,
            ScopeSettings(source=ScopeSource.TAGS, tag_scopes={"database": [DB_WRITE]}),
        )
        assert derived_scopes() == [DB_WRITE]


class TestStartupValidation:
    async def test_a_tag_matching_no_component_is_reported(
        self, mcp: FastMCP, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=SCOPES_LOGGER):
            await wire_scopes(mcp, ScopeSettings(tag_scopes={"datbase": [DB_WRITE]}))
        assert any("DATBASE" in record.message for record in caplog.records)

    async def test_the_tags_actually_in_use_are_reported_alongside(
        self, mcp: FastMCP, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=SCOPES_LOGGER):
            await wire_scopes(mcp, ScopeSettings(tag_scopes={"datbase": [DB_WRITE]}))
        assert any("DATABASE" in record.message for record in caplog.records)

    async def test_a_tag_that_matches_is_not_reported(
        self, mcp: FastMCP, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=SCOPES_LOGGER):
            await wire_scopes(mcp, ScopeSettings(tag_scopes={"database": [DB_WRITE]}))
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    async def test_requirements_shadowed_by_the_scope_source_are_reported(
        self, mcp: FastMCP, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=SCOPES_LOGGER):
            await wire_scopes(
                mcp,
                ScopeSettings(source=ScopeSource.CODE, tag_scopes={"database": [DB_WRITE]}),
            )
        assert any("inert" in record.message for record in caplog.records)

    async def test_code_declarations_shadowed_by_the_scope_source_are_reported(
        self, mcp: FastMCP, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=SCOPES_LOGGER):
            await wire_scopes(mcp, ScopeSettings(source=ScopeSource.TAGS))
        assert any("declared in code are inert" in record.message for record in caplog.records)

    async def test_active_tag_scopes_are_not_reported_as_inert_code(
        self, mcp: FastMCP, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Tag-applied checks record scopes too; they must not be re-reported as
        # code declarations shadowed under source=tags.
        await wire_scopes(
            mcp,
            ScopeSettings(source=ScopeSource.TAGS, tag_scopes={"database": [DB_WRITE]}),
        )
        with caplog.at_level(logging.WARNING, logger=SCOPES_LOGGER):
            await wire_scopes(mcp)
        inert_code = [r for r in caplog.records if "declared in code are inert" in r.message]
        assert all(DB_WRITE not in r.message for r in inert_code)

    async def test_declared_requirements_are_reported(
        self, mcp: FastMCP, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger=SCOPES_LOGGER):
            await wire_scopes(mcp, ScopeSettings(tag_scopes={"database": [DB_WRITE]}))
        assert any(
            "scope requirements declared" in record.message.lower() for record in caplog.records
        )
