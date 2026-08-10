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
"""Per-component OAuth scope requirements."""

import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from fastmcp import FastMCP
from fastmcp.server.auth import AuthContext
from fastmcp.server.auth.providers.jwt import JWTVerifier
from fastmcp.server.auth.providers.jwt import RSAKeyPair
from mcp.server.auth.provider import AccessToken

from datarobot_genai.drmcpbase.oauth_scopes import DECLARED_SCOPES_ATTR
from datarobot_genai.drmcpbase.oauth_scopes import TOKEN_FLOOR_ATTR
from datarobot_genai.drmcpbase.oauth_scopes import ScopeSettings
from datarobot_genai.drmcpbase.oauth_scopes import ScopeSource
from datarobot_genai.drmcpbase.oauth_scopes import _presented_tokens
from datarobot_genai.drmcpbase.oauth_scopes import apply_tag_scopes
from datarobot_genai.drmcpbase.oauth_scopes import apply_token_floor
from datarobot_genai.drmcpbase.oauth_scopes import collect_code_declared_scopes
from datarobot_genai.drmcpbase.oauth_scopes import configure_scopes
from datarobot_genai.drmcpbase.oauth_scopes import derived_scopes
from datarobot_genai.drmcpbase.oauth_scopes import normalize_tag
from datarobot_genai.drmcpbase.oauth_scopes import probe_verification_keys
from datarobot_genai.drmcpbase.oauth_scopes import request_scopes
from datarobot_genai.drmcpbase.oauth_scopes import require_scopes
from datarobot_genai.drmcpbase.oauth_scopes import reset_scope_state
from datarobot_genai.drmcpbase.oauth_scopes import satisfies
from datarobot_genai.drmcpbase.oauth_scopes import wire_scopes

EXECUTE = "mcp:tools:execute"
DB_WRITE = "mcp:tools:database:write"
READ = "mcp:tools:read"

ISSUER = "https://issuer.example.com/oauth2/aus1"
AUDIENCE = "https://mcp.example.com/mcp"

SCOPES_LOGGER = "datarobot_genai.drmcpbase.oauth_scopes"

#: The header a DataRobot agent forwards the end user's IdP token in.
DR_HEADER = "x-datarobot-external-access-token"

#: The global MCP path's header, which this server deliberately does not read.
GATEWAY_HEADER = "x-datarobot-authorization"

#: Shaped like a DataRobot API token: opaque, so nothing can be read off it.
API_KEY = "NjM5YmY2ZGQ4YTFmMmM0ZTdkOWIwYTMx"


def enforcing(**overrides: Any) -> ScopeSettings:
    """GIVEN a server that can verify tokens, and therefore enforces requirements."""
    return ScopeSettings(issuer=ISSUER, audience=AUDIENCE, **overrides)


def _presenting(monkeypatch: pytest.MonkeyPatch, headers: dict[str, str]) -> None:
    """GIVEN an HTTP request arriving with these headers."""
    monkeypatch.setattr(
        "datarobot_genai.drmcpbase.oauth_scopes.get_http_headers",
        lambda include_all=False: headers,  # noqa: ARG005 - matches the real signature
    )


def _token(keys: RSAKeyPair, scopes: list[str], **claims: Any) -> str:
    """Mint a real RS256 token from that key, ours unless a claim is overridden."""
    return keys.create_token(**{"issuer": ISSUER, "audience": AUDIENCE, "scopes": scopes, **claims})


async def _context(mcp: FastMCP, name: str = "run_sql") -> AuthContext:
    """GIVEN no auth provider, so nothing has populated the token for us."""
    tool = next(t for t in await mcp._list_tools() if t.name == name)
    return AuthContext(token=None, component=tool)


@pytest.fixture(autouse=True)
def _clean_scope_state() -> Iterator[None]:
    """GIVEN no scope settings left over from another test."""
    reset_scope_state()
    yield
    reset_scope_state()


@pytest.fixture(scope="module")
def keys() -> RSAKeyPair:
    """GIVEN an IdP whose signing key this server trusts. Generated once — it is slow."""
    return RSAKeyPair.generate()


@pytest.fixture
def verifying(keys: RSAKeyPair, monkeypatch: pytest.MonkeyPatch) -> None:
    """GIVEN a server that can verify that IdP's tokens.

    The verifier is handed the public key directly rather than a JWKS URI, so the
    test exercises real RS256 verification without reaching the network.
    """
    verifier = JWTVerifier(public_key=keys.public_key, issuer=ISSUER, audience=AUDIENCE)
    monkeypatch.setattr("datarobot_genai.drmcpbase.oauth_scopes._verifier", lambda: verifier)
    configure_scopes(enforcing())


@pytest.fixture
def mcp() -> FastMCP:
    """GIVEN a server with a code-guarded tool, a tag-only tool, and an open one."""
    server: FastMCP = FastMCP(name="test")

    @server.tool(tags={"database"}, auth=require_scopes(EXECUTE))
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


class TestSatisfies:
    """The matching rule: subset, not equality."""

    def test_every_required_scope_must_be_present(self) -> None:
        assert not satisfies(frozenset({EXECUTE, DB_WRITE}), frozenset({EXECUTE}))

    def test_exact_cover_passes(self) -> None:
        assert satisfies(frozenset({EXECUTE, DB_WRITE}), frozenset({EXECUTE, DB_WRITE}))

    def test_extra_scopes_are_ignored(self) -> None:
        """GIVEN a token also carrying scopes meant for other components."""
        held = frozenset({EXECUTE, DB_WRITE, READ, "openid"})

        assert satisfies(frozenset({EXECUTE, DB_WRITE}), held)

    def test_requiring_nothing_passes_on_any_token(self) -> None:
        assert satisfies(frozenset(), frozenset())

    def test_no_token_passes_when_the_server_cannot_verify_one(self) -> None:
        """GIVEN no way to judge a token, THEN the check is not the thing that blocks."""
        configure_scopes(ScopeSettings())

        assert satisfies(frozenset({EXECUTE}), None)

    def test_no_token_is_denied_once_the_server_can_verify_one(self) -> None:
        configure_scopes(enforcing())

        assert not satisfies(frozenset({EXECUTE}), None)

    def test_an_unguarded_component_is_admitted_either_way(self) -> None:
        """Scope matching alone never blocks a component that requires nothing.

        Whether such a component is *served* is a separate question, answered by
        the access floor — see :class:`TestTokenFloor`.
        """
        configure_scopes(enforcing())

        assert satisfies(frozenset(), None)

    def test_a_verified_token_short_of_a_scope_is_denied_either_way(self) -> None:
        """Enforcement governs the no-token case only; a real token is always judged."""
        configure_scopes(ScopeSettings())

        assert not satisfies(frozenset({EXECUTE}), frozenset({READ}))


class TestPresentedTokens:
    """Which headers a token is read off, and in what order they are tried."""

    def test_a_request_with_no_headers_presents_nothing(self) -> None:
        """GIVEN stdio, a background task or a test — anywhere without a request."""
        assert _presented_tokens() == []

    @pytest.mark.parametrize(
        "headers,expected",
        [
            pytest.param({}, [], id="no headers"),
            pytest.param({"authorization": f"Bearer {API_KEY}"}, [API_KEY], id="bearer"),
            pytest.param({"authorization": f"bearer {API_KEY}"}, [API_KEY], id="lowercase scheme"),
            pytest.param({"authorization": API_KEY}, [API_KEY], id="no scheme"),
            pytest.param({"authorization": "   "}, [], id="blank"),
            pytest.param({"authorization": "Bearer  "}, [], id="scheme with no token"),
            pytest.param({DR_HEADER: API_KEY}, [API_KEY], id="datarobot header alone"),
            pytest.param(
                {"authorization": "Bearer direct", DR_HEADER: "forwarded"},
                ["forwarded", "direct"],
                id="the forwarded user token is tried first",
            ),
            pytest.param(
                {GATEWAY_HEADER: "relayed", "authorization": "Bearer direct"},
                ["direct"],
                id="the global MCP header is not read",
            ),
            pytest.param(
                {"authorization": "Bearer same", DR_HEADER: "same"},
                ["same"],
                id="the same token twice is verified once",
            ),
            pytest.param(
                {DR_HEADER: "   ", "authorization": "Bearer only"},
                ["only"],
                id="a blank first header does not mask the next",
            ),
        ],
    )
    def test_every_candidate_header_is_returned(
        self, monkeypatch: pytest.MonkeyPatch, headers: dict[str, str], expected: list[str]
    ) -> None:
        _presenting(monkeypatch, headers)

        assert _presented_tokens() == expected


class TestRequestScopes:
    """Reading scopes off a live request, with no auth provider in play."""

    async def test_a_verified_token_yields_its_scopes(
        self, mcp: FastMCP, keys: RSAKeyPair, verifying: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _presenting(monkeypatch, {"authorization": f"Bearer {_token(keys, [EXECUTE, READ])}"})

        assert await request_scopes(await _context(mcp)) == frozenset({EXECUTE, READ})

    async def test_an_api_key_does_not_mask_a_token_in_the_datarobot_header(
        self, mcp: FastMCP, keys: RSAKeyPair, verifying: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN the deployment shape: the platform's own token, plus a forwarded one.

        Stopping at the first header carrying a value would read the opaque API
        key, fail to verify it, and report the caller as holding no scopes.
        """
        _presenting(
            monkeypatch,
            {"authorization": f"Bearer {API_KEY}", DR_HEADER: _token(keys, [EXECUTE])},
        )

        assert await request_scopes(await _context(mcp)) == frozenset({EXECUTE})

    async def test_the_forwarded_user_token_wins_when_both_headers_verify(
        self, mcp: FastMCP, keys: RSAKeyPair, verifying: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It names the principal the call is made on behalf of, so it is the answer."""
        _presenting(
            monkeypatch,
            {
                "authorization": f"Bearer {_token(keys, [EXECUTE])}",
                DR_HEADER: _token(keys, [READ]),
            },
        )

        assert await request_scopes(await _context(mcp)) == frozenset({READ})

    async def test_the_global_mcp_header_is_not_a_source_of_scopes(
        self, mcp: FastMCP, keys: RSAKeyPair, verifying: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN a perfectly good token in a header this server does not read."""
        _presenting(monkeypatch, {GATEWAY_HEADER: f"Bearer {_token(keys, [EXECUTE])}"})

        assert await request_scopes(await _context(mcp)) is None

    @pytest.mark.parametrize(
        "headers",
        [
            pytest.param({}, id="no token at all"),
            pytest.param({"authorization": f"Bearer {API_KEY}"}, id="a DataRobot API key"),
            pytest.param({"authorization": "Bearer not-a-jwt"}, id="garbage"),
            pytest.param(
                {"authorization": f"Bearer {API_KEY}", DR_HEADER: "sk-live-abc123"},
                id="two API keys",
            ),
        ],
    )
    async def test_nothing_verifiable_reads_as_no_judgement(
        self,
        mcp: FastMCP,
        verifying: None,
        monkeypatch: pytest.MonkeyPatch,
        headers: dict[str, str],
    ) -> None:
        """None, not an empty set: the server established nothing about this caller."""
        _presenting(monkeypatch, headers)

        assert await request_scopes(await _context(mcp)) is None

    @pytest.mark.parametrize(
        "claims",
        [
            pytest.param({"audience": "https://elsewhere"}, id="minted for another audience"),
            pytest.param({"issuer": "https://other.idp"}, id="from another issuer"),
            pytest.param({"expires_in_seconds": -60}, id="expired"),
        ],
    )
    async def test_a_token_that_is_not_ours_reads_as_no_judgement(
        self,
        mcp: FastMCP,
        keys: RSAKeyPair,
        verifying: None,
        monkeypatch: pytest.MonkeyPatch,
        claims: dict[str, Any],
    ) -> None:
        """A real signed token, refused on its claims rather than its signature."""
        _presenting(monkeypatch, {"authorization": f"Bearer {_token(keys, [EXECUTE], **claims)}"})

        assert await request_scopes(await _context(mcp)) is None

    async def test_a_foreign_signing_key_reads_as_no_judgement(
        self, mcp: FastMCP, verifying: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Right issuer, right audience, right scopes — signed by an IdP we do not trust."""
        impostor = RSAKeyPair.generate()
        _presenting(monkeypatch, {"authorization": f"Bearer {_token(impostor, [EXECUTE])}"})

        assert await request_scopes(await _context(mcp)) is None

    async def test_an_auth_provider_token_is_read_before_any_header(
        self, mcp: FastMCP, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN a configured auth provider, THEN the headers are not consulted."""
        _presenting(monkeypatch, {"authorization": f"Bearer {API_KEY}"})
        tool = next(t for t in await mcp._list_tools() if t.name == "run_sql")
        ctx = AuthContext(token=SimpleNamespace(scopes=[DB_WRITE]), component=tool)  # type: ignore[arg-type]

        assert await request_scopes(ctx) == frozenset({DB_WRITE})


class TestNormalizeTag:
    @pytest.mark.parametrize(
        "written,expected",
        [("database", "DATABASE"), ("read-only", "READ_ONLY"), ("  Ops  ", "OPS")],
    )
    def test_folds_case_and_dashes(self, written: str, expected: str) -> None:
        assert normalize_tag(written) == expected

    def test_dash_and_underscore_spellings_are_the_same_tag(self) -> None:
        assert normalize_tag("read-only") == normalize_tag("read_only")


class TestScopeSource:
    def test_defaults_to_both(self) -> None:
        """Each mechanism applies where declared, so neither has to be switched on."""
        assert ScopeSource.parse(None) is ScopeSource.BOTH
        assert ScopeSource.parse("") is ScopeSource.BOTH
        assert ScopeSettings().source is ScopeSource.BOTH

    @pytest.mark.parametrize(
        "value,expected",
        [("code", ScopeSource.CODE), ("TAGS", ScopeSource.TAGS), (" both ", ScopeSource.BOTH)],
    )
    def test_accepts_known_values_case_insensitively(
        self, value: str, expected: ScopeSource
    ) -> None:
        assert ScopeSource.parse(value) is expected

    def test_unknown_value_falls_back_rather_than_raising(self) -> None:
        """A typo should not turn into an outage."""
        assert ScopeSource.parse("banana") is ScopeSource.BOTH

    @pytest.mark.parametrize(
        "source,reads_code,reads_tags",
        [
            (ScopeSource.CODE, True, False),
            (ScopeSource.TAGS, False, True),
            (ScopeSource.BOTH, True, True),
        ],
    )
    def test_reads_flags(self, source: ScopeSource, reads_code: bool, reads_tags: bool) -> None:
        assert source.reads_code is reads_code
        assert source.reads_tags is reads_tags


class TestEnforcementIsDerived:
    """Enforcement follows from being able to verify a token, and is not configured."""

    def test_a_server_with_no_oauth_configured_does_not_enforce(self) -> None:
        assert ScopeSettings().enforced is False

    def test_a_server_that_can_verify_tokens_enforces(self) -> None:
        assert enforcing().enforced is True

    @pytest.mark.parametrize("missing", ["issuer", "audience"])
    def test_half_configured_oauth_does_not_enforce(self, missing: str) -> None:
        """GIVEN one of the three unset, THEN no token can be judged, so none is denied."""
        values = {"issuer": ISSUER, "audience": AUDIENCE}
        values[missing] = ""

        assert ScopeSettings(**values).enforced is False

    def test_an_explicit_jwks_uri_is_enough_without_the_okta_default(self) -> None:
        settings = ScopeSettings(issuer=ISSUER, audience=AUDIENCE, jwks_uri="https://k/jwks.json")

        assert settings.verification_target() == ("https://k/jwks.json", ISSUER, AUDIENCE)

    def test_the_verification_target_is_what_enforcement_reads(self) -> None:
        """One predicate, so verifying and enforcing can never disagree."""
        assert ScopeSettings().verification_target() is None
        assert enforcing().verification_target() is not None


class TestTokenFloor:
    """Who gets served anything at all, decided before which components they reach."""

    @pytest.mark.parametrize(
        "headers",
        [
            pytest.param({}, id="no token at all"),
            pytest.param({"authorization": f"Bearer {API_KEY}"}, id="a DataRobot API key"),
            pytest.param({"authorization": "Bearer not-a-jwt"}, id="garbage"),
        ],
    )
    async def test_an_unidentified_caller_is_served_nothing(
        self,
        mcp: FastMCP,
        verifying: None,
        monkeypatch: pytest.MonkeyPatch,
        headers: dict[str, str],
    ) -> None:
        """Not a half-open server: the unguarded tools go too."""
        _presenting(monkeypatch, headers)
        await wire_scopes(mcp, enforcing(source="code"))

        assert await _visible(mcp) == set()

    async def test_a_token_for_another_audience_is_not_an_identity_here(
        self, mcp: FastMCP, keys: RSAKeyPair, verifying: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN a real token from our own IdP, minted for somebody else."""
        _presenting(
            monkeypatch,
            {DR_HEADER: _token(keys, [EXECUTE], audience="https://elsewhere")},
        )
        await wire_scopes(mcp, enforcing(source="code"))

        assert await _visible(mcp) == set()

    async def test_a_verified_token_carrying_no_scopes_still_reaches_the_open_tools(
        self, mcp: FastMCP, keys: RSAKeyPair, verifying: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The floor asks who you are; the scope rules ask what you may reach."""
        _presenting(monkeypatch, {DR_HEADER: _token(keys, [])})
        await wire_scopes(mcp, enforcing(source="code"))

        assert await _visible(mcp) == {"harmless", "list_tables"}

    async def test_a_server_that_cannot_verify_tokens_serves_everyone(
        self, mcp: FastMCP, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN an API-key deployment, THEN nothing here takes access away."""
        _presenting(monkeypatch, {"authorization": f"Bearer {API_KEY}"})
        await wire_scopes(mcp, ScopeSettings(source="code"))

        assert await _visible(mcp) == {"harmless", "list_tables", "run_sql"}

    async def test_the_floor_is_attached_to_every_component(self, mcp: FastMCP) -> None:
        configure_scopes(enforcing())

        assert await apply_token_floor(mcp) == 3

    async def test_no_floor_without_a_way_to_verify(self, mcp: FastMCP) -> None:
        configure_scopes(ScopeSettings())

        assert await apply_token_floor(mcp) == 0

    async def test_rewiring_replaces_the_floor_rather_than_stacking_it(self, mcp: FastMCP) -> None:
        await wire_scopes(mcp, enforcing(source="code"))
        await wire_scopes(mcp, enforcing(source="code"))

        floors = [
            check
            for check in await _checks_on(mcp, "harmless")
            if getattr(check, TOKEN_FLOOR_ATTR, False)
        ]
        assert len(floors) == 1

    async def test_dropping_oauth_removes_the_floor_on_a_rewire(self, mcp: FastMCP) -> None:
        """GIVEN a server rewired without verification, THEN the floor is not left behind."""
        await wire_scopes(mcp, enforcing(source="code"))
        await wire_scopes(mcp, ScopeSettings(source="code"))

        assert not any(
            getattr(check, TOKEN_FLOOR_ATTR, False) for check in await _checks_on(mcp, "harmless")
        )


class TestCodeDeclaredScopes:
    """Scopes declared on the component itself."""

    async def test_scopes_are_readable_back_off_the_component(self, mcp: FastMCP) -> None:
        """The whole point: FastMCP's own require_scopes hides them in a closure."""
        configure_scopes(ScopeSettings(source="code"))

        assert await collect_code_declared_scopes(mcp) == {EXECUTE}

    async def test_they_reach_the_published_list(self, mcp: FastMCP) -> None:
        await wire_scopes(mcp, ScopeSettings(source="code"))

        assert derived_scopes() == [EXECUTE]

    async def test_a_caller_without_the_scope_loses_the_component(
        self, mcp: FastMCP, keys: RSAKeyPair, verifying: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN an identified caller short of the scope, THEN only that tool is hidden."""
        _presenting(monkeypatch, {DR_HEADER: _token(keys, [READ])})
        await wire_scopes(mcp, enforcing(source="code"))

        assert await _visible(mcp) == {"harmless", "list_tables"}


class TestTagScopes:
    """Scopes required through a tag the component already declares."""

    async def test_guards_every_component_carrying_the_tag(
        self, mcp: FastMCP, keys: RSAKeyPair, verifying: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _presenting(monkeypatch, {DR_HEADER: _token(keys, [READ])})
        await wire_scopes(mcp, enforcing(source="tags", tag_scopes={"DATABASE": [DB_WRITE]}))

        assert await _visible(mcp) == {"harmless"}

    async def test_untagged_components_are_unaffected(
        self, mcp: FastMCP, keys: RSAKeyPair, verifying: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _presenting(monkeypatch, {DR_HEADER: _token(keys, [READ])})
        await wire_scopes(mcp, enforcing(source="tags", tag_scopes={"NOBODY": [DB_WRITE]}))

        assert await _visible(mcp) == {"harmless", "list_tables", "run_sql"}

    async def test_env_var_spelling_matches_the_component_tag(
        self, mcp: FastMCP, keys: RSAKeyPair, verifying: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN MCP_OAUTH_TAG_SCOPES_DATABASE, THEN a tool tagged `database` is guarded."""
        _presenting(monkeypatch, {DR_HEADER: _token(keys, [READ])})
        await wire_scopes(mcp, enforcing(source="tags", tag_scopes={"DaTaBaSe": [DB_WRITE]}))

        assert "list_tables" not in await _visible(mcp)

    async def test_code_declarations_are_inert_under_tags(
        self, mcp: FastMCP, keys: RSAKeyPair, verifying: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN source=tags, THEN a code declaration neither guards nor publishes."""
        _presenting(monkeypatch, {DR_HEADER: _token(keys, [READ])})
        await wire_scopes(mcp, enforcing(source="tags"))

        assert derived_scopes() == []
        assert "run_sql" in await _visible(mcp)

    async def test_tag_scopes_are_inert_under_code(self, mcp: FastMCP) -> None:
        configure_scopes(ScopeSettings(source=ScopeSource.CODE, tag_scopes={"DATABASE": [READ]}))

        assert await apply_tag_scopes(mcp) == 0

    async def test_tags_apply_under_the_default_source(self, mcp: FastMCP) -> None:
        """GIVEN no scope source configured, THEN a tag mapping still guards."""
        await wire_scopes(mcp, ScopeSettings(tag_scopes={"DATABASE": [DB_WRITE]}))

        assert derived_scopes() == sorted([DB_WRITE, EXECUTE])

    async def test_rewiring_does_not_stack_duplicate_checks(self, mcp: FastMCP) -> None:
        """GIVEN wiring runs twice, THEN the tag rule is replaced, not doubled."""
        settings = ScopeSettings(source="tags", tag_scopes={"DATABASE": [DB_WRITE]})
        await wire_scopes(mcp, settings)
        after_first = len(await _checks_on(mcp, "list_tables"))

        await wire_scopes(mcp, settings)

        assert after_first == 1
        assert len(await _checks_on(mcp, "list_tables")) == 1

    async def test_switching_away_from_tags_removes_the_tag_check(self, mcp: FastMCP) -> None:
        await wire_scopes(mcp, ScopeSettings(source="tags", tag_scopes={"DATABASE": [DB_WRITE]}))

        await wire_scopes(mcp, ScopeSettings(source="code"))

        assert await _checks_on(mcp, "list_tables") == []

    async def test_a_code_declaration_survives_tag_rewiring(self, mcp: FastMCP) -> None:
        """Checks declared in code are not ours to remove."""
        await wire_scopes(mcp, ScopeSettings(source="tags", tag_scopes={"DATABASE": [DB_WRITE]}))

        checks = await _checks_on(mcp, "run_sql")

        declared = {frozenset(getattr(c, DECLARED_SCOPES_ATTR, frozenset())) for c in checks}
        assert frozenset({EXECUTE}) in declared


class TestBothSources:
    """`both` applies each mechanism wherever it is declared."""

    async def test_published_list_is_the_union(self, mcp: FastMCP) -> None:
        await wire_scopes(mcp, ScopeSettings(source="both", tag_scopes={"DATABASE": [DB_WRITE]}))

        assert derived_scopes() == sorted([DB_WRITE, EXECUTE])

    async def test_a_component_in_both_places_requires_the_union(self, mcp: FastMCP) -> None:
        """GIVEN run_sql declares one scope in code and its tag maps another."""
        await wire_scopes(mcp, ScopeSettings(source="both", tag_scopes={"DATABASE": [DB_WRITE]}))

        required = {
            scope
            for check in await _checks_on(mcp, "run_sql")
            for scope in getattr(check, DECLARED_SCOPES_ATTR, frozenset())
        }

        assert required == {EXECUTE, DB_WRITE}

    async def test_a_component_in_one_place_is_guarded_by_that_one(self, mcp: FastMCP) -> None:
        await wire_scopes(mcp, ScopeSettings(source="both", tag_scopes={"DATABASE": [DB_WRITE]}))

        required = {
            scope
            for check in await _checks_on(mcp, "list_tables")
            for scope in getattr(check, DECLARED_SCOPES_ATTR, frozenset())
        }

        assert required == {DB_WRITE}


class TestDerivedScopes:
    """Advertising follows enforcement."""

    async def test_nothing_is_advertised_when_nothing_is_enforced(self, mcp: FastMCP) -> None:
        await wire_scopes(mcp, ScopeSettings(source="tags"))

        assert derived_scopes() == []

    async def test_sorted_and_deduplicated(self, mcp: FastMCP) -> None:
        await wire_scopes(
            mcp,
            ScopeSettings(source="both", tag_scopes={"DATABASE": [EXECUTE, READ], "X": [READ]}),
        )

        assert derived_scopes() == sorted({EXECUTE, READ})


class TestStartupValidation:
    """A requirement that quietly does nothing is the failure worth catching."""

    @pytest.fixture(autouse=True)
    def _warnings(self, caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
        caplog.set_level(logging.WARNING, logger=SCOPES_LOGGER)
        return caplog

    async def test_a_tag_matching_no_component_is_reported(
        self, mcp: FastMCP, caplog: pytest.LogCaptureFixture
    ) -> None:
        """GIVEN MCP_OAUTH_TAG_SCOPES_DATBASE, THEN the typo is not silent."""
        await wire_scopes(mcp, enforcing(tag_scopes={"DATBASE": [DB_WRITE]}))

        assert "DATBASE" in caplog.text
        assert "guard nothing" in caplog.text

    async def test_the_tags_actually_in_use_are_reported_alongside(
        self, mcp: FastMCP, caplog: pytest.LogCaptureFixture
    ) -> None:
        await wire_scopes(mcp, enforcing(tag_scopes={"DATBASE": [DB_WRITE]}))

        assert "DATABASE" in caplog.text

    async def test_a_tag_that_matches_is_not_reported(
        self, mcp: FastMCP, caplog: pytest.LogCaptureFixture
    ) -> None:
        await wire_scopes(mcp, enforcing(tag_scopes={"DATABASE": [DB_WRITE]}))

        assert "guard nothing" not in caplog.text

    async def test_unenforceable_requirements_are_reported(
        self, mcp: FastMCP, caplog: pytest.LogCaptureFixture
    ) -> None:
        """GIVEN scopes declared but no way to verify a token, THEN say so."""
        await wire_scopes(mcp, ScopeSettings(tag_scopes={"DATABASE": [DB_WRITE]}))

        assert "every caller passes every scope check" in caplog.text

    async def test_enforceable_requirements_are_not_reported(
        self, mcp: FastMCP, caplog: pytest.LogCaptureFixture
    ) -> None:
        await wire_scopes(mcp, enforcing(tag_scopes={"DATABASE": [DB_WRITE]}))

        assert "every caller passes every scope check" not in caplog.text

    async def test_a_server_declaring_no_scopes_is_not_warned_at(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The unconfigured server is the common case and has nothing to report."""
        bare: FastMCP = FastMCP(name="bare")

        @bare.tool
        def harmless() -> str:
            """Guarded by nothing."""
            return "ok"

        await wire_scopes(bare, ScopeSettings())

        assert caplog.text == ""

    async def test_requirements_shadowed_by_the_scope_source_are_reported(
        self, mcp: FastMCP, caplog: pytest.LogCaptureFixture
    ) -> None:
        """GIVEN source=code, THEN tag requirements that will never apply are called out."""
        await wire_scopes(mcp, enforcing(source="code", tag_scopes={"DATABASE": [DB_WRITE]}))

        assert "inert" in caplog.text

    async def test_code_declarations_shadowed_by_the_scope_source_are_reported(
        self, mcp: FastMCP, caplog: pytest.LogCaptureFixture
    ) -> None:
        await wire_scopes(mcp, enforcing(source="tags", tag_scopes={"DATABASE": [DB_WRITE]}))

        assert EXECUTE in caplog.text
        assert "inert" in caplog.text


class _CountingVerifier:
    """Returns a fixed answer and counts how often it was asked."""

    def __init__(self, scopes: list[str] | None) -> None:
        self.scopes = scopes
        self.calls = 0

    async def verify_token(self, token: str) -> AccessToken | None:
        self.calls += 1
        if self.scopes is None:
            return None
        return AccessToken(token=token, client_id="caller", scopes=self.scopes, expires_at=None)


def _judging(monkeypatch: pytest.MonkeyPatch, scopes: list[str] | None) -> _CountingVerifier:
    """GIVEN an enforcing server whose verifier gives every token this verdict."""
    verifier = _CountingVerifier(scopes)
    configure_scopes(enforcing())
    monkeypatch.setattr("datarobot_genai.drmcpbase.oauth_scopes._verifier", lambda: verifier)
    return verifier


def _on_request(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """GIVEN one HTTP request for every check to run against."""
    request = SimpleNamespace(state=SimpleNamespace())
    monkeypatch.setattr("datarobot_genai.drmcpbase.oauth_scopes.get_http_request", lambda: request)
    return request


class TestRequestMemo:
    """The token is verified once per request, whichever component's check asks first."""

    async def test_one_verification_serves_every_check_on_a_request(
        self, mcp: FastMCP, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        verifier = _judging(monkeypatch, scopes=[READ])
        _presenting(monkeypatch, {DR_HEADER: "a-token"})
        _on_request(monkeypatch)
        ctx = await _context(mcp)

        first = await request_scopes(ctx)
        second = await request_scopes(ctx)

        assert first == second == frozenset({READ})
        assert verifier.calls == 1

    async def test_a_new_request_is_judged_afresh(
        self, mcp: FastMCP, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        verifier = _judging(monkeypatch, scopes=[READ])
        _presenting(monkeypatch, {DR_HEADER: "a-token"})
        ctx = await _context(mcp)

        _on_request(monkeypatch)
        await request_scopes(ctx)
        _on_request(monkeypatch)
        await request_scopes(ctx)

        assert verifier.calls == 2

    async def test_no_judgement_is_memoised_too(
        self, mcp: FastMCP, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unverifiable token is not re-verified for every component it is denied."""
        verifier = _judging(monkeypatch, scopes=None)
        _presenting(monkeypatch, {"authorization": f"Bearer {API_KEY}"})
        _on_request(monkeypatch)
        ctx = await _context(mcp)

        assert await request_scopes(ctx) is None
        assert await request_scopes(ctx) is None
        assert verifier.calls == 1

    async def test_outside_an_http_request_nothing_is_memoised(
        self, mcp: FastMCP, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No request object, no cache lifetime to borrow — judge every time."""
        verifier = _judging(monkeypatch, scopes=[READ])
        _presenting(monkeypatch, {DR_HEADER: "a-token"})
        ctx = await _context(mcp)

        await request_scopes(ctx)
        await request_scopes(ctx)

        assert verifier.calls == 2


class TestUnverifiableTokenWarning:
    """Said out loud once, then at DEBUG: expected traffic must not flood the log."""

    async def test_the_first_unverifiable_token_warns(
        self, mcp: FastMCP, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _judging(monkeypatch, scopes=None)
        _presenting(monkeypatch, {"authorization": f"Bearer {API_KEY}"})
        ctx = await _context(mcp)

        with caplog.at_level(logging.WARNING, logger=SCOPES_LOGGER):
            await request_scopes(ctx)
            await request_scopes(ctx)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "served nothing" in warnings[0].getMessage()

    async def test_reconfiguring_arms_the_warning_again(
        self, mcp: FastMCP, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """New settings, possibly a new cause — one report per configuration."""
        _judging(monkeypatch, scopes=None)
        _presenting(monkeypatch, {"authorization": f"Bearer {API_KEY}"})
        ctx = await _context(mcp)

        with caplog.at_level(logging.WARNING, logger=SCOPES_LOGGER):
            await request_scopes(ctx)
            configure_scopes(enforcing())
            await request_scopes(ctx)

        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2

    async def test_a_verified_token_does_not_warn(
        self, mcp: FastMCP, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _judging(monkeypatch, scopes=[READ])
        _presenting(monkeypatch, {DR_HEADER: "a-token"})
        ctx = await _context(mcp)

        with caplog.at_level(logging.WARNING, logger=SCOPES_LOGGER):
            await request_scopes(ctx)

        assert caplog.text == ""


class _FakeJWKSResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> Any:
        return self._payload


class _FakeHTTPXModule:
    """Stands in for the httpx module: one client, one canned answer."""

    def __init__(self, payload: Any = None, error: Exception | None = None) -> None:
        self._payload = payload
        self._error = error
        self.fetched: list[str] = []

    # The probe calls httpx.AsyncClient(timeout=...) as a context manager.
    def AsyncClient(self, **_: Any) -> "_FakeHTTPXModule":  # noqa: N802 - mimics httpx
        return self

    async def __aenter__(self) -> "_FakeHTTPXModule":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def get(self, url: str) -> _FakeJWKSResponse:
        self.fetched.append(url)
        if self._error is not None:
            raise self._error
        return _FakeJWKSResponse(self._payload)


def _jwks_serving(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> _FakeHTTPXModule:
    fake = _FakeHTTPXModule(**kwargs)
    monkeypatch.setattr("datarobot_genai.drmcpbase.oauth_scopes.httpx", fake)
    return fake


class TestProbeVerificationKeys:
    """An unreachable IdP is a startup log line, not a silently empty server."""

    async def test_a_reachable_jwks_is_reported_with_its_key_count(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        fake = _jwks_serving(monkeypatch, payload={"keys": [{"kid": "a"}, {"kid": "b"}]})
        configure_scopes(enforcing())

        with caplog.at_level(logging.INFO, logger=SCOPES_LOGGER):
            await probe_verification_keys()

        assert fake.fetched == [f"{ISSUER}/v1/keys"]
        assert "reachable (2 signing key(s))" in caplog.text

    async def test_an_unreachable_jwks_warns_and_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An outage is not a configuration to refuse to boot on."""
        _jwks_serving(monkeypatch, error=ConnectionError("no route to host"))
        configure_scopes(enforcing())

        with caplog.at_level(logging.WARNING, logger=SCOPES_LOGGER):
            await probe_verification_keys()

        assert "could not be fetched" in caplog.text
        assert "served nothing" in caplog.text

    async def test_a_document_with_no_keys_is_called_out(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """GIVEN a URI that answers but is not a JWKS — a misconfiguration, not an outage."""
        _jwks_serving(monkeypatch, payload={"detail": "not found"})
        configure_scopes(enforcing())

        with caplog.at_level(logging.WARNING, logger=SCOPES_LOGGER):
            await probe_verification_keys()

        assert "is it really a JWKS URI?" in caplog.text

    async def test_a_server_that_cannot_verify_does_not_probe(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        fake = _jwks_serving(monkeypatch, payload={"keys": []})
        configure_scopes(ScopeSettings())

        with caplog.at_level(logging.INFO, logger=SCOPES_LOGGER):
            await probe_verification_keys()

        assert fake.fetched == []
        assert caplog.text == ""


class TestPartialConfiguration:
    """Half a verification config is the state worth shouting about."""

    async def test_a_missing_audience_is_named(
        self, mcp: FastMCP, caplog: pytest.LogCaptureFixture
    ) -> None:
        """GIVEN an issuer and no audience — whoever set it believed OAuth was on."""
        with caplog.at_level(logging.WARNING, logger=SCOPES_LOGGER):
            await wire_scopes(mcp, ScopeSettings(issuer=ISSUER))

        assert "partially configured" in caplog.text
        assert "no audience" in caplog.text
        assert "every caller passes every scope check" in caplog.text

    async def test_a_missing_issuer_is_named(
        self, mcp: FastMCP, caplog: pytest.LogCaptureFixture
    ) -> None:
        """GIVEN an audience alone: no issuer, and no JWKS URI to resolve from it."""
        with caplog.at_level(logging.WARNING, logger=SCOPES_LOGGER):
            await wire_scopes(mcp, ScopeSettings(audience=AUDIENCE))

        assert "partially configured" in caplog.text
        assert "no issuer" in caplog.text
        assert "no JWKS URI" in caplog.text

    async def test_declared_requirements_are_mentioned_when_present(
        self, mcp: FastMCP, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=SCOPES_LOGGER):
            await wire_scopes(
                mcp, ScopeSettings(issuer=ISSUER, tag_scopes={"DATABASE": [DB_WRITE]})
            )

        assert "published but not enforced" in caplog.text

    async def test_a_full_configuration_is_not_partial(
        self, mcp: FastMCP, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=SCOPES_LOGGER):
            await wire_scopes(mcp, enforcing())

        assert "partially configured" not in caplog.text

    async def test_no_configuration_at_all_is_not_partial(
        self, mcp: FastMCP, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The API-key deployment configured nothing and has nothing to be warned about."""
        with caplog.at_level(logging.WARNING, logger=SCOPES_LOGGER):
            await wire_scopes(mcp, ScopeSettings())

        assert "partially configured" not in caplog.text


class TestScopeSettings:
    def test_tag_keys_are_normalized_on_construction(self) -> None:
        settings = ScopeSettings(tag_scopes={"read-only": [READ]})

        assert settings.tag_scopes == {"READ_ONLY": [READ]}

    def test_jwks_uri_defaults_to_the_okta_layout(self) -> None:
        settings = ScopeSettings(issuer="https://issuer.example.com/oauth2/aus1/")

        assert settings.resolved_jwks_uri() == "https://issuer.example.com/oauth2/aus1/v1/keys"

    def test_an_explicit_jwks_uri_wins(self) -> None:
        settings = ScopeSettings(issuer="https://issuer.example.com", jwks_uri="https://k/keys")

        assert settings.resolved_jwks_uri() == "https://k/keys"

    def test_no_issuer_means_no_jwks_uri(self) -> None:
        assert ScopeSettings().resolved_jwks_uri() is None
