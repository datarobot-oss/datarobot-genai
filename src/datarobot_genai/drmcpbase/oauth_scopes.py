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

"""Per-component OAuth scope requirements.

The DataRobot gateway does the *authentication*; this module does the
*authorization*, per component, on every request. ``scopes_supported`` in the
published RFC 9728 document cannot do that job on its own: it is a flat list
with no component in it, so it can say "this server understands
``mcp:tools:write``" but never "``run_sql`` requires ``mcp:tools:write``".
Something has to bind component to scopes, and there are two places to write
that binding:

in code
    ``@dr_mcp_tool(auth=require_scopes("mcp:tools:write"))`` — the requirement
    travels with the component it guards and survives a tag rename.

in configuration
    a scope requirement keyed on a tag the component already declares, so one
    setting guards every component carrying it and can differ per environment
    without a code change.

The scope source selects which is live and defaults to ``both``, so each
mechanism simply applies wherever it is declared; set ``code`` or ``tags`` only
to silence the other one. It governs enforcement *and* the published
``scopes_supported`` together, so the server can never advertise a scope it is
not enforcing — that list is always derived from the rules, never hand-written.

Matching is a subset test, never equality. Every required scope must be present;
anything else the token carries is not examined here, because one token serves a
whole session and normally holds the scopes for every component the client might
call. Requirements also stack: each declaration adds its own check and all of
them must pass, so a component matching two mapped tags — or carrying a code
declaration *and* a tag under ``both`` — requires the union. Nothing is any-of.

Whether a caller presenting no usable OAuth token is blocked is **derived, not
configured**: it is blocked exactly when this server is able to verify tokens in
the first place. There is no enforcement mode to set and therefore no way to
declare a requirement that silently does not apply. See
:attr:`ScopeSettings.enforced`.

Such a caller is served *nothing* — not the guarded components only. Once tokens
can be verified, :func:`require_verified_token` sits beneath the scope rules on
every component, so the question "who is this?" is answered before "what may they
reach?". A caller holding a verifiable token then sees exactly the components
whose scopes it covers, which for a component declaring none is all of them. The
practical consequence is worth stating plainly: configuring OAuth closes the
server to DataRobot API-key callers, because an opaque API key carries no
identity this server can check. A deployment that must serve both is a deployment
that should not configure OAuth verification.

This module holds the mechanism and nothing about where settings come from:
:class:`ScopeSettings` is handed in by the server that owns the configuration,
the same way the protected-resource metadata entities take theirs.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from typing import Any
from typing import cast

import httpx
from fastmcp.server.auth import AuthCheck
from fastmcp.server.auth import AuthContext
from fastmcp.server.auth.providers.jwt import JWTVerifier
from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.dependencies import get_http_request
from starlette.requests import Request

logger = logging.getLogger(__name__)


class ScopeSource(StrEnum):
    """Which declaration mechanism the server reads.

    Defaults to :attr:`BOTH`, so each mechanism simply applies wherever it is
    declared and neither has to be switched on. ``CODE`` and ``TAGS`` exist to
    deliberately silence the other one.
    """

    CODE = "code"
    TAGS = "tags"
    BOTH = "both"

    @classmethod
    def parse(cls, value: object) -> ScopeSource:
        """Return the matching member, falling back to the default.

        An unrecognised value logs and falls back rather than failing the
        server: refusing to start would turn a typo into an outage.
        """
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower()
        if not text:
            return cls.BOTH
        try:
            return cls(text)
        except ValueError:
            logger.warning(
                "Scope source %r is not one of %s; falling back to %r.",
                text,
                ", ".join(member.value for member in cls),
                cls.BOTH.value,
            )
            return cls.BOTH

    @property
    def reads_code(self) -> bool:
        """Whether in-code ``require_scopes(...)`` declarations are read."""
        return self in {ScopeSource.CODE, ScopeSource.BOTH}

    @property
    def reads_tags(self) -> bool:
        """Whether the tag-keyed requirements are read."""
        return self in {ScopeSource.TAGS, ScopeSource.BOTH}


#: Attribute :func:`require_scopes` records its scope names on, so they can be
#: read back and published. FastMCP's own ``require_scopes`` keeps them in a
#: closure where nothing can reach them.
DECLARED_SCOPES_ATTR = "dr_declared_scopes"

#: Marks a check that :func:`apply_tag_scopes` attached, so re-wiring can strip
#: its own previous work instead of stacking a second copy of every rule.
TAG_APPLIED_ATTR = "dr_tag_scope_check"

#: Marks the check :func:`apply_token_floor` attached, for the same reason.
TOKEN_FLOOR_ATTR = "dr_token_floor_check"

#: Headers an OAuth bearer token may arrive in, tried in this order. Both can be
#: populated at once, and the first one carrying *a* value is often not the one
#: carrying an OAuth token — see :func:`_presented_tokens`.
#:
#: ``x-datarobot-external-access-token`` is what the gateway forwards an end
#: user's IdP token in, and the only header that exclusively carries one. It names
#: the principal the call is made on behalf of, so it wins where both verify.
#: ``authorization`` is kept for callers that reach the container directly — local
#: runs, tests, a deployment that is not behind the gateway — and not because the
#: gateway forwards it, which it does not.
#:
#: Deliberately absent: ``x-datarobot-authorization``, which belongs to the global
#: MCP path rather than this server's, and the API-key headers
#: ``x-datarobot-api-key`` / ``x-datarobot-api-token``, which never carry an IdP
#: token and would only spend a verification per request to reach the same answer.
#: Contrast ``drmcputils.constants.HEADER_TOKEN_CANDIDATE_NAMES``, which is
#: looking for a DataRobot API token and therefore wants exactly those.
OAUTH_TOKEN_HEADERS = (
    "x-datarobot-external-access-token",
    "authorization",
)

BEARER_SCHEME = "bearer"

#: Okta's JWKS layout, used when no JWKS URI is configured explicitly.
DEFAULT_JWKS_PATH = "/v1/keys"


def normalize_tag(tag: str) -> str:
    """Return a tag in the form used to match configuration against components.

    Tags are written one way in code (``read-only``) and another in an
    environment variable name (``READ_ONLY``), so both sides are folded to upper
    case with ``-`` treated as ``_``. A consequence worth knowing: ``read-only``
    and ``read_only`` are the same tag as far as scope configuration goes.
    """
    return tag.strip().upper().replace("-", "_")


@dataclass(frozen=True)
class ScopeSettings:
    """What the scope machinery needs, independent of where it was configured."""

    #: Which declaration mechanism is read. Both, unless narrowed.
    source: ScopeSource = ScopeSource.BOTH
    #: ``{tag: scopes a caller must hold}``, keyed by :func:`normalize_tag`.
    #: All scopes listed for a tag are required, not any one of them.
    tag_scopes: Mapping[str, list[str]] = field(default_factory=dict)
    #: Token issuer, used to verify bearer tokens for scope reading.
    issuer: str | None = None
    #: Expected ``aud`` on those tokens.
    audience: str | None = None
    #: JWKS URI; defaults to ``<issuer>/v1/keys``, which is Okta's layout.
    jwks_uri: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", ScopeSource.parse(self.source))
        object.__setattr__(
            self,
            "tag_scopes",
            {normalize_tag(tag): list(scopes) for tag, scopes in self.tag_scopes.items()},
        )

    @property
    def code_active(self) -> bool:
        """Whether in-code ``require_scopes(...)`` declarations are read."""
        return self.source.reads_code

    @property
    def tags_active(self) -> bool:
        """Whether the tag-keyed requirements are read."""
        return self.source.reads_tags

    def resolved_jwks_uri(self) -> str | None:
        """Return the JWKS URI to verify with, or None when unconfigured."""
        if self.jwks_uri:
            return self.jwks_uri
        if self.issuer:
            return f"{self.issuer.rstrip('/')}{DEFAULT_JWKS_PATH}"
        return None

    def verification_target(self) -> tuple[str, str, str] | None:
        """Return ``(jwks_uri, issuer, audience)``, or None if tokens cannot be verified.

        All three are needed to judge a token, so they are resolved together
        rather than checked separately in each caller — a partially configured
        server would otherwise enforce in one place and not the other.
        """
        jwks_uri = self.resolved_jwks_uri()
        issuer = (self.issuer or "").rstrip("/")
        audience = (self.audience or "").strip()
        if not (jwks_uri and issuer and audience):
            return None
        return jwks_uri, issuer, audience

    @property
    def enforced(self) -> bool:
        """Whether a caller with no usable OAuth token is turned away.

        Derived from the configuration rather than set by hand. A server that
        can verify tokens is one where OAuth is genuinely set up, so a request
        arriving without a verifiable token is a request that has been judged —
        and a declared requirement should mean what it says. Such a caller is
        served no component at all, not merely the unguarded ones: see
        :func:`require_verified_token`.

        Where it cannot verify, the opposite holds: an unverifiable token is
        indistinguishable from an API-key or gateway-authenticated call, so
        denying would take access away from callers over a question this server
        was never configured to answer. Requirements are then advertised but not
        enforced, which :func:`wire_scopes` says out loud at startup.
        """
        return self.verification_target() is not None


class _State:
    """Holder for the settings in force.

    An object rather than a module-level name so installing new settings is an
    attribute write, not a ``global`` rebind. It has to live at module level at
    all because auth checks are built at wiring time but evaluated per request,
    and FastMCP gives a check no route back to the server that registered it.
    """

    settings = ScopeSettings()

    #: Whether the one WARNING about presented-but-unverifiable tokens has been
    #: emitted. Only the first occurrence warns — on a server whose regular
    #: traffic includes API-key callers, every one of their requests lands in
    #: that branch, and a warning per request is a flooded log, not a signal.
    warned_unverifiable = False


_state = _State()

# Scopes declared in code, read back once at startup.
_code_declared_scopes: set[str] = set()

# JWTVerifier caches the IdP's JWKS on the instance, so rebuilding one per
# request would refetch the keys every time. Keyed by the settings that define it.
#
# Key rotation needs no handling here: the verifier refreshes its JWKS hourly
# and refetches immediately when a token presents a ``kid`` it does not know,
# so a token minted right after a rotation verifies on first sight. The cost of
# that behaviour is bounded — there is no negative cache, so a well-formed JWT
# with an unknown ``kid`` spends one JWKS fetch per request, but opaque API
# keys fail at parse before any fetch and the per-request memo caps the chain
# at one per request.
_verifier_cache: dict[tuple[str, str, str], JWTVerifier] = {}


def configure_scopes(settings: ScopeSettings) -> None:
    """Install the settings every scope check reads."""
    _state.settings = settings
    _state.warned_unverifiable = False
    _verifier_cache.clear()


def active_settings() -> ScopeSettings:
    """Return the settings currently in force."""
    return _state.settings


def reset_scope_state() -> None:
    """Forget installed settings, collected scopes and cached verifiers."""
    _state.settings = ScopeSettings()
    _state.warned_unverifiable = False
    _code_declared_scopes.clear()
    _verifier_cache.clear()


def _verifier() -> JWTVerifier | None:
    """Build a verifier for reading scopes off a request, or None if unconfigured."""
    key = _state.settings.verification_target()
    if key is None:
        return None
    if key not in _verifier_cache:
        jwks_uri, issuer, audience = key
        _verifier_cache[key] = JWTVerifier(jwks_uri=jwks_uri, issuer=issuer, audience=audience)
    return _verifier_cache[key]


def _presented_tokens() -> list[str]:
    """Return every bearer token on the current request, in the order to try them.

    All candidate headers are returned, not just the first one carrying a value:
    on a DataRobot deployment ``authorization`` usually holds the platform's own
    API token while the end user's forwarded IdP token sits in
    ``x-datarobot-external-access-token``. Stopping at the first non-empty header
    would hand the verifier an opaque API key and never reach the OAuth token
    behind it, which reads as "this caller has no scopes".

    Duplicates are dropped, so a client that sets both headers to the same value
    does not pay for verifying it twice.
    """
    try:
        headers = get_http_headers(include_all=True)
    except Exception:  # not an HTTP request (stdio, tests, background task)
        return []
    if not headers:
        return []

    found: list[str] = []
    for name in OAUTH_TOKEN_HEADERS:
        value = (headers.get(name) or "").strip()
        # Split on the scheme rather than trimming a ``"bearer "`` prefix, so a
        # header holding the scheme and nothing else presents no token at all
        # instead of the word "Bearer". The DataRobot header carries a bare
        # token, which has no scheme to split off.
        scheme, _, credentials = value.partition(" ")
        if scheme.lower() == BEARER_SCHEME:
            value = credentials.strip()
        if value and value not in found:
            found.append(value)
    return found


#: ``request.state`` attribute the request's scope answer is memoised under.
#: The floor and every scope rule ask the same question, per component — a
#: tool listing over N components would otherwise verify the same token up to
#: 2N times. The request object *is* the cache's lifetime, so nothing needs
#: invalidating: the next request starts empty.
_REQUEST_SCOPES_STATE_ATTR = "dr_request_scopes"

#: Distinguishes "not judged yet" from a memoised ``None`` (nothing to read).
_UNSET = object()


def _current_http_request() -> Request | None:
    """Return the active HTTP request, or None outside one (stdio, tests)."""
    try:
        return get_http_request()
    except Exception:
        return None


async def request_scopes(ctx: AuthContext) -> frozenset[str] | None:
    """Return the scopes carried by this request, or None when there are none to read.

    ``None`` means "no OAuth scopes to judge by" and is deliberately *not* the
    same as an empty set. An empty set is a verified token that happens to carry
    no scopes, and fails every requirement; ``None`` is the absence of a
    judgement, and what follows from it depends on whether this server can
    verify tokens at all — see :attr:`ScopeSettings.enforced`.

    It arises from more than a missing header. A server behind the gateway
    legitimately sees non-OAuth traffic: an API-key deployment has no OAuth in
    play, ``Authorization`` often carries a DataRobot API token rather than an
    IdP one, and a gateway may authenticate without forwarding the header at
    all. A token that fails verification lands here too — it is not an OAuth
    token *for us*, which is the same position as no token, and minting an empty
    scope set for it would report a distinction this server never established.

    Reads ``ctx.token`` first, which an auth provider populates when one is
    configured, and otherwise verifies the request's own bearer tokens. That
    second path is what lets scopes be enforced behind the gateway with no auth
    provider, and therefore without a blanket 401 of our own.

    The verification is done once per request, whichever component's check asks
    first, and memoised on ``request.state`` for the rest.
    """
    if ctx.token is not None:
        return frozenset(ctx.token.scopes)

    request = _current_http_request()
    if request is not None:
        memoised = getattr(request.state, _REQUEST_SCOPES_STATE_ATTR, _UNSET)
        if memoised is not _UNSET:
            return cast("frozenset[str] | None", memoised)

    scopes = await _read_request_scopes()
    if request is not None:
        setattr(request.state, _REQUEST_SCOPES_STATE_ATTR, scopes)
    return scopes


async def _read_request_scopes() -> frozenset[str] | None:
    """Verify the request's own bearer tokens and read scopes off the first that does.

    Each candidate header is tried until one verifies, rather than only the first
    one carrying a value — see :func:`_presented_tokens`. Where more than one
    verifies, :data:`OAUTH_TOKEN_HEADERS` order decides, so a request that was
    already unambiguous keeps the answer it had.
    """
    presented = _presented_tokens()
    if not presented:
        return None

    verifier = _verifier()
    if verifier is None:
        logger.debug("Scope check: no verifier configured; treating as no OAuth token.")
        return None

    for token in presented:
        try:
            access = await verifier.verify_token(token)
        except Exception:
            logger.debug("Scope check: a bearer token could not be verified.", exc_info=True)
            continue
        if access is not None:
            return frozenset(access.scopes)

    # Very often a DataRobot API token rather than an OAuth one — expected
    # traffic, so it cannot warn per request. But saying it *once* matters: on
    # this server the caller is about to be served nothing, and if that is a
    # surprise, the reason (wrong issuer or audience, an unreachable JWKS, an
    # API key where a token was meant) is otherwise only visible at DEBUG.
    if not _state.warned_unverifiable:
        _state.warned_unverifiable = True
        logger.warning(
            "A request presented %d bearer token(s) and none verified, so the caller is "
            "served nothing. Expected for DataRobot API-key callers on a server that "
            "verifies OAuth tokens; if this caller should have verified, check the "
            "issuer, audience and JWKS URI. Further occurrences log at DEBUG.",
            len(presented),
        )
    else:
        logger.debug(
            "Scope check: none of the %d presented token(s) is an OAuth token for this server.",
            len(presented),
        )
    return None


def satisfies(required: frozenset[str], presented: frozenset[str] | None) -> bool:
    """Return whether the scopes on a request cover a requirement.

    A subset test, not equality: every required scope must be present, and extra
    scopes are not examined. ``None`` — nothing to judge by — is denied on a
    server that can verify tokens, and admitted on one that cannot.
    """
    if presented is None:
        return not (required and _state.settings.enforced)
    return required.issubset(presented)


def _declared_scopes_of(check: Any) -> frozenset[str]:
    """Return scopes recorded on an auth check by :func:`require_scopes`, or empty."""
    found = getattr(check, DECLARED_SCOPES_ATTR, None)
    return found if isinstance(found, frozenset) else frozenset()


def _as_check_list(auth: Any) -> list[Any]:
    """Return a component's auth checks as a list, whatever shape they are stored in."""
    if auth is None:
        return []
    return [auth] if callable(auth) else list(auth)


def require_scopes(*scopes: str) -> AuthCheck:
    """Require OAuth scopes on one component, declared where it is defined.

    A drop-in for ``fastmcp.server.auth.require_scopes`` with three differences
    that matter on a DataRobot-hosted server:

    * **The scope names stay readable.** FastMCP's version captures them in a
      closure nothing can reach, so a server declaring scopes only in code
      cannot enumerate them and ``scopes_supported`` has to be hand-maintained.
      These are recorded on the check and collected by
      :func:`collect_code_declared_scopes`.
    * **It works with no auth provider**, by verifying the request's own bearer
      token. FastMCP's version reads ``ctx.token``, which is ``None`` unless a
      provider is configured — so on a gateway-authenticated deployment it hides
      the component from *everyone*, including callers the gateway already
      authenticated.
    * **A caller with no usable OAuth token is let through** on a server that
      cannot verify tokens, keeping API-key deployments working. Once one can,
      the same caller is denied — see :attr:`ScopeSettings.enforced`.

    All listed scopes are required, not any one of them::

        @dr_mcp_tool(
            tags={"database"},
            auth=require_scopes("mcp:tools:execute", "mcp:tools:database:write"),
        )
        def run_sql(query: str) -> str: ...
    """
    required = frozenset(scopes)

    async def check(ctx: AuthContext) -> bool:
        if not _state.settings.code_active:
            return True
        return satisfies(required, await request_scopes(ctx))

    setattr(check, DECLARED_SCOPES_ATTR, required)
    return check


def restrict_tag_scopes(tag: str, scopes: list[str]) -> AuthCheck:
    """Require ``scopes`` on components carrying ``tag``.

    Ours rather than FastMCP's ``restrict_tag`` for the same reasons as
    :func:`require_scopes`: that one reads ``ctx.token`` and would hide every
    tagged component on a gateway-authenticated deployment.
    """
    required = frozenset(scopes)
    wanted = normalize_tag(tag)

    async def check(ctx: AuthContext) -> bool:
        if wanted not in {normalize_tag(each) for each in ctx.component.tags}:
            return True
        return satisfies(required, await request_scopes(ctx))

    setattr(check, DECLARED_SCOPES_ATTR, required)
    setattr(check, TAG_APPLIED_ATTR, True)
    return check


def require_verified_token() -> AuthCheck:
    """Require a token this server can verify, whatever scopes it carries.

    The floor beneath the per-component scope rules: those decide *which*
    components a caller reaches, this decides whether the caller is someone this
    server can identify at all. Attached to every component — including ones
    declaring no scopes — so a caller it cannot identify sees an empty server
    rather than a partial one.

    Without it, "no verifiable token" produced a half-open server: guarded
    components hidden, unguarded ones served. That is a hard shape to reason
    about, and it hands an unidentified caller a working tool surface on a
    deployment whose whole configuration says callers are meant to be identified.
    """

    async def check(ctx: AuthContext) -> bool:
        if not _state.settings.enforced:
            return True
        return await request_scopes(ctx) is not None

    setattr(check, TOKEN_FLOOR_ATTR, True)
    return check


async def _all_components(mcp: Any) -> list[Any]:
    """Return every registered component, unfiltered.

    The private listers rather than the public ``list_*()`` ones: the public
    calls apply these very auth checks, so a guarded component disappears from
    them — which is the point. Enumerating through them would be order-dependent
    and would skip already-guarded components on a re-run.

    Private API, so a fastmcp upgrade can rename them out from under us. The
    failure mode is an ``AttributeError`` from the first :func:`wire_scopes` at
    startup — loud and immediate, which is the acceptable end of that risk.
    """
    components: list[Any] = []
    for lister in (
        mcp._list_tools,
        mcp._list_resources,
        mcp._list_resource_templates,
        mcp._list_prompts,
    ):
        components.extend(await lister())
    return components


async def collect_code_declared_scopes(mcp: Any) -> set[str]:
    """Read back the scopes declared in code so they reach the published document.

    Must run *after* the component modules are imported, and after any dynamic
    registration: before that there is nothing to read.

    Checks attached by :func:`apply_tag_scopes` record their scopes too, but
    they are configuration, not code, and are skipped here — counting them
    would re-report every active tag rule as a code declaration, and under
    ``source=tags`` warn it inert against the very source enforcing it.
    """
    found: set[str] = set()
    for component in await _all_components(mcp):
        for check in _as_check_list(component.auth):
            if getattr(check, TAG_APPLIED_ATTR, False):
                continue
            found.update(_declared_scopes_of(check))

    _code_declared_scopes.clear()
    _code_declared_scopes.update(found)
    if found:
        logger.info("Scopes declared in code: %s", sorted(found))
    return found


async def declared_scopes_for_one_tool(mcp: Any, tool_name: str) -> frozenset[str] | None:
    for tool in await mcp._list_tools():
        if tool.name == tool_name:
            required: set[str] = set()
            for check in _as_check_list(tool.auth):
                required.update(_declared_scopes_of(check))
            return frozenset(required)
    return None


async def apply_tag_scopes(mcp: Any) -> int:
    """Attach per-tag scope requirements to the components carrying those tags.

    Returns the number of components that gained a check. Covers resources,
    resource templates and prompts as well as tools: this is the server's only
    scope enforcement, so leaving the other component types out would make them
    unguardable.

    A no-op unless the scope source selects tags.
    """
    settings = _state.settings
    guarded = 0
    matched: set[str] = set()
    tags_in_use: set[str] = set()
    for component in await _all_components(mcp):
        # Drop any checks a previous call attached, so re-wiring replaces the
        # tag rules rather than stacking a second copy of each. Checks declared
        # in code are left alone — they are not ours to remove.
        existing = [
            check
            for check in _as_check_list(component.auth)
            if not getattr(check, TAG_APPLIED_ATTR, False)
        ]
        checks: list[AuthCheck] = []
        if settings.tags_active:
            carried = {normalize_tag(tag) for tag in component.tags}
            tags_in_use.update(carried)
            hits = [tag for tag in settings.tag_scopes if tag in carried]
            matched.update(hits)
            checks = [restrict_tag_scopes(tag, list(settings.tag_scopes[tag])) for tag in hits]
        if checks or len(existing) != len(_as_check_list(component.auth)):
            component.auth = existing + checks
        guarded += 1 if checks else 0

    if not settings.tags_active:
        logger.debug("Tag scopes not applied: scope source is %s", settings.source.value)
        return 0

    if unmatched := sorted(set(settings.tag_scopes) - matched):
        # The tag comes from the variable name, so a misspelt one is not a
        # missing setting — it is a rule that reads as configured and guards
        # nothing at all.
        logger.warning(
            "Scope requirements are configured for tag(s) %s, which no registered "
            "component carries, so they guard nothing. Tags actually in use: %s",
            unmatched,
            sorted(tags_in_use),
        )

    logger.info(
        "Tag scopes applied to %d component(s) from tags %s",
        guarded,
        sorted(matched),
    )
    return guarded


async def apply_token_floor(mcp: Any) -> int:
    """Require a verifiable token on every component, once this server can verify one.

    Returns the number of components that gained the check. A no-op — and a
    remover, on a re-wire — when the server cannot verify tokens, which is what
    leaves API-key deployments untouched.

    Attached per component rather than as a provider-level 401 so the protected
    resource metadata stays reachable: a caller that cannot see a single tool can
    still read the well-known document and find out which authorization server to
    go to. A 401 would need an auth provider, which would answer for every route.
    """
    settings = _state.settings
    attached = 0
    for component in await _all_components(mcp):
        existing = [
            check
            for check in _as_check_list(component.auth)
            if not getattr(check, TOKEN_FLOOR_ATTR, False)
        ]
        checks = [require_verified_token()] if settings.enforced else []
        if checks or len(existing) != len(_as_check_list(component.auth)):
            component.auth = existing + checks
        attached += 1 if checks else 0

    if attached:
        logger.info(
            "A verifiable OAuth token is required on all %d component(s): this server can "
            "verify tokens, so a caller it cannot identify is served nothing.",
            attached,
        )
    return attached


def derived_scopes() -> list[str]:
    """Return every scope this server enforces, from whichever mechanisms are selected.

    Both mechanisms are readable data, which is what lets the published
    ``scopes_supported`` be generated rather than maintained by hand.
    """
    settings = _state.settings
    advertised: set[str] = set()
    if settings.code_active:
        advertised.update(_code_declared_scopes)
    if settings.tags_active:
        for scopes in settings.tag_scopes.values():
            advertised.update(scopes)
    return sorted(advertised)


def report_enforcement_state() -> None:
    """Say at startup whether the declared requirements can actually be enforced.

    A scope requirement fails silently in both directions — one that cannot be
    enforced still publishes, and one that is shadowed by the scope source still
    looks configured — so the state is stated once rather than inferred from
    behaviour later. A *partially* configured verifier is called out even when
    no requirement is declared at all: whoever set an issuer believed they were
    turning OAuth on, and nothing else about the server's behaviour would ever
    say otherwise.
    """
    settings = _state.settings
    required = derived_scopes()

    if settings.tag_scopes and not settings.tags_active:
        logger.warning(
            "Tag scope requirements %s are set but inert: the scope source is %r.",
            sorted(settings.tag_scopes),
            settings.source.value,
        )
    if _code_declared_scopes and not settings.code_active:
        logger.warning(
            "Scopes %s declared in code are inert: the scope source is %r.",
            sorted(_code_declared_scopes),
            settings.source.value,
        )

    if settings.enforced:
        if required:
            logger.info(
                "OAuth scope enforcement is active for %s. A caller with no verifiable token "
                "is served nothing at all; one holding a verifiable token reaches every "
                "component whose scopes it covers.",
                required,
            )
        return

    # The JWKS URI resolves from the issuer, so issuer and audience are the two
    # independent inputs — but report all three, because an explicit JWKS URI
    # alone is also a half-turned key.
    parts = {
        "audience": settings.audience,
        "issuer": settings.issuer,
        "JWKS URI": settings.resolved_jwks_uri(),
    }
    if any(parts.values()):
        logger.warning(
            "OAuth token verification is partially configured and therefore OFF: %s set, "
            "but no %s. Until verification resolves, every caller passes every scope "
            "check%s.",
            ", ".join(sorted(name for name, value in parts.items() if value)),
            " and no ".join(sorted(name for name, value in parts.items() if not value)),
            f" and the requirements {required} are published but not enforced" if required else "",
        )
        return
    if required:
        logger.warning(
            "OAuth scopes %s are enforced only against tokens this server can read, and "
            "it can read none: verification needs an issuer, an audience and a JWKS URI, "
            "and none is set. Until all three resolve, every caller passes every scope "
            "check.",
            required,
        )


async def probe_verification_keys() -> None:
    """Fetch the JWKS once so an unreachable IdP is a startup warning, not a mystery.

    A server that verifies tokens serves nothing to a caller it cannot identify,
    and a JWKS that cannot be fetched makes *every* caller such a caller: the
    verifier reports a failed key fetch the same way it reports an invalid
    token, at DEBUG. This one probe turns that failure mode into a line in the
    startup log. Reachability now is no guarantee for later — the IdP can still
    go down — so a failure only warns and the server starts anyway; an outage is
    not a configuration to refuse to boot on.

    A no-op on a server that cannot verify tokens. Not part of
    :func:`wire_scopes`, which must stay network-free: it runs on every re-wire,
    and in tests.
    """
    target = _state.settings.verification_target()
    if target is None:
        return
    jwks_uri, _, _ = target

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(jwks_uri)
        response.raise_for_status()
        keys = response.json().get("keys")
    except Exception as error:
        logger.warning(
            "The JWKS at %s could not be fetched (%s). Until it can be, no token "
            "verifies and every caller is served nothing.",
            jwks_uri,
            error,
        )
        return

    if isinstance(keys, list) and keys:
        logger.info("JWKS at %s is reachable (%d signing key(s)).", jwks_uri, len(keys))
    else:
        logger.warning(
            "The document at %s has no signing keys in it — is it really a JWKS URI? "
            "Until it serves keys, no token verifies and every caller is served nothing.",
            jwks_uri,
        )


async def wire_scopes(mcp: Any, settings: ScopeSettings | None = None) -> None:
    """Install settings, apply tag scopes, and read back code-declared ones.

    Call at startup *after* every component is registered, and again whenever
    one is registered later — wiring attaches checks to the components that
    exist, so a component added afterwards carries none of them until the next
    call. Re-wiring is idempotent: each pass replaces its own previous checks
    rather than stacking a second copy, and leaves checks declared in code
    alone. Called with no settings, the ones already installed are reused.
    """
    if settings is not None:
        configure_scopes(settings)
    await apply_tag_scopes(mcp)
    await apply_token_floor(mcp)
    await collect_code_declared_scopes(mcp)
    report_enforcement_state()
