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

The DataRobot gateway does the *authentication*; the server's ASGI middleware
does the *authorization* (see ``drmcp.core.middleware``): the audience claim is
validated on every request and a ``tools/call`` is checked against the scopes
the tool declares. This module owns those *declarations* — which scopes a
component requires. ``scopes_supported`` in the published RFC 9728 document
cannot do that job on its own: it is a flat list with no component in it, so it
can say "this server understands ``mcp:tools:write``" but never "``run_sql``
requires ``mcp:tools:write``". Something has to bind component to scopes, and
there are two places to write that binding:

in code
    ``@dr_mcp_tool(auth=require_scopes("mcp:tools:write"))`` — :func:`require_scopes`
    from this module (re-exported by ``datarobot_genai.drmcp``), on the tool's own
    decorator. The requirement travels with the component it guards and survives
    a tag rename.

in configuration
    a scope requirement keyed on a tag the component already declares, so one
    setting guards every component carrying it and can differ per environment
    without a code change.

The checks attached here do two jobs. They **record** the requirement on the
component itself, where :func:`declared_scopes_for_one_tool` (what the
scope-validation middleware enforces a ``tools/call`` against) and
:func:`derived_scopes` (what the published ``scopes_supported`` is generated from)
can read it back. And they **enforce** it at the tool level with the same subset
test (:func:`satisfies`) against the scopes on the request's token — the token
the DataRobot gateway authenticated and ``OAuthJWTTokenHandlerMiddleware`` parsed
into ``request.scope["user"]``, which FastMCP hands to the check as ``ctx.token``
— so an under-scoped token does not see the component in ``tools/list``, while a
``tools/call`` is refused with 403 by the middleware before FastMCP is reached. No
token on the request (the gate off, or no request at all) admits: this module
verifies nothing itself, authentication is the gateway's job. Server-side listings
that must see every component regardless of the caller's token (the ``/static/*``
catalog, the lineage sync) run under :func:`without_component_auth_checks`.

The scope source selects which declaration mechanism is live and defaults to
``both``, so each mechanism simply applies wherever it is declared; set
``code`` or ``tags`` only to silence the other one. It governs what the
middleware enforces *and* the published ``scopes_supported`` together, so the
server can never advertise a scope it is not enforcing — that list is always
derived from the declarations, never hand-written.

Matching (done by the middleware) is a subset test, never equality. Every
required scope must be present; anything else the token carries is not examined,
because one token serves a whole session and normally holds the scopes for every
component the client might call. Requirements also stack: each declaration adds
its own scopes and a ``tools/call`` must cover the union, so a component
matching two mapped tags — or carrying a code declaration *and* a tag under
``both`` — requires all of them. Nothing is any-of.

This module holds the mechanism and nothing about where settings come from:
:class:`ScopeSettings` is handed in by the server that owns the configuration,
the same way the protected-resource metadata entities take theirs.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from typing import Any

from fastmcp.server.auth import AuthCheck
from fastmcp.server.auth import AuthContext
from fastmcp.server.context import reset_transport
from fastmcp.server.context import set_transport

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
        """Whether ``require_scopes(...)`` declared on components in code are read."""
        return self in {ScopeSource.CODE, ScopeSource.BOTH}

    @property
    def reads_tags(self) -> bool:
        """Whether the tag-keyed requirements are read."""
        return self in {ScopeSource.TAGS, ScopeSource.BOTH}


#: Attribute :func:`require_scopes` records its scope names on, so they can
#: be read back and published. FastMCP's own ``require_scopes`` keeps them in a
#: closure where nothing can reach them.
DECLARED_SCOPES_ATTR = "dr_declared_scopes"

#: Marks a check that :func:`apply_tag_scopes` attached, so re-wiring can strip
#: its own previous work instead of stacking a second copy of every rule.
TAG_APPLIED_ATTR = "dr_tag_scope_check"


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
    """What the scope declarations need, independent of where they were configured."""

    #: Which declaration mechanism is read. Both, unless narrowed.
    source: ScopeSource = ScopeSource.BOTH
    #: ``{tag: scopes a caller must hold}``, keyed by :func:`normalize_tag`.
    #: All scopes listed for a tag are required, not any one of them.
    tag_scopes: Mapping[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", ScopeSource.parse(self.source))
        object.__setattr__(
            self,
            "tag_scopes",
            {normalize_tag(tag): list(scopes) for tag, scopes in self.tag_scopes.items()},
        )

    @property
    def code_active(self) -> bool:
        """Whether ``require_scopes(...)`` declared on components in code are read."""
        return self.source.reads_code

    @property
    def tags_active(self) -> bool:
        """Whether the tag-keyed requirements are read."""
        return self.source.reads_tags


class _State:
    """Holder for the settings in force.

    An object rather than a module-level name so installing new settings is an
    attribute write, not a ``global`` rebind. It has to live at module level at
    all because declarations are attached at wiring time but read back per
    request, and the readers have no route to the server that configured them.
    """

    settings = ScopeSettings()


_state = _State()

# Scopes declared in code, read back once at startup.
_code_declared_scopes: set[str] = set()


def configure_scopes(settings: ScopeSettings) -> None:
    """Install the settings every declaration reader consults."""
    _state.settings = settings


def active_settings() -> ScopeSettings:
    """Return the settings currently in force."""
    return _state.settings


def reset_scope_state() -> None:
    """Forget installed settings and collected scopes."""
    _state.settings = ScopeSettings()
    _code_declared_scopes.clear()


def _declared_scopes_of(check: Any) -> frozenset[str]:
    """Return scopes recorded on an auth check by :func:`require_scopes`, or empty."""
    found = getattr(check, DECLARED_SCOPES_ATTR, None)
    return found if isinstance(found, frozenset) else frozenset()


def _as_check_list(auth: Any) -> list[Any]:
    """Return a component's auth checks as a list, whatever shape they are stored in."""
    if auth is None:
        return []
    return [auth] if callable(auth) else list(auth)


def request_scopes(ctx: AuthContext) -> frozenset[str] | None:
    """Return the scopes on this request's token, or ``None`` when it carries none.

    ``ctx.token`` is what FastMCP read off ``request.scope["user"]`` — set by drmcp's
    ``OAuthJWTTokenHandlerMiddleware`` from the gateway-authenticated token while
    ``MCP_ENABLE_OAUTH_CLAIM_VALIDATION`` is on. Nothing is verified here.
    """
    if ctx.token is None:
        return None
    return frozenset(ctx.token.scopes or ())


def satisfies(required: frozenset[str], presented: frozenset[str] | None) -> bool:
    """Return whether the scopes on a request cover a requirement.

    A subset test, never equality: every required scope must be present, anything
    else the token carries is not examined. ``None`` — no token on the request —
    admits: with the gate off no token is parsed, and outside a request there is
    nothing to check against.
    """
    if presented is None:
        return True
    return required.issubset(presented)


def require_scopes(*scopes: str) -> AuthCheck:
    """Require OAuth scopes on one component, declared where it is defined.

    Use on the tool's own decorator — ``@dr_mcp_tool(auth=require_scopes("a", "b"))``;
    all listed scopes are required, not any one of them. The check enforces
    :func:`satisfies` against the request's token (:func:`request_scopes`), so a
    token short of a scope does not see the tool in ``tools/list``; the
    scope-validation middleware refuses its ``tools/call`` with 403
    ``insufficient_scope`` first. A request without a token is admitted. It also
    records the scope names on the component (:data:`DECLARED_SCOPES_ATTR`) so the
    middleware, ``scopes_supported`` and the REST ``required_scopes`` field can read
    them back — FastMCP's own ``require_scopes`` keeps them in a closure and fails a
    tokenless request, which is why it is not used directly.

    Inert under ``MCP_OAUTH_SCOPE_SOURCE=tags``.
    """
    required = frozenset(scopes)

    async def check(ctx: AuthContext) -> bool:
        if not _state.settings.code_active:
            return True
        return satisfies(required, request_scopes(ctx))

    setattr(check, DECLARED_SCOPES_ATTR, required)
    return check


def restrict_tag_scopes(tag: str, scopes: list[str]) -> AuthCheck:
    """Require ``scopes`` on a component carrying ``tag``.

    The same contract as :func:`require_scopes` — subset test against the request's
    token, tokenless requests admitted, scope names recorded. :func:`apply_tag_scopes`
    attaches it only to the components that actually carry the tag, so no membership
    test is needed at check time. The tag marker records which tag produced the check
    (any truthy value marks it as configuration rather than code), so re-wiring can
    replace these without touching in-code declarations. Inert under
    ``MCP_OAUTH_SCOPE_SOURCE=code``.
    """
    required = frozenset(scopes)

    async def check(ctx: AuthContext) -> bool:
        if not _state.settings.tags_active:
            return True
        return satisfies(required, request_scopes(ctx))

    setattr(check, DECLARED_SCOPES_ATTR, required)
    setattr(check, TAG_APPLIED_ATTR, normalize_tag(tag))
    return check


async def _all_components(mcp: Any) -> list[Any]:
    """Return every registered component, unfiltered.

    The private listers rather than the public ``list_*()`` ones: enumerating
    through the public calls would apply whatever auth checks components carry,
    making the walk order-dependent. Private API, so a fastmcp upgrade can
    rename them out from under us. The failure mode is an ``AttributeError``
    from the first :func:`wire_scopes` at startup — loud and immediate, which
    is the acceptable end of that risk.
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


def declared_scopes_of_component(component: Any) -> frozenset[str]:
    """Return the scopes *component* declares, whichever way they were declared.

    The union across both declaration spellings — ``require_scopes(...)`` declared
    on the component in code and tag-keyed configuration.
    Honours the scope source, so a declaration the source silences is not
    reported either: in-code
    declarations contribute nothing under ``source=tags``. (Tag rules under
    ``source=code`` are never attached at all — see :func:`apply_tag_scopes` —
    but the tag test keeps a stale check harmless between re-wires.)

    One reader for everything that answers "what does this component require":
    the scope-validation middleware (via :func:`declared_scopes_for_one_tool`)
    and the static-catalog REST route's ``required_scopes`` field, so what is
    reported is always what is enforced.
    """
    settings = _state.settings
    required: set[str] = set()
    for check in _as_check_list(getattr(component, "auth", None)):
        if getattr(check, TAG_APPLIED_ATTR, False):
            if not settings.tags_active:
                continue
        elif not settings.code_active:
            continue
        required.update(_declared_scopes_of(check))
    return frozenset(required)


async def declared_scopes_for_one_tool(mcp: Any, tool_name: str) -> frozenset[str] | None:
    """Return the scopes *tool_name* declares, or ``None`` when no such tool exists.

    What the scope-validation middleware checks a ``tools/call``'s token
    against — :func:`declared_scopes_of_component`, looked up by name.
    """
    for tool in await mcp._list_tools():
        if tool.name == tool_name:
            return declared_scopes_of_component(tool)
    return None


@contextmanager
def without_component_auth_checks() -> Iterator[None]:
    """Make FastMCP skip per-component ``auth`` checks for the listings inside.

    For code that describes the server rather than serving a caller — the
    ``/static/*`` catalog, the lineage sync. FastMCP evaluates ``auth`` checks inside
    every ``list_tools()`` against the current request's token, so such a listing
    made during a REST call would shrink to what *that* caller may call. FastMCP
    skips component auth only for the stdio transport (see its ``_get_auth_context``),
    so this borrows that switch for the duration of the block through FastMCP's
    public ``set_transport`` / ``reset_transport``.
    """
    token = set_transport("stdio")
    try:
        yield
    finally:
        reset_transport(token)


async def apply_tag_scopes(mcp: Any) -> int:
    """Attach per-tag scope declarations to the components carrying those tags.

    Returns the number of components that gained a declaration. Covers
    resources, resource templates and prompts as well as tools: this is where
    every component's requirements are recorded, so leaving the other component
    types out would make them undeclarable.

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


def derived_scopes() -> list[str]:
    """Return every scope this server declares, from whichever mechanisms are selected.

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
    """Say at startup which scope requirements are declared and which are inert.

    A declaration silenced by the scope source fails silently in both
    directions — it still publishes nothing and it still *looks* configured —
    so the state is stated once rather than inferred from behaviour later.
    Enforcement itself is the scope-validation middleware's job (each
    ``tools/call`` is checked against :func:`declared_scopes_for_one_tool`),
    so what this module can report on is the declarations.
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

    if required:
        logger.info(
            "OAuth scope requirements declared for %s. Each tools/call is checked "
            "against the called tool's declared scopes by the scope-validation "
            "middleware, and the union is published as scopes_supported.",
            required,
        )


async def wire_scopes(mcp: Any, settings: ScopeSettings | None = None) -> None:
    """Install settings, apply tag scopes, and read back code-declared ones.

    Call at startup *after* every component is registered, and again whenever
    one is registered later — wiring attaches declarations to the components
    that exist, so a component added afterwards carries none of them until the
    next call. Re-wiring is idempotent: each pass replaces its own previous
    declarations rather than stacking a second copy, and leaves checks declared
    in code alone. Called with no settings, the ones already installed are
    reused.
    """
    if settings is not None:
        configure_scopes(settings)
    await apply_tag_scopes(mcp)
    await collect_code_declared_scopes(mcp)
    report_enforcement_state()
