<!--
  ~ Copyright 2026 DataRobot, Inc. and its affiliates.
  ~
  ~ Licensed under the Apache License, Version 2.0 (the "License");
  ~ you may not use this file except in compliance with the License.
  ~ You may obtain a copy of the License at
  ~
  ~     http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS,
  ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ~ See the License for the specific language governing permissions and
  ~ limitations under the License.
-->

# OAuth scopes and Cross-Application Access

An MCP server published by this library advertises scopes in two places, and
they are enforced by two different parties. This is what each one is, who
checks it, and what a client should ask for.

It was written from a live failure: an agent that had completed the
Cross-Application Access exchange, reached the MCP server, and then reported
"I only have access to `whoami`".

## Two lists, two gates

| | `scopes_supported` | `cross_application_access.token_request.scopes` |
|---|---|---|
| Example value | `mcp:tools:admin`, `mcp:tools:builder` | `dr.impersonation` |
| Generated from | `derived_scopes()` — every `require_scopes(...)` declared in code plus every `MCP_OAUTH_TAG_SCOPES_<TAG>` setting | the `MCP_XAA_SCOPES` setting, verbatim |
| Enforced by | **this server's own ASGI middleware**, per `tools/call`, against the called tool's declarations | **the DataRobot public-API gateway**, before the request reaches the workload |
| Answers | "may this token call these tools?" | "may this token act on a DataRobot user's behalf at all?" |
| Code | `drmcpbase.oauth_scopes`, `drmcp.core.middleware` | DataRobot platform setting, not this library |

### What `dr.impersonation` actually is

It is what the **exchange asks for**, and — where an install configures it —
an **admission** scope. It is not a minting permission, and it is not
unconditionally required.

Two separate things get conflated here, and this document previously conflated
them:

1. **`cross_application_access.token_request.scopes`.** By the block's own
   definition (`CrossAppTokenRequest.scopes`, "Scopes requested in Step 2")
   this is what the Cross-App exchange asks the authorization server for on its
   second hop. In this demo's Okta tenant it is also the *anchor* scope whose
   access-policy rule decides who may obtain tokens at all
   (`scripts/okta_setup.py --anchor-scope`), which is why it looks load-bearing.

2. **`PUBLIC_API_JWT_REQUIRED_SCOPES`.** A DataRobot platform setting
   (`<dr>/classic-admin/system-configuration/jwtTokens`) — "space-delimited
   scopes that must be present". Where it is set, a token short of those scopes
   is refused before the workload sees it. It is configured **per install**, it
   is published in no document a client can read, and it is empty in installs
   where a token without `dr.impersonation` reaches the workload perfectly
   well.

The two are not the same field and need not hold the same value; the demo sets
both to `dr.impersonation`, which is what made them look like one rule.

Neither `api-gateway` nor `api-gateway-ext` checks an OAuth scope at all — the
only `scopes` in either repo are an Azure Redis credential and a sushi
validator called with `nil`. An external JWT is resolved by
`/api/v2/account/info/`; the gateway then rewrites `Authorization` to a minted
impersonated token and forwards the caller's original in
`x-datarobot-external-access-token` (for `dr-workload-type` `mcp`/`agent`
only — a client cannot inject that header, it is overwritten or removed).

Nothing in this library reads the scope either: no tool declares it, and the
scope-validation middleware only ever checks a `tools/call` against the called
tool's own declarations.

It is easy to describe it as "the scope that lets an agent mint tokens",
because in practice the two coincide — Okta only grants `dr.impersonation` to a
client its policy allows it for, so "can obtain `dr.impersonation`" is close to
"is allowed to impersonate a user". But the enforcement point is DataRobot's
gateway, not Okta's minting, and the distinction matters when debugging: a
missing `dr.impersonation` is a 401 at the door, while a missing
`mcp:tools:builder` is a tool quietly absent from `tools/list`.

Observed end to end in the A2A chain: external service app → agent 1 → agent 2
→ the final MCP token, every hop carrying `dr.impersonation` and nothing else,
with access gated by the per-hop audience binding rather than by differentiated
scopes.

### Why the XAA block carries it

Because the MCP server sits behind the gateway. The exchange has to produce a
token the gateway will admit, so a client has to know to ask for
`dr.impersonation` — and `token_request.scopes` is the only place the document
says so.

## What `scopes_supported` publishes: the union

`MCPOAuthProtectedResourceMetadataConfig.from_settings` publishes
`scopes_supported` as the **union** of the server's own scope declarations and
`cross_application_access.token_request.scopes`.

RFC 9728 §2 defines the field as the scope values "used in authorization
requests to request access to this protected resource" — not "the scopes this
resource enforces". A token that works needs the gateway's admission scope
*and* the tool scopes, so the union is the honest answer to the question the
field asks.

Publishing the tool declarations alone describes a set no working token can be
minted from. A client that asks for exactly `scopes_supported` — which is what
`mcp-remote` does, and with it Cursor and Claude Code — then authorizes
cleanly and is refused at every call.

This changes nothing about enforcement. Each `tools/call` is still checked
against the called tool's own declarations, never against this list.

Cross-Application Access is our extension and the RFCs say nothing about the
`cross_application_access` member. They do say something about
`scopes_supported`, and the union is the reading its own wording supports.

## What a client should ask for

`scopes_supported`, and nothing else.

That is one list and the whole requirement, because the server already folded
both halves into it. It is also what every other MCP client does, so the agent
behaves like the rest of the ecosystem rather than on a private rule.

`token_request.scopes` stays in the document as the *input* to that union, and
as the one place a reader can see which half is the gateway's. It is not a
second list to ask from: a client that combined the two itself would be
second-guessing the server, and a server that deliberately chose not to
advertise a scope would find it asked for anyway.

### The bug this fixes

`datarobot_user_mcp_xaa_client.parse_xaa_params_from_mcp_auth_server_metadata`
used to set `id_jag_scopes` from `token_request.scopes` alone. On a server
publishing `dr.impersonation` there and `mcp:tools:*` in `scopes_supported`,
the exchange produced a token that cleared the gateway and opened no tool:
every tool with a scope requirement is hidden from `tools/list` and answered
403 on `tools/call`, leaving only the tools that declare none.

That surfaces as *"I only have access to `whoami`"* — which reads as the server
exposing one tool rather than as a token short of a scope, and sends you
looking in the wrong place.

It now reads `scopes_supported`. See `_requested_scopes` in that module.

## What is actually granted

Asking is not getting. Okta issues the **intersection** of the request with the
agent's resource connection, so a connection granting less than the server
advertises hands back less — which is how a "builder but not admin"
demonstration is set up, and it is working as intended.

One case costs something: a client that is not *allowed* a scope at all can
have the whole exchange refused with `invalid_scope` rather than having the
request trimmed. If a server advertises more than a given agent may have, pin
`cross_application_access` on the MCP client plugin's own config to ask for an
exact list instead of reading the document.

## Checklist when a tool is missing

1. **Refused at the door, nothing reached the server.** Read the
   `WWW-Authenticate` header rather than guessing: `error="invalid_token"` with
   "audience claim validation failed" is the `aud` check, and that is the
   common one. The fix is usually the **`resource` parameter**, not the server:
   an authorization-code flow gets the authorization server's *default*
   audience only when it asks for a resource the AS does not recognise. Ask
   for the value the document publishes as
   `cross_application_access.token_request.audience` — which Okta does honour
   when that URI is a configured audience on the AS — and an ordinary PKCE
   token comes back correctly bound, no exchange and no server change needed.
   (`mcp-remote --resource <that URI>`; the inspector sends its Audience field
   there, prefilled from the same value.) Only where the install sets
   `PUBLIC_API_JWT_REQUIRED_SCOPES` is a missing scope also a door check.
2. **Connected, but a tool is absent from `tools/list`.** The token lacks that
   tool's declared scope. Compare the token's `scp` with the tool's
   `require_scopes(...)` — `scopes_supported` is the union of every such
   declaration.
3. **Connected, the tool is listed, `tools/call` answers 403
   `insufficient_scope`.** Same cause, seen from the other side: the
   middleware checks the call even when the listing let it through.
4. **The request asked for a scope and the token came back without it.** Okta
   granted the intersection with the agent's resource connection. Change the
   connection's grant, not the request.

## Where this lives

| Concern | Module |
|---|---|
| Scope declarations (`require_scopes`, tag settings, `derived_scopes`) | `drmcpbase.oauth_scopes` |
| Published document, including the union | `drmcpbase.oauth_protected_resource_metadata.entities` |
| Server settings (`MCP_XAA_*`, `MCP_OAUTH_*`) | `drmcp.core.config` |
| Per-`tools/call` enforcement | `drmcp.core.middleware` |
| The agent's side of the exchange | `dragent.plugins.datarobot_user_mcp_xaa_client` |
| The two-hop flow itself (RFC 8693 → ID-JAG → RFC 7523) | `dragent.plugins.okta_a2a_auth` |
