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

# A2A Authentication

This guide covers how to configure authentication for Agent-to-Agent (A2A)
communication. There are two supported authentication methods:

1. **DataRobot API key** — simple bearer token auth for DataRobot-hosted agents.
2. **Okta cross-application access (XAA)** — two-step token exchange for
   federated Okta environments (hybrid RFC 8693 / RFC 7523 flow).

Both methods use the `authenticated_a2a_client` function group on the client
side and the `cross_application_access` server-side config to publish
requirements in the agent card.

## Installation

```bash
# With Okta XAA support (adds okta-client-python)
pip install "datarobot-genai[dragent,langgraph,auth]>=0.15.40"
```

Replace `langgraph` with `crewai` or `llamaindex` depending on your framework.

## Option 1: DataRobot API key authentication

Use this when calling a DataRobot-hosted A2A agent that accepts a standard
DataRobot API token.

### Environment variables

| Variable | Description |
|----------|-------------|
| `DATAROBOT_API_TOKEN` | Your DataRobot API token. |

The token is loaded automatically — no need to put it in `workflow.yaml`.

### `workflow.yaml`

```yaml
general:
  front_end:
    _type: dragent_fastapi
    step_adaptor:
      mode: 'off'
    a2a:
      server:
        name: My Agent
        description: A helpful assistant agent.

function_groups:
  remote_agent:
    _type: authenticated_a2a_client
    url: "https://app.datarobot.com/api/v2/deployments/<deployment-id>/directAccess/a2a/"
    auth_provider: datarobot_auth

llms:
  datarobot_llm:
    _type: datarobot-llm-component

workflow:
  _type: langgraph_agent
  llm_name: datarobot_llm
  description: My agent
  tool_names:
    - remote_agent

authentication:
  datarobot_auth:
    _type: datarobot_api_key
```

### How it works

1. On each A2A call, the `datarobot_api_key` auth provider injects the
   `DATAROBOT_API_TOKEN` as an `Authorization: Bearer <token>` header.
2. The remote agent's A2A endpoint validates the token against the DataRobot
   platform.

This is the simplest setup — no agent card extensions or multi-step flows
involved.

> **Important:** `datarobot_api_key` is the default authentication mechanism
> for DataRobot-hosted agents. However, when the remote agent card declares a
> specific mechanism (e.g. OAuth2 via `cross_application_access`), security-scheme
> negotiation validates and requires a matching auth provider on the client side.
> Use `okta_cross_app_access` (Option 2) for OAuth2-protected agents.

## Option 2: Okta cross-application access (XAA)

Use this when calling an agent protected by Okta's federated identity model.
The flow obtains a scoped access token through a two-step exchange.

### Prerequisites

- An Okta organization with Cross-Application Access enabled.
- A registered AI agent principal in Okta with a private key pair.
- The `auth` extra installed (`pip install "datarobot-genai[auth]"`).

### Environment variables

| Variable | Description |
|----------|-------------|
| `IDP_AGENT_ID` | Okta AI agent principal ID (used as `iss`/`sub` in JWT client assertions). |
| `IDP_AGENT_PRIVATE_KEY_JWK` | Base64-encoded or raw-JSON private JWK (signs JWT client assertions). |

Both are loaded automatically from env vars, `.env`, DataRobot Runtime
Parameters, or `file_secrets`.

### Full `workflow.yaml` example

```yaml
general:
  front_end:
    _type: dragent_fastapi
    step_adaptor:
      mode: 'off'
    a2a:
      server:
        name: Blog Content Writer
        description: >-
          An AI content writing agent that researches and writes
          well-structured blog posts.
      skills:
        - id: write_blog
          name: Write Blog Post
          description: Researches and writes a blog post on the given topic.
          tags: []
          examples:
            - Write a blog post about the future of AI in healthcare
            - Create an article about sustainable energy trends

      # Server-side: advertise XAA requirements in the agent card
      cross_application_access:
        # Step 1: Token exchange to fetch the ID-JAG (RFC 8693)
        token_exchange:
          trusted_issuer: "https://your-org.okta.com"
          audience: "https://your-org.okta.com/oauth2/ausXXXXXXXXXXXXXXX"
        # Step 2: Execute the Final Grant (RFC 7523)
        token_request:
          token_url: "https://your-org.okta.com/oauth2/ausXXXXXXXXXXXXXXX/v1/token"
          audience: "https://app.datarobot.com/<org_id>/<agent_id>"
          scopes:
            - "blog:write"

      # Optional: external identity and URL overrides published on the agent card
      external:
        id: "my-agent-id"       # Emitted as urn:datarobot:agent:identity:external
        url: "https://my-agent-id.example.com/a2a/"  # Overrides the auto-generated card URL

# Client-side: call a remote XAA-protected agent
function_groups:
  remote_agent:
    _type: authenticated_a2a_client
    url: "https://app.datarobot.com/api/v2/deployments/<deployment-id>/directAccess/a2a/"
    auth_provider: okta_auth

llms:
  datarobot_llm:
    _type: datarobot-llm-component

workflow:
  _type: langgraph_agent
  llm_name: datarobot_llm
  description: LangGraph planner/writer agent
  tool_names:
    - remote_agent

authentication:
  okta_auth:
    _type: okta_cross_app_access
    # okta_token_header: "x-custom-header"  # Optional: override default header name
```

### How it works

The XAA flow operates in two steps:

1. **Token Exchange (RFC 8693)** — The caller's incoming Okta access token
   (from the `okta_token_header`) is exchanged for an ID-JAG (Identity
   Assertion Authorization Grant) via the org-level Authorization Server
   (`token_exchange.trusted_issuer`).

2. **JWT Bearer Grant (RFC 7523)** — The ID-JAG is exchanged for a scoped
   access token at the resource AS token endpoint (`token_request.token_url`),
   granting access to the target agent with the requested scopes.

Both steps authenticate the client using the same method
(private key jwt), signing JWT client assertions
with the private key from `IDP_AGENT_PRIVATE_KEY_JWK`.

### Server-side configuration reference: `cross_application_access`

These fields are declared in the serving agent's `workflow.yaml` and published
on the A2A agent card.

| Field | Required | Default | Purpose |
|-------|----------|---------|---------|
| `token_exchange.trusted_issuer` | Yes | — | Org-level Authorization Server issuer URL. |
| `token_exchange.audience` | Yes | — | Resource AS base URL (where ID-JAG is fetched from). |
| `token_request.token_url` | Yes | — | Token endpoint of the resource AS. |
| `token_request.audience` | Yes | — | Final resource identifier for the agent. |
| `token_request.scopes` | No | `["read_data"]` | Scopes the caller must request. |

> **Note:** `grant_type` URNs are injected automatically by the generator — do not
> set them in `workflow.yaml`. The agent card will always contain
> `urn:ietf:params:oauth:grant-type:token-exchange` (Step 1) and
> `urn:ietf:params:oauth:grant-type:jwt-bearer` (Step 2).

### Server-side configuration reference: `external`

Optional fields under `general.frontend.a2a.external` that control additional
identity metadata and the agent card URL.

| Field | Purpose |
|-------|---------|
| `external.id` | Catalog discovery identifier. Emitted as the `urn:datarobot:agent:identity:external` extension on the agent card. |
| `external.url` | Overrides the auto-generated agent card endpoint URL. Used as-is — no normalization is applied. |

### Client-side configuration reference: `okta_cross_app_access`

These fields configure how the calling agent authenticates when invoking a
remote XAA-protected agent.

| Field | Default | Purpose |
|-------|---------|---------|
| `okta_token_header` | `x-datarobot-external-access-token` | Incoming request header carrying the caller's Okta access token. |
| `principal_id` | `IDP_AGENT_ID` env var | Okta AI agent principal ID. |
| `private_jwk` | `IDP_AGENT_PRIVATE_KEY_JWK` env var | Base64-encoded or raw-JSON private JWK. |

Exchanged access tokens are cached by default to avoid repeating the two-step
Okta flow on every A2A call. Configure via:

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_CARD_XAA_TOKEN_CACHE_ENABLED` | `true` | Enable exchanged-token cache. |
| `AGENT_CARD_XAA_TOKEN_CACHE_BACKEND` | `memory` | `memory` or `redis` (uses registry Redis URL/prefix). |
| `AGENT_CARD_XAA_TOKEN_SKEW_SECONDS` | `60` | Evict cached tokens this many seconds before JWT `exp`. |
| `AGENT_CARD_XAA_TOKEN_MAX_TTL_SECONDS` | `3600` | Cap cache TTL regardless of token `exp`. |

### Agent card mapping

The `cross_application_access` configuration is split across two parts of the
published A2A agent card:

- **`securitySchemes.oauth2.flows.clientCredentials`** — Standard OpenAPI
  fields: `tokenUrl` (from `token_request.token_url`) and `scopes` (from
  `token_request.scopes`).
- **`capabilities.extensions[]`** (URI: `urn:ietf:params:oauth:grant-type:jwt-bearer`)
  — Non-standard XAA parameters for SDK consumption:
  `token_endpoint_auth_method`, `token_exchange.*` (including hardcoded
  `grant_type` and `requested_token_type` URNs), and `token_request.audience`.

In addition, the agent card may include up to two identity extensions:

- **`urn:datarobot:agent:identity:internal`** — Emitted automatically in deployed
  environments (when `MLOPS_DEPLOYMENT_ID` is set). Carries `deployment_id` for
  internal DataRobot routing. Not emitted in local development.
- **`urn:datarobot:agent:identity:external`** — Emitted when `external.id` is
  set in `workflow.yaml`. Carries the developer-supplied catalog identifier.

This separation follows the A2A spec convention: standard OAuth fields belong
in `securitySchemes`, while flow-specific parameters go in
`capabilities.extensions`.

## Troubleshooting

### Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Authorization` header missing on A2A RPC calls | The remote agent card declares `securitySchemes` but the client uses `datarobot_api_key`. When `securitySchemes` are present, the `A2ACredentialService` performs OAuth2 security-scheme negotiation and drops incompatible credentials. | Switch to an OAuth2-compatible auth provider (e.g. `okta_cross_app_access`) that matches the security scheme advertised by the remote agent card. |
| `RuntimeError: Header 'x-datarobot-external-access-token' not found` | The incoming request doesn't carry the Okta token. | Ensure the upstream caller forwards the Okta access token in the expected header. |
| `ValueError: principal_id is required` | `IDP_AGENT_ID` env var not set. | Set `IDP_AGENT_ID` in your environment or Runtime Parameters. |
| `ValueError: Could not parse private_jwk` | `IDP_AGENT_PRIVATE_KEY_JWK` is neither valid base64-encoded JSON nor raw JSON. | Verify your JWK — try `echo $IDP_AGENT_PRIVATE_KEY_JWK | base64 -d | python -m json.tool`. |
| `ValueError: Agent card ... missing required fields` | Remote agent card doesn't have the XAA extension. | Verify the remote agent has `cross_application_access` configured. |
| `RuntimeError: Failed to fetch agent card` | Network/auth issue reaching the agent card URL. | Check the `url` in your `function_groups` config and network connectivity. |
