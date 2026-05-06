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

---

## Installation

```bash
# With Okta XAA support (adds okta-client-python)
pip install "datarobot-genai[dragent,langgraph,auth]>=0.15.35"
```

Replace `langgraph` with `crewai` or `llamaindex` depending on your framework.

---

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

---

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
| `PRINCIPAL_ID` | Okta AI agent principal ID (used as `iss`/`sub` in JWT client assertions). |
| `PRIVATE_JWK` | Base64-encoded or raw-JSON private JWK (signs JWT client assertions). |

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
        token_endpoint_auth_method: "private_key_jwt"

        # Step 1: Token exchange to fetch the ID-JAG (RFC 8693)
        token_exchange:
          trusted_issuer: "https://your-org.oktapreview.com"
          audience: "https://your-org.oktapreview.com/oauth2/ausXXXXXXXXXXXXXXX"

        # Step 2: Execute the Final Grant (RFC 7523)
        token_request:
          grant_type: "urn:ietf:params:oauth:grant-type:jwt-bearer"
          token_url: "https://your-org.oktapreview.com/oauth2/ausXXXXXXXXXXXXXXX/v1/token"
          audience: "https://app.datarobot.com/<org_id>/<agent_id>"
          scopes:
            - "blog:write"

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
    # id_jag_scopes: ["openid", "profile"]  # Optional: override Step 1 scopes
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
(`token_endpoint_auth_method: private_key_jwt`), signing JWT client assertions
with the private key from `PRIVATE_JWK`.

### Server-side configuration reference: `cross_application_access`

These fields are declared in the serving agent's `workflow.yaml` and published
on the A2A agent card.

| Field | Section | Purpose |
|-------|---------|---------|
| `token_endpoint_auth_method` | root | How the client authenticates to token endpoints (e.g. `private_key_jwt`). |
| `token_exchange.trusted_issuer` | Step 1 | Org-level Authorization Server issuer URL. |
| `token_exchange.audience` | Step 1 | Resource AS base URL (where ID-JAG is fetched from). |
| `token_request.grant_type` | Step 2 | Must be `urn:ietf:params:oauth:grant-type:jwt-bearer`. |
| `token_request.token_url` | Step 2 | Token endpoint of the resource AS. |
| `token_request.audience` | Step 2 | Final resource identifier for the agent. |
| `token_request.scopes` | Step 2 | Scopes the caller must request (default: `["read_data"]`). |

### Client-side configuration reference: `okta_cross_app_access`

These fields configure how the calling agent authenticates when invoking a
remote XAA-protected agent.

| Field | Default | Purpose |
|-------|---------|---------|
| `okta_token_header` | `x-datarobot-okta-access-token` | Incoming request header carrying the caller's Okta access token. |
| `principal_id` | `PRINCIPAL_ID` env var | Okta AI agent principal ID. |
| `private_jwk` | `PRIVATE_JWK` env var | Base64-encoded or raw-JSON private JWK. |
| `id_jag_scopes` | `["read_data"]` | Scopes for the Step 1 ID-JAG request. |

### Agent card mapping

The `cross_application_access` configuration is split across two parts of the
published A2A agent card:

- **`securitySchemes.oauth2.flows.clientCredentials`** — Standard OpenAPI
  fields: `tokenUrl` (from `token_request.token_url`) and `scopes` (from
  `token_request.scopes`).
- **`capabilities.extensions[]`** (URI: `urn:ietf:params:oauth:grant-type:jwt-bearer`)
  — Non-standard XAA parameters for SDK consumption:
  `token_endpoint_auth_method`, `token_exchange.*`, and
  `token_request.grant_type` / `token_request.audience`.

This separation follows the A2A spec convention: standard OAuth fields belong
in `securitySchemes`, while flow-specific parameters go in
`capabilities.extensions`.

---

## Choosing between the two methods

| Criteria | DataRobot API key | Okta XAA |
|----------|-------------------|----------|
| Setup complexity | Minimal — one env var. | Moderate — Okta principal + key pair. |
| Identity model | Service-level API token. | User-delegated federated identity. |
| Token type | Static API key. | Short-lived scoped access token. |
| Use case | Internal DataRobot agents. | Cross-organization / federated agents. |
| Extra install | None. | `auth` extra required. |

---

## Troubleshooting

### Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `RuntimeError: Header 'x-datarobot-okta-access-token' not found` | The incoming request doesn't carry the Okta token. | Ensure the upstream caller forwards the Okta access token in the expected header. |
| `ValueError: principal_id is required` | `PRINCIPAL_ID` env var not set. | Set `PRINCIPAL_ID` in your environment or Runtime Parameters. |
| `ValueError: Could not parse private_jwk` | `PRIVATE_JWK` is neither valid base64-encoded JSON nor raw JSON. | Verify your JWK — try `echo $PRIVATE_JWK | base64 -d | python -m json.tool`. |
| `ValueError: Agent card ... missing required fields` | Remote agent card doesn't have the XAA extension. | Verify the remote agent has `cross_application_access` configured. |
| `RuntimeError: Failed to fetch agent card` | Network/auth issue reaching the agent card URL. | Check the `url` in your `function_groups` config and network connectivity. |
