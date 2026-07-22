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

# A2A client (calling remote agents)

A NAT workflow can call other agents over the [A2A protocol](https://google.github.io/A2A/) by
adding an `authenticated_a2a_client` entry under **`function_groups`**. The remote agent appears as
a tool the orchestrator can invoke, just like MCP tools or local functions.

```yaml
function_groups:
  remote_agent:
    _type: authenticated_a2a_client
    url: "https://app.datarobot.com/api/v2/deployments/<deployment-id>/directAccess/a2a/"
    auth_provider: datarobot_auth

workflow:
  _type: langgraph_agent
  llm_name: datarobot_llm
  tool_names:
    - remote_agent          # ← the orchestrator can now call this agent

authentication:
  datarobot_auth:
    _type: datarobot_api_key
```

The function group handles agent card discovery, authentication for both the discovery and RPC
phases, and SSE streaming, all driven by `workflow.yaml`.

For details on which auth providers are available and how to configure them (DataRobot API key,
Okta XAA), see [a2a-auth.md](a2a-auth.md).

## Agent card resolution

Before the first RPC call, the client must obtain the remote agent's **agent card** — a JSON
document describing the agent's capabilities and authentication requirements. There are two
mutually exclusive ways to obtain it.

### Direct fetch (`url`)

This is the simplest setup — use it when the card endpoint is directly reachable with the same
credentials used for RPC calls.

In a direct fetch, the client fetches the card from `{url}/.well-known/agent-card.json`. The `auth_provider` is used
for both the card fetch and subsequent RPC calls.

```yaml
function_groups:
  remote_agent:
    _type: authenticated_a2a_client
    url: "https://app.datarobot.com/api/v2/deployments/<deployment-id>/directAccess/a2a/"
    auth_provider: datarobot_auth
```


### Central registry (`registry`)

In DataRobot deployments, the agent card endpoint is protected by per-agent AuthN/AuthZ. However, the
card itself describes *how* to authenticate, creating a chicken-and-egg problem. The **central
agent card registry** solves this by exposing all agent cards in the tenant at a single endpoint
that requires only a standard `DATAROBOT_API_TOKEN`.

**Lookup by deployment ID:**

```yaml
function_groups:
  remote_agent:
    _type: authenticated_a2a_client
    registry:
      deployment_id: "64a1b2c3d4e5f6a7b8c9d0e1"
    auth_provider: okta_auth
```

**Lookup by external ID** (when the remote agent sets `general.front_end.a2a.external.id`):

```yaml
function_groups:
  remote_agent:
    _type: authenticated_a2a_client
    registry:
      external_id: "my-remote-agent"
    auth_provider: okta_auth
```

> [!NOTE]
> When using the registry the RPC base URL is derived from the card's advertised `url` — you do not need to specify it.

#### Batch fetching

When a workflow has many registry-backed function groups, all cards are resolved in a maximum of two HTTP calls: one for deployment IDs, one for external IDs. Results are cached (in-process by default, or in-process L1 + shared Redis L2 when `AGENT_CARD_REGISTRY_BACKEND=redis`) and reused until the TTL expires.

On dragent startup, all registry IDs from `workflow.yaml` are **prefetched** in the same batch (enabled by default via `AGENT_CARD_REGISTRY_PREFETCH_ON_STARTUP`). Disable with `AGENT_CARD_REGISTRY_PREFETCH_ON_STARTUP=false` if you need to defer registry access until the first tool call.

#### Registry environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATAROBOT_API_TOKEN` | Yes | DataRobot API token for registry authentication. |
| `DATAROBOT_ENDPOINT` | Yes | DataRobot API base URL, e.g. `https://app.datarobot.com/api/v2`. |
| `AGENT_CARD_REGISTRY_CACHE_TTL` | No | Cache TTL in seconds. Default `86400` (24 h). Set to `0` to disable caching. |
| `AGENT_CARD_REGISTRY_TIMEOUT` | No | HTTP timeout in seconds for registry requests. Default `30`. |
| `AGENT_CARD_REGISTRY_PREFETCH_ON_STARTUP` | No | When `true` (default), batch-fetch all registry-backed agent cards during dragent startup before accepting traffic. Set to `false` to disable. |
| `AGENT_CARD_REGISTRY_MAX_STALENESS_SECONDS` | No | Maximum age in seconds for serving a cached card when the registry is unreachable (stale-if-error). Default `86400` (24 h). |
| `AGENT_CARD_REGISTRY_STALE_IF_ERROR` | No | When `true` (default), return the last-known-good cached card if a registry fetch fails and the entry is within `AGENT_CARD_REGISTRY_MAX_STALENESS_SECONDS`. |
| `AGENT_CARD_REGISTRY_BACKEND` | No | Cache backend: `memory` (default, in-process only) or `redis` (L1 + shared Redis L2). |
| `AGENT_CARD_REGISTRY_REDIS_URL` | When `backend=redis` | Redis connection URL, e.g. `redis://cache.secondary.svc:6379/0`. |
| `AGENT_CARD_REGISTRY_REDIS_PREFIX` | No | Key prefix for Redis entries. Default `dragent:`. |
| `AGENT_CARD_REGISTRY_ON_DUPLICATE` | No | Strategy when multiple cards share the same external ID: `first` keeps the earliest registered card, `last` keeps the most recently registered card, `error` raises an exception. Default: `first`. |

Variables are loaded via `DataRobotAppFrameworkBaseSettings`, which supports env vars, `.env`
files, file secrets, Runtime Parameters, and Pulumi config.

## Configuration reference

### `authenticated_a2a_client` function group

| Field | Default | Description |
|-------|---------|-------------|
| `url` | — | Base URL for direct card fetch. Mutually exclusive with `registry`. |
| `registry` | — | Registry lookup block. Mutually exclusive with `url`. |
| `auth_provider` | `None` | Name of an `authentication` entry for A2A RPC calls. |
| `agent_card_path` | `/.well-known/agent-card.json` | Card path for direct fetch — ignored when using `registry`. |

### `registry` block

Exactly one field must be set.

| Field | Description |
|-------|-------------|
| `deployment_id` | DataRobot deployment ID. |
| `external_id` | External agent catalogue identifier. |

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `RuntimeError: Failed to fetch agent card from …` | Direct-fetch URL unreachable or auth failed. | Verify `url` and `auth_provider` configuration. |
| `AgentCardRegistryError: DataRobot API token is required` | `DATAROBOT_API_TOKEN` not set. | Export the variable or add it to `.env`. |
| `AgentCardRegistryError: DataRobot API endpoint is required` | `DATAROBOT_ENDPOINT` not set. | Export the variable or add it to `.env`. |
| `AgentCardRegistryError: … HTTP 401` | Token invalid or expired. | Regenerate your API token in the DataRobot console. |
| `AgentCardRegistryError: No agent card found …` | Deployment not in the registry. | Confirm the deployment has an A2A agent card published. |
| `ValueError: … 'url' … or 'registry' …, not both` | Both fields set. | Remove one — they are mutually exclusive. |
| Stale card after redeployment | Cache TTL has not expired. | Set `AGENT_CARD_REGISTRY_CACHE_TTL=0` or wait for TTL to elapse. |
