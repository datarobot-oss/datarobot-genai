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

**Lookup by workload ID** (for an agent served by the Workload API runtime, where the card is keyed by workload rather than deployment):

```yaml
function_groups:
  remote_agent:
    _type: authenticated_a2a_client
    registry:
      workload_id: "64a1b2c3d4e5f6a7b8c9d0e2"
    auth_provider: okta_auth
```

> [!NOTE]
> When using the registry the RPC base URL is derived from the card's advertised `url` — you do not need to specify it.

#### Batch fetching

When a workflow has many registry-backed function groups, all cards are resolved in a maximum of three HTTP calls: one per ID kind (deployment, external, workload). ID kinds are never mixed in a single request — `deploymentIds` together with `workloadIds` is rejected with HTTP 400, and either combined with `externalIds` matches nothing. Results are cached in-memory and reused until the TTL expires.

The registry API caps each ID parameter at 20 values, so a longer list of one kind is split into chunks of 20 and issued as consecutive requests; the responses are merged into a single result set before the duplicate strategy is applied.

On dragent startup, all registry IDs from `workflow.yaml` are **prefetched** in the same batch before the server accepts traffic.

While the server is running, registered cards are **refreshed in the background** every 30 minutes. Only entries past the soft cache TTL are re-fetched; failures are logged and existing cache entries are retained. If a registry fetch fails, the last-known-good cached card is served.

#### Registry environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATAROBOT_API_TOKEN` | Yes | DataRobot API token for registry authentication. |
| `DATAROBOT_ENDPOINT` | Yes | DataRobot API base URL, e.g. `https://app.datarobot.com/api/v2`. |
| `AGENT_CARD_REGISTRY_CACHE_TTL` | No | Cache TTL in seconds. Default `86400` (24 h). Set to `0` to disable caching. |
| `AGENT_CARD_REGISTRY_TIMEOUT` | No | HTTP timeout in seconds for registry requests. Default `30`. |
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

Exactly one of the three fields must be set.

| Field | Description |
|-------|-------------|
| `deployment_id` | DataRobot deployment ID. |
| `external_id` | External agent catalogue identifier. |
| `workload_id` | DataRobot workload ID (Workload API runtime). |

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `RuntimeError: Failed to fetch agent card from …` | Direct-fetch URL unreachable or auth failed. | Verify `url` and `auth_provider` configuration. |
| `AgentCardRegistryError: DataRobot API token is required` | `DATAROBOT_API_TOKEN` not set. | Export the variable or add it to `.env`. |
| `AgentCardRegistryError: DataRobot API endpoint is required` | `DATAROBOT_ENDPOINT` not set. | Export the variable or add it to `.env`. |
| `AgentCardRegistryError: … HTTP 401` | Token invalid or expired. | Regenerate your API token in the DataRobot console. |
| `AgentCardRegistryError: No agent card found …` | The deployment or workload is not in the registry. | Confirm the agent is running and has an A2A agent card published. |
| `ValueError: … 'url' … or 'registry' …, not both` | Both fields set. | Remove one — they are mutually exclusive. |
| `ValueError: Specify exactly one of 'deployment_id', 'external_id' or 'workload_id' …` | More than one identifier set inside `registry`. | Keep only the one that identifies the agent. |
| `AgentCardRegistryError: Cannot request 'deploymentIds' and 'workloadIds' in the same … request` | Both ID kinds reached one registry request — the API answers HTTP 400. | Internal invariant; report it, as each kind is meant to be fetched separately. |
| Stale card after redeployment | Cache TTL has not expired. | Set `AGENT_CARD_REGISTRY_CACHE_TTL=0` or wait for TTL to elapse. |
