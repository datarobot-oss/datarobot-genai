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

# MCP tools in NAT workflows

This matches the **`function_groups`**, **`authentication`**, and **`workflow.tool_names`** sections in [`e2e-tests/dragent/nat/workflow.yaml`](../../e2e-tests/dragent/nat/workflow.yaml).

## `function_groups` — attach MCP servers

One block per server. `datarobot_mcp_client` is NAT's `mcp_client` with **one added
field** — `server.name` — and one relaxed rule: NAT requires `server.url`, here it is
optional. Everything else on the block (`include`, `exclude`, `tool_overrides`, the
timeouts, the `reconnect_*` family, `session_aware_tools`, `max_sessions`,
`session_idle_timeout`) is NAT's and behaves exactly as
[NAT documents it](https://docs.nvidia.com/nemo/agent-toolkit/latest/build-workflows/mcp-client.html).

### Three ways to say where the server is

**By name**, with the address in the environment. The same `workflow.yaml` then works on
a laptop and in production, and an address a deploy creates resolves once the deploy has
run:

```yaml
function_groups:
  analytics:
    _type: datarobot_mcp_client
    server:
      name: analytics
      auth_provider: datarobot_mcp_auth
  docs:
    _type: datarobot_mcp_client
    server:
      name: docs
      auth_provider: datarobot_mcp_auth
```

```bash
# one small group of variables per server -- no packed JSON, nothing to edit in Python
analytics_mcp_deployment_id=69331f1f30548f83b668d9dc
docs_mcp_local_port=9001
```

Each server sets exactly one address: `deployment_id`, `workload_id`, `local_port` or
`url`. The address fields also accept a short form without `_mcp` (`docs_local_port`).

**Inline**, exactly as NAT documents it — right for an address that is genuinely the
same in every environment:

```yaml
function_groups:
  partner:
    _type: datarobot_mcp_client
    server:
      transport: streamable-http
      url: "https://partner.example.com/mcp"
      auth_provider: none
```

**A local stdio process**, also as NAT documents it:

```yaml
function_groups:
  mcp_time:
    _type: datarobot_mcp_client
    server:
      transport: stdio
      command: "python"
      args: ["-m", "mcp_server_time"]
```

A stdio server is a child process: it has no URL and no identity, so `auth_provider`
does not apply to it (NAT supports that for `streamable-http` only).

### One source per server

A server's address comes from its environment group **or** from an inline block, never
both — and the same goes for its identity, which is either `server.auth_provider` or
`<name>_mcp_auth_provider`. Declaring either in two places raises at build, naming both
locations. It is an error rather than a precedence rule because there is no correct
answer to pick, and picking one silently is the failure this design removes.

`server.name` defaults to `default`, so a block that omits it keeps working with the
pre-existing single-server variables (`MCP_DEPLOYMENT_ID`, `MCP_WORKLOAD_ID`,
`MCP_SERVER_PORT`, `EXTERNAL_MCP_URL`).

Two blocks may name one server: the same server reached under two identities is two
blocks with two `auth_provider` values. Two *definitions* of one name is an error.

A server that cannot be resolved fails the build, naming the server. It does not degrade
into an agent with no tools.

## `authentication` — MCP auth block

```yaml
authentication:
  datarobot_mcp_auth:
    _type: datarobot_mcp_auth
```

This ties MCP HTTP calls to DataRobot-style auth. Per-request headers (API token,
identity context) are read from NAT request context at runtime so MCP and LLM calls stay
consistent.

NAT hands **one instance of this provider to every block that names it**, which is
correct here because it produces the same credentials for every server: forwarded
headers, `Authorization: Bearer`, `x-datarobot-api-key`, and the authorization context.
The API-key header is sent to every DataRobot-hosted server rather than to workloads
only — the Workload API gateway needs it and a deployment or local process ignores it —
so nothing varies per server and the provider needs no per-server state. A third-party
server never reaches this provider at all, because it names `auth_provider: none`.

`headers:` on this block is merged **last** and therefore wins. It is the only way to
attach a static header to a DataRobot-hosted server.

## `workflow.tool_names` — expose MCP to the orchestrator

The orchestrator only sees tools you list. Include the **group name** — one per server —
not individual MCP tool names:

```yaml
workflow:
  tool_names:
    - planner
    - writer
    - analytics
    - docs
    - generate_objectid
```

MCP tools show up prefixed with their group (`analytics__list_deployments`); that is
normal, and it is what keeps two servers' identically-named tools apart.

## Custom Python tools vs MCP

**`functions:`** defines one-off tools (e.g. `generate_objectid`) registered from [`register.py`](../../e2e-tests/dragent/nat/register.py). **MCP** brings a whole group from a server. Both appear in **`tool_names`** side by side.
