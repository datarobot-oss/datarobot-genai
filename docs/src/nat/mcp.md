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

One block per server. Each names a configured server and the identity to reach it as. The
**address never appears here**: it comes from `MCP_SERVERS`, so the same `workflow.yaml`
works on a laptop and in production, and an address that a deploy creates resolves once
the deploy has run.

```yaml
function_groups:
  analytics:
    _type: datarobot_mcp_client
    server: { name: analytics, auth_provider: datarobot_mcp_auth }
  docs:
    _type: datarobot_mcp_client
    server: { name: docs, auth_provider: datarobot_mcp_auth }
```

The block key (`analytics`) is the tool namespace; `server.name` is which configured
server it connects to. `name` defaults to `default`, so a block that omits it keeps
working with the single-server variables.

Declare the servers themselves in one variable:

```bash
MCP_SERVERS='[
  {"name": "analytics", "deployment_id": "69331f1f30548f83b668d9dc"},
  {"name": "docs",      "local_port": 9001}
]'
```

Each entry sets exactly one address — `deployment_id`, `workload_id`, `local_port` or
`url` — and that choice decides both how the server is found and what credentials it
receives. A third-party `url` receives no DataRobot credentials at all. See
[langgraph/mcp.md](../langgraph/mcp.md) for the full table.

Two blocks may name one server: the same server reached under two identities is two
blocks with two `auth_provider` values. Two *entries* sharing a name is an error.

Setting an address in the block itself — `url`, `transport`, `custom_headers` — raises,
naming `MCP_SERVERS` as where it belongs. Those fields are inherited from NAT's base
config and this client ignores them; a field that is silently ignored is worse than one
that raises.

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

NAT hands **one instance of this provider to every block that names it**, so the block
being called supplies its own resolved server with each request. Which credentials a
server receives depends on its kind — only a workload gets `x-datarobot-api-key`, and a
third-party server gets none — so a fleet cannot share one globally resolved answer.

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
