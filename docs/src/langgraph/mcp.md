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

# Tools and MCP (LangGraph sample)

## What you see when MCP is enabled

DataRobot can attach **extra tools** (including from an MCP deployment) when the agent runs. In the graph sample, those appear **alongside** tools you define in Python. If you list only your own tools and ignore the injected list, **MCP tools will not be available** to the model.

Practical rule: **merge** platform tools with yours wherever you bind tools to the graph (the e2e `myagent.py` shows the intended pattern).

## Declaring the servers an agent can reach

Each server is one small group of flat variables, prefixed by the server's name. There is
no limit of one server, and no limit of one server per kind. Nothing in `config.py` names
them: adding a server is two lines in `.env`.

```bash
analytics_mcp_deployment_id=69331f1f30548f83b668d9dc
search_mcp_workload_id=6a72dd6d4417b3136f64fef0
docs_mcp_local_port=9001
docs_mcp_local_host=mcp-docs          # for docker compose
externally_deployed_mcp_url=https://mcp.example.com/mcp
```

The address fields also accept a short form without `_mcp` — `docs_local_port`.

| Address field | Where the server is |
| --- | --- |
| `<name>_mcp_deployment_id` | a DataRobot custom model deployment |
| `<name>_mcp_workload_id` | a Workload API container |
| `<name>_mcp_local_port` (+ `_local_host`) | a local process |
| `<name>_mcp_url` | a server addressed verbatim |

**How it is addressed no longer decides how it is authenticated.** That is
`<name>_mcp_auth_provider`, naming an `authentication:` entry or `none`:

| `auth_provider` | What the server receives |
| --- | --- |
| `datarobot_mcp_auth` *(default for the DataRobot-hosted kinds)* | forwarded headers, `Authorization: Bearer`, `x-datarobot-api-key`, auth context |
| an `okta_cross_app_access` entry | an exchanged per-user token — and **no** `DATAROBOT_API_TOKEN` is required to resolve the server |
| `none` *(default for `url`)* | only the static `<name>_mcp_headers` |

`x-datarobot-api-key` goes to every DataRobot-hosted server, not workloads only: the
Workload API gateway needs it and a deployment or local process ignores it. Because
nothing then varies per server, one auth provider instance serves the whole fleet.

An externally hosted server gets no DataRobot identity, so if it needs a credential of
its own that credential goes in `<name>_mcp_headers`, under whatever header name the
server expects:

```bash
vendor_mcp_url=https://vendor.example.com/mcp
vendor_mcp_headers={"Authorization": "Bearer the-vendors-own-token"}
```

Setting two addresses on one server is an error, not a precedence contest. So is a
loopback host under `url`: that would silently send no credentials, which works on a
laptop and fails once deployed. Use `local_port` for a local server. Sending DataRobot
credentials to a `url` on a host other than `DATAROBOT_ENDPOINT` also raises, with no
override: say `<name>_mcp_auth_provider=none`, or address the server by
`<name>_mcp_workload_id` / `<name>_mcp_deployment_id` so its URL is derived rather than
asserted.

Deployed, the same variables arrive as runtime parameters and nothing else changes:

```python
[
    CustomModelRuntimeParameterValueArgs(
        key="ANALYTICS_MCP_DEPLOYMENT_ID", type="string", value=analytics_mcp.id),
    CustomModelRuntimeParameterValueArgs(
        key="SEARCH_MCP_WORKLOAD_ID", type="string", value=search_mcp.id),
]
```

Resolution performs no network call: every kind composes its URL, a workload included.

## Attaching the tools

Resolve the fleet and enter one context per server. `AsyncExitStack` keeps them all open
for the life of the agent:

```python
from contextlib import AsyncExitStack

from datarobot_genai.core.mcp import aresolve_mcp_targets
from datarobot_genai.langgraph.mcp import mcp_tools_context

async with AsyncExitStack() as stack:
    tools = []
    for target in await aresolve_mcp_targets():
        tools += await stack.enter_async_context(
            mcp_tools_context(
                target,
                forwarded=forwarded_headers,
                auth_context=authorization_context,
            )
        )
    agent = MyAgent(llm=llm, tools=tools + my_own_tools)
```

`aresolve_mcp_targets()` resolves every configured server; pass a list of names to
resolve only some of them. `mcp_tools_context` also takes `extra=` — headers merged last,
which is how a caller that runs its own token exchange presents the exchanged token.

Each server's tools are namespaced `<server name>__<tool>`, matching how NAT names
function-group tools. That is what lets two servers each exposing `search` coexist —
without it one would silently shadow the other. Pass `prefix=""` to keep raw names, which
is safe only with a single server.

A server that is configured but unreachable **raises**. Yielding an empty tool list would
make it indistinguishable from a server that was never configured, which is a legitimate
state. Pass `strict=False` for the older degrade-quietly behaviour.

## What you see when MCP is enabled

DataRobot can attach **extra tools** when the agent runs. In the graph sample, those
appear **alongside** tools you define in Python. If you list only your own tools and
ignore the resolved list, **MCP tools will not be available** to the model. Merge them
wherever you bind tools to the graph (the e2e `myagent.py` shows the intended pattern).

The NAT workflow example declares servers in YAML instead; see [nat/mcp.md](../nat/mcp.md).

## Migrating from the single-server variables

`MCP_DEPLOYMENT_ID`, `MCP_WORKLOAD_ID`, `MCP_SERVER_PORT` and `EXTERNAL_MCP_URL` keep
working: each resolves as the server named `default`, so no existing `.env` breaks. Setting
two of the three *remote* ones is now an error rather than a silent precedence.
`MCP_SERVER_PORT` is exempt — it names the port an MCP *server* binds, which the
application templates set unconditionally for their bundled server — and stays a fallback
used only when no remote address is set.

**Declaring any per-server variables supersedes all of them wholesale.** They are a
fallback for a configuration that has not adopted the per-server form, not entries merged
into one that has, so the fleet you declare is the fleet you get — a leftover
`MCP_SERVER_PORT` cannot add a server you did not ask for.

`MCPConfig` is deprecated and no longer reads the environment or builds headers.
`mcp_tools_context` takes an `MCPTarget` from `aresolve_mcp_targets()` or `build_target()`,
and the request-scoped `forwarded_headers` / `authorization_context` are now arguments
rather than fields — as fields on a shared config object they were what made it unsafe to
copy between requests.

## Automated tests

`e2e-tests/dragent_tests/test_mcp.py` exercises MCP tool calls when per-server variables declare a server (or one of the single-server variables they supersede is set) and a tool-capable agent is configured.
