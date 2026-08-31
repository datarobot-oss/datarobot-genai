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

Every MCP server lives in one variable, `MCP_SERVERS`: a JSON array in which each entry
is a **name** plus **exactly one address**. There is no limit of one server, and no limit
of one server per kind.

```bash
# .env -- python-dotenv reads a quoted value across newlines. Keep it on one line if the
# same file is fed to `docker --env-file`, which does no quote processing.
MCP_SERVERS='[
  {"name": "analytics", "deployment_id": "69331f1f30548f83b668d9dc"},
  {"name": "search",    "workload_id":   "6a72dd6d4417b3136f64fef0"},
  {"name": "docs",      "local_port":    9001},
  {"name": "partner",   "url": "https://partner.example.com/mcp"}
]'
```

| Address         | Where the server is                | What it receives                          |
| --------------- | ---------------------------------- | ----------------------------------------- |
| `deployment_id` | a DataRobot custom model deployment | bearer token, auth context                |
| `workload_id`   | a Workload API container            | bearer token, auth context, `x-datarobot-api-key` |
| `local_port`    | a local process (`local_host` too, for compose) | bearer token, auth context   |
| `url`           | a third-party server                | **no DataRobot credentials**, plus any `headers` you declare |

Setting two addresses on one entry is an error, not a precedence contest. So is a
loopback host under `url`: that would silently send no credentials, which works on a
laptop and fails once deployed. Use `local_port` for a local server.

Deployed, the same variable arrives as a runtime parameter and nothing else changes:

```python
CustomModelRuntimeParameterValueArgs(
    key="MCP_SERVERS", type="string",
    value=pulumi.Output.json_dumps([
        {"name": "analytics", "deployment_id": analytics_mcp.id},
        {"name": "search",    "workload_id":   search_mcp.id},
    ]),
)
```

A `workload_id` costs one lookup: the agent asks the platform where that workload is
served (`GET /api/v2/workloads/<id>/`) and appends `/mcp` to the endpoint it reports, so
the same entry works however your cluster routes workloads. The agent's API token
therefore needs read access to the workload. If the lookup cannot answer, the build fails
naming the server rather than guessing a URL that would be wrong on some clusters.

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

`aresolve_mcp_targets()` resolves every configured server, running the workload lookups
concurrently; pass a list of names to resolve only some of them.

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
working: each resolves as the entry named `default`, so no existing `.env` breaks. Setting
two of the three *remote* ones is now an error rather than a silent precedence.
`MCP_SERVER_PORT` is exempt — it names the port an MCP *server* binds, which the
application templates set unconditionally for their bundled server — and stays a fallback
used only when no remote address is set.

**Declaring `MCP_SERVERS` supersedes all of them wholesale.** They are a fallback for a
configuration that has not adopted the list, not entries merged into one that has, so the
fleet you declare is the fleet you get — a leftover `MCP_SERVER_PORT` or `MCP_DEPLOYMENT_ID`
cannot add a server you did not ask for.

`MCPConfig` is deprecated and no longer reads the environment or builds headers.
`mcp_tools_context` takes an `MCPTarget` from `aresolve_mcp_targets()` or `build_target()`,
and the request-scoped `forwarded_headers` / `authorization_context` are now arguments
rather than fields — as fields on a shared config object they were what made it unsafe to
copy between requests.

## Automated tests

`e2e-tests/dragent_tests/test_mcp.py` exercises MCP tool calls when **`MCP_SERVERS`** declares a server (or one of the single-server variables it supersedes is set) and a tool-capable agent is configured.
