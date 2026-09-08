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

## Where the servers are configured

`workflow.yaml` says **what** the agent may reach — one `function_groups` entry per server, listed in `tool_names`. The **address** is not in the YAML: it comes from that server's `<name>_mcp_*` variables in your deployment or environment, so the same file runs locally and deployed. `server.name` is the join. See **MCP servers** in the agent application template's docs for the full setup, and [nat/mcp.md](../nat/mcp.md) for what this library adds to NAT.


## Getting the tools

Let NAT build them, **once**, outside your per-request handler:

```python
tools = await builder.get_tools(
    tool_names=config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN
)
```

NAT keeps those clients connected and the auth provider recomputes credentials per request.

For code that does not use NAT, `mcp_tools_context(target, ...)` yields the LangChain tools for one resolved server. It bakes the headers in at connect time, so enter it at startup rather than inside a response function — doing the latter reconnects to every server on every prompt.

## Automated tests

`e2e-tests/dragent_tests/test_mcp.py` exercises MCP tool calls when either **`MCP_DEPLOYMENT_ID`** or **`MCP_WORKLOAD_ID`** is set and a tool-capable agent is configured.
