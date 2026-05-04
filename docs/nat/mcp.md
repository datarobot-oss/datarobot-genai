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

## `function_groups` — attach an MCP server

```yaml
function_groups:
  mcp_tools:
    _type: datarobot_mcp_client
```

**`mcp_tools`** is an arbitrary label. DRAgent resolves the MCP server URL, transport, and default auth from **deployment settings and environment** (MCP deployment id, external URL, etc.—what you configure for your app). You do not paste secrets into this block in the example; runtime headers are merged when the config is loaded for a request.

## `authentication` — MCP auth block

```yaml
authentication:
  datarobot_mcp_auth:
    _type: datarobot_mcp_auth
```

This ties MCP HTTP calls to DataRobot-style auth. Request headers (API token, identity context) are applied when loading the workflow so MCP and LLM calls stay consistent.

## `workflow.tool_names` — expose MCP to the orchestrator

The orchestrator only sees tools you list. Include the **group name** (`mcp_tools` in the example), not individual MCP tool names:

```yaml
workflow:
  tool_names:
    - planner
    - writer
    - mcp_tools
    - generate_objectid
```

MCP tools often show up in traces with a prefix (e.g. `mcp_tools__...`); that is normal.

## Custom Python tools vs MCP

**`functions:`** defines one-off tools (e.g. `generate_objectid`) registered from [`register.py`](../../e2e-tests/dragent/nat/register.py). **MCP** brings a whole group from a server. Both appear in **`tool_names`** side by side.
