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

# The workflow file (agent, tools, runner)

This describes **`workflow.yaml`** as users see it in the NAT example: [`e2e-tests/dragent/nat/workflow.yaml`](../../e2e-tests/dragent/nat/workflow.yaml).

## `llms` — name the models DRAgent can use

You declare one or more entries; each has an **`_type`** (see [llm.md](llm.md)). The example names a single LLM `datarobot_llm` and points the workflow at it with **`llm_name`**.

## `functions` — tools implemented as NAT functions

Each key under **`functions:`** is a **tool name** the orchestrator can call. In the example you will see:

- **`chat_completion`** sub-workflows (planner, writer): each has its own `system_prompt`, `description`, and **`llm_name`** referencing `datarobot_llm`.
- A custom tool **`generate_objectid`**: `_type` matches a tool registered from Python (see [`register.py`](../../e2e-tests/dragent/nat/register.py) in the same folder).

Those names are what you list under **`workflow.tool_names`**.

## `function_groups` — bundled tools (MCP)

**`mcp_tools`** in the example is a **group** (`_type: datarobot_mcp_client`), not a single function. It expands to the MCP tools exposed by your deployment. You still add **`mcp_tools`** to **`tool_names`** so the orchestrator may call them. Details: [mcp.md](mcp.md).

## `authentication` — credentials MCP calls should use

The example defines **`datarobot_mcp_auth`** so MCP requests carry the same kind of auth as the rest of the DataRobot stack. See [mcp.md](mcp.md).

## `workflow` — the top-level runner

This block picks **which agent pattern** runs and **which tools** are in play.

| Field in the example | What it means |
|---|---|
| **`_type: per_user_tool_calling_agent`** | NAT/DRAgent tool-calling agent: one LLM orchestrates calls to the listed tools. |
| **`llm_name: datarobot_llm`** | Uses the LLM defined under **`llms`**. |
| **`tool_names`** | Ordered list of tools/groups the model may invoke: here `planner`, `writer`, `mcp_tools`, `generate_objectid`. |
| **`return_direct`** | Tools whose output should be returned to the user as-is (here `writer`). |
| **`system_prompt`** | Instructions for the orchestrator (how to chain planner → writer, when to use MCP, etc.). |
| **`verbose`** | Extra logging from the runner. |

Other samples in the repo use different **`workflow._type`** values for LangGraph, LlamaIndex, CrewAI, or a minimal base agent—see those folders’ READMEs; the **`llms:`** + **`general:`** pattern is the same idea.

## Optional Python beside the YAML

[`register.py`](../../e2e-tests/dragent/nat/register.py) only registers extra tools so names like `generate_objectid` resolve. You do not duplicate the graph in code—the **declared structure is the YAML**.

---

**Legacy note:** a Python-only path can load this YAML without DRAgent. Prefer **`nat dragent run`** / **`nat dragent serve`** and the file above; that path is what we document and extend.
