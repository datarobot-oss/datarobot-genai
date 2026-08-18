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

## Configuration outside the repo

MCP server URL and credentials come from **your deployment / environment**, not from the LangGraph `workflow.yaml` in the minimal sample. The NAT workflow example adds **`function_groups`** in YAML instead; see [nat/mcp.md](../nat/mcp.md). With `MCP_WORKLOAD_ID`, the agent asks the platform where that workload is served (`GET /api/v2/workloads/<id>/`) and appends `/mcp` to the endpoint it reports, so the same variable works whether your cluster routes workloads through different API Gateway. The agent's API token therefore needs read access to the workload. If the lookup cannot answer, the agent runs without MCP tools and logs a warning. Then, no URL is guessed, because a composed one would be wrong on some clusters. Use `EXTERNAL_MCP_URL` to address a workload directly instead.

## Automated tests

`e2e-tests/dragent_tests/test_mcp.py` exercises MCP tool calls when either **`MCP_DEPLOYMENT_ID`** or **`MCP_WORKLOAD_ID`** is set and a tool-capable agent is configured.
