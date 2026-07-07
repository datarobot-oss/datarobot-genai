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

# Tools and MCP (LlamaIndex sample)

When the platform injects tools (including MCP), they are **added to** each workflow agent’s existing tools in the DataRobot integration so your hand-authored tools stay available.

Configure MCP deployment / URLs in your environment; the minimal **`workflow.yaml`** in the LlamaIndex folder does not show **`function_groups`**—see [nat/mcp.md](../nat/mcp.md) for the YAML-heavy pattern.
