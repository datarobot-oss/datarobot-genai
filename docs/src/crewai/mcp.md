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

# Tools and MCP (CrewAI sample)

When tools are injected at runtime, the DataRobot-wrapped crew agent **keeps your original per-agent tools** and **appends** injected ones (so MCP and your task-specific tools can coexist). The minimal e2e YAML does not declare MCP groups; use [nat/mcp.md](../nat/mcp.md) for a full **`function_groups`** example.
