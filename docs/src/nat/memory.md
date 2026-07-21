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

# Mem0 memory in NAT workflows

`datarobot-genai` includes a NAT `MemoryEditor` provider that adapts the DataRobot Mem0 client for NAT's `auto_memory_agent`. This avoids the upstream `nvidia-nat-mem0ai` plugin while still using NAT's standard `memory:` configuration.

Install the NAT extra (includes `mem0ai`):

```bash
pip install "datarobot-genai[nat]"
```

Set the Mem0 API key at runtime:

```bash
export MEM0_API_KEY=...
```

## Configure the memory provider

```yaml
memory:
  mem0_memory:
    _type: dr_mem0_memory
    # Optional explicit override; defaults from MEM0_API_KEY runtime settings.
    # api_key: ${MEM0_API_KEY}
    # Optional Mem0 organization/project routing:
    # host: https://api.mem0.ai
    # org_id: ...
    # project_id: ...
```

**`mem0_memory`** is the local name you reference from the workflow. **`dr_mem0_memory`** is the DataRobot provider type registered by this package. The provider reads `MEM0_API_KEY` through DataRobot app framework settings by default; use `api_key` only when you need an explicit workflow-level override.

## Wrap an agent with automatic memory

```yaml
middleware:
  datarobot_dragent_normalization:
    _type: datarobot_dragent_normalization

functions:
  nat_agent:
    _type: per_user_tool_calling_agent
    llm_name: datarobot_llm
    # Required for inner agents: NAT middleware is per-function and opt-in — it is
    # NOT inherited from the parent workflow. The memory wrapper calls the inner
    # agent's stream directly, so the normalization middleware must be declared here
    # for the inner agent to yield DRAgentEventResponse.
    middleware:
      - datarobot_dragent_normalization
    tool_names:
      - planner
      - writer

workflow:
  _type: auto_memory_agent
  inner_agent_name: nat_agent
  memory_name: mem0_memory
  llm_name: datarobot_llm
  description: "Agent with automatic memory capture and retrieval."
```

`per_user_tool_calling_agent` emits native NAT output (a `str` for single responses and
`ChatResponseChunk` for streaming). The `datarobot_dragent_normalization` middleware converts
that into DRAgent's canonical `DRAgentEventResponse`. When a `per_user_tool_calling_agent` is the
top-level workflow (not wrapped by a memory agent), declare the middleware in the `workflow:`
block instead — as the last (innermost) entry so it runs before any moderation or telemetry
middleware.

NAT supplies the runtime `user_id` to the memory backend from the session user manager, `X-User-ID` header, or its local fallback. The provider forwards that `user_id` into Mem0 v2 search filters so memories remain isolated per user.

## Optional search and add parameters

Parameters under `search_params` are passed to `MemoryEditor.search()`, and parameters under `add_params` are passed to `MemoryEditor.add_items()`:

```yaml
workflow:
  _type: auto_memory_agent
  inner_agent_name: nat_agent
  memory_name: mem0_memory
  llm_name: datarobot_llm
  search_params:
    top_k: 5
  add_params:
    agent_id: blog_agent
```

The provider also supports `host`, `org_id`, and `project_id` on the memory config for Mem0 deployment routing.
