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

# LLM provider fallback (router)

Use this when you want one primary LLM provider/model and one-or-more fallback providers/models in case the primary fails.

This is the native `datarobot-genai` router path backed by `litellm.Router`.

## Agent-component change checklist

When updating an existing agent component, do these changes in order:

1. In `workflow.yaml`, change the LLM `_type` from `datarobot-llm-component` to `datarobot-llm-router`.
2. Add a `primary:` config.
3. Add at least one item in `fallbacks:`.
4. Keep `workflow.llm_name` pointing to the same LLM key.
5. Do not change your `myagent.py` graph/crew/workflow code just for fallback support.

## Required `workflow.yaml` change

### Before

```yaml
llms:
  datarobot_llm:
    _type: datarobot-llm-component
```

### After

```yaml
llms:
  datarobot_llm:
    _type: datarobot-llm-router
    primary:
      use_datarobot_llm_gateway: true
      llm_default_model: azure/gpt-5-mini-2025-08-07
    fallbacks:
      - use_datarobot_llm_gateway: true
        llm_default_model: anthropic/claude-opus-4-20250514
    num_retries: 3
```

`workflow.llm_name` stays the same:

```yaml
workflow:
  llm_name: datarobot_llm
```

## `primary` and `fallbacks` fields

Each `primary`/`fallbacks[*]` entry uses the same core LLM config shape.

| Field | Meaning |
|---|---|
| `use_datarobot_llm_gateway` | `true` routes via DataRobot LLM Gateway; `false` uses deployment/NIM/external based on ids. |
| `llm_default_model` | Model id for that entry (for example `azure/gpt-5-mini-2025-08-07`). |
| `llm_deployment_id` | DataRobot deployment id for deployment routing. |
| `nim_deployment_id` | DataRobot deployment id for NIM routing. |
| `datarobot_endpoint` | Optional per-entry endpoint override (usually from env). |
| `datarobot_api_token` | Optional per-entry token override (usually from env). |

Router-level tuning fields:

| Field | Meaning |
|---|---|
| `num_retries` | Number of retries before the router surfaces a failure. |

## Minimal drop-in example for an existing component

Use this pattern in any component `workflow.yaml` (LangGraph, CrewAI, LlamaIndex, NAT):

```yaml
llms:
  datarobot_llm:
    _type: datarobot-llm-router
    primary:
      use_datarobot_llm_gateway: true
      llm_default_model: "your-primary-model-id"
    fallbacks:
      - use_datarobot_llm_gateway: true
        llm_default_model: "your-fallback-model-id"
    num_retries: 1

workflow:
  llm_name: datarobot_llm
```

Reference examples:

- `e2e-tests/dragent/langgraph/workflow-router-fallback-used.yaml`
- `e2e-tests/dragent/crewai/workflow-router-fallback-used.yaml`
- `e2e-tests/dragent/llamaindex/workflow-router-fallback-used.yaml`

## Python API (non-NAT path)

If you create LLM objects directly in Python, use `get_router_llm(primary, fallbacks, router_settings)`.

Import paths by framework:

- LangGraph: `datarobot_genai.langgraph.llm.get_router_llm`
- CrewAI: `datarobot_genai.crewai.llm.get_router_llm`
- LlamaIndex: `datarobot_genai.llama_index.llm.get_router_llm`

Example:

```python
from datarobot_genai.core.config import LLMConfig
from datarobot_genai.langgraph.llm import get_router_llm

primary = LLMConfig(
    use_datarobot_llm_gateway=True,
    llm_default_model="azure/gpt-5-mini-2025-08-07",
)
fallbacks = [
    LLMConfig(
        use_datarobot_llm_gateway=True,
        llm_default_model="anthropic/claude-opus-4-20250514",
    )
]

llm = get_router_llm(primary, fallbacks, {"num_retries": 3})
```

## What does not need to change

- Your agent graph/crew/workflow-building code in `myagent.py`
- Tool definitions and MCP wiring
- `workflow._type` (`langgraph_agent`, `crewai_agent`, `llamaindex_agent`, etc.)

Only the LLM provider block changes unless you are also redesigning your agent behavior.
