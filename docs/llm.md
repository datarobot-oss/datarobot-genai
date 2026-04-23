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

# LLM configuration

Whether you use **`workflow.yaml`** (`llms:` blocks) or Python agents, the same **routing idea** applies: DataRobot **LLM Gateway**, a **model deployment**, **NIM**, or an **external** provider via LiteLLM.

In Python, each integration exposes the same helpers from its `llm` submodule&mdash;swap the import path only:

| Integration | Import from |
|---|---|
| LangGraph | `datarobot_genai.langgraph.llm` |
| LlamaIndex | `datarobot_genai.llama_index.llm` |
| CrewAI | `datarobot_genai.crewai.llm` |

Values are read from the process environment (and `.env` in the working directory when your runner loads it). For DataRobot-hosted routes you typically set `DATAROBOT_ENDPOINT` and `DATAROBOT_API_TOKEN` (see [Example.ipynb](Example.ipynb)).

## DataRobot LLM Gateway

Use **`get_datarobot_gateway_llm()`** when you always want the gateway regardless of `USE_DATAROBOT_LLM_GATEWAY`. Calls go through the gateway URL derived from `DATAROBOT_ENDPOINT`; use `DATAROBOT_API_TOKEN` as the API key.

**Pick a real gateway model id.** Do **not** rely on the default model name for gateway use: `get_datarobot_gateway_llm()` falls back to `LLM_DEFAULT_MODEL` which might be empty in your env.

1. Install `dr` CLI plugin that lists models your account can use ([get-datarobot-llms](https://github.com/carsongee/get-datarobot-llms)):

   ```bash
   uv tool install git+https://github.com/carsongee/get-datarobot-llms.git
   ```

2. With `DATAROBOT_ENDPOINT` and `DATAROBOT_API_TOKEN` set, list ids you can pass as `datarobot-model`:

   ```bash
   dr get-llms
   ```

3. Pass a chosen id explicitly (LangGraph example):

   ```python
   from datarobot_genai.langgraph.llm import get_datarobot_gateway_llm

   llm = get_datarobot_gateway_llm("datarobot/azure/gpt-5-nano-2025-08-07")
   ```

## LLM deployment

Use **`get_datarobot_deployment_llm(deployment_id, ...)`** to send chat completions to a specific DataRobot deployment. The client uses `DATAROBOT_ENDPOINT` and `DATAROBOT_API_TOKEN` to build `{endpoint}/deployments/{deployment_id}/chat/completions`. You may pass `model_name` and `parameters` like other helpers.

```python
from datarobot_genai.langgraph.llm import get_datarobot_deployment_llm

llm = get_datarobot_deployment_llm("your-deployment-id")
```

## NIM deployment

Use **`get_datarobot_nim_llm(nim_deployment_id, ...)`** for an NVIDIA NIM deployment hosted like a DataRobot deployment (same URL shape as **`get_datarobot_deployment_llm`**).

```python
from datarobot_genai.langgraph.llm import get_datarobot_nim_llm

llm = get_datarobot_nim_llm("your-nim-deployment-id", model_name="optional-model-id")
```

## External providers (LiteLLM)

Use **`get_external_llm()`** when you want LiteLLM to call **external** providers (for example OpenAI) using **their** environment variables (for example `OPENAI_API_KEY`). Model names should **not** rely on the `datarobot/` prefix in this mode.

```python
from datarobot_genai.langgraph.llm import get_external_llm

llm = get_external_llm("gpt-4o-mini")
```

To reach this route via **`get_llm()`**, turn the gateway off and unset both `LLM_DEPLOYMENT_ID` and `NIM_DEPLOYMENT_ID` (see **get_llm()** below).

## `get_llm()` (environment-driven routing)

Prefer the explicit **`get_*`** helpers above when you know the route. Use **`get_llm()`** as a single entry point when you want one code path and you steer behavior entirely with configuration. It inspects settings and delegates to the same underlying helpers as those sections.

Routing order:

1. **Gateway** if `USE_DATAROBOT_LLM_GATEWAY=true` (default).
2. Else **deployment** if `LLM_DEPLOYMENT_ID` is set.
3. Else **NIM** if `NIM_DEPLOYMENT_ID` is set.
4. Else **external** (LiteLLM using provider-specific environment variables).

If both `LLM_DEPLOYMENT_ID` and `NIM_DEPLOYMENT_ID` are set with the gateway off, **deployment wins**.

These variables control **`get_llm()`** specifically:

| Variable | Role |
|---|---|
| `USE_DATAROBOT_LLM_GATEWAY` | When `true` (default), use the **DataRobot LLM Gateway**. |
| `LLM_DEPLOYMENT_ID` | When the gateway is off, use this **LLM deployment** chat endpoint. |
| `NIM_DEPLOYMENT_ID` | When the gateway is off and no LLM deployment id is set, use this **NIM** deployment. |
| `LLM_DEFAULT_MODEL` | Default model id when you omit `model_name` on **`get_llm()`** |

Example (LangGraph; adjust the import for LlamaIndex or CrewAI):

```python
from datarobot_genai.langgraph.llm import get_llm

llm = get_llm()  # optional: model_name="...", parameters={...}, streaming=True
```

# In `workflow.yaml`

The e2e samples usually declare one named LLM and reference it from `workflow`:

```yaml
llms:
  datarobot_llm:
    _type: datarobot-llm-component
```

That component follows the same four outcomes using fields on the block (gateway flag, deployment ids, etc.) and the environment.
