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

# LLMs in `workflow.yaml`

This matches what appears under **`llms:`** in the examples, e.g. [`e2e-tests/dragent/nat/workflow.yaml`](../../e2e-tests/dragent/nat/workflow.yaml).

## What you usually declare

```yaml
llms:
  datarobot_llm:
    _type: datarobot-llm-component
```

**`datarobot-llm-component`** is the flexible option: it follows the same **gateway / deployment / NIM / external** routing as your **environment variables** (see [LLM configuration (shared)](../llm.md)). You can name the key anything (`datarobot_llm` is just the label used elsewhere as **`llm_name`**).

## Other `_type` values you may see

| `_type` | When it appears |
|---|---|
| **`datarobot-llm-gateway`** | Gateway only; no per-deployment id in the block. |
| **`datarobot-llm-deployment`** | Fixed LLM deployment; often includes deployment id and optional headers in YAML. |
| **`datarobot-nim`** | NIM deployment on DataRobot. |
| **`datarobot-litellm`** | External LiteLLM providers; provider keys still come from the environment. |
| **`datarobot-llm-router`** | Primary + fallback LLMs with automatic failover via LiteLLM Router; includes `primary`, `fallbacks`, and optional tuning field `num_retries`. |

The exact fields inside each block mirror what you would set in env for routing (model name, gateway on/off, deployment ids). Prefer the **same env vars as the e2e tests** unless you need to pin something in YAML for a deployment.

## Additional model parameters (`llm_additional_model_params`)

Add **`llm_additional_model_params`** to any supported LLM block to pass **extra LiteLLM keyword arguments** through to the framework client (LangChain, CrewAI, or LlamaIndex). The map is merged on top of the block’s other fields (`model`, `temperature`, `api_key`, and so on).

If the block does not set this field, NAT uses the process default from the **`LLM_ADDITIONAL_MODEL_PARAMS`** environment variable (JSON string). See [LLM configuration (shared)](../llm.md#additional-model-parameters-llm_additional_model_params).

```yaml
llms:
  datarobot_llm:
    _type: datarobot-llm-component
    llm_additional_model_params:
      max_tokens: 4096
```

Gateway extended thinking (body field) example:

```yaml
llms:
  datarobot_llm:
    _type: datarobot-llm-gateway
    model: datarobot/bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0
    llm_additional_model_params:
      extra_body:
        thinking:
          type: enabled
          budget_tokens: 1024
```

For **`datarobot-llm-router`**, put the map on each **`primary`** / **`fallbacks`** entry (same shape as other LLM config fields on those nodes).

| Source | Format |
|---|---|
| `LLM_ADDITIONAL_MODEL_PARAMS` env | JSON object (required for env; not Python `repr`) |
| `llm_additional_model_params` in YAML | YAML mapping |

## Passing extra kwargs with `extra_body`

Add **`extra_body`** to any LLM block to forward arbitrary key-value pairs in the request body. Works with every `_type`. This is a **dedicated** top-level field; LangGraph clients move it into `model_kwargs.extra_body` automatically.

For other LiteLLM options (or nested body shapes), use **`llm_additional_model_params`** instead, or set `extra_body` inside that map (see above).

```yaml
llms:
  datarobot_llm:
    _type: datarobot-llm-component
    extra_body:
      mock_response: "this is a mock response"
```

## Linking workflows to an LLM

Any **`workflow`** or **`functions.*`** entry that needs a model sets **`llm_name:`** to the key under **`llms:`** (e.g. `llm_name: datarobot_llm`).
