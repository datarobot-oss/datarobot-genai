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

The exact fields inside each block mirror what you would set in env for routing (model name, gateway on/off, deployment ids). Prefer the **same env vars as the e2e tests** unless you need to pin something in YAML for a deployment.

## Linking workflows to an LLM

Any **`workflow`** or **`functions.*`** entry that needs a model sets **`llm_name:`** to the key under **`llms:`** (e.g. `llm_name: datarobot_llm`).
