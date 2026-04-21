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

# LLM configuration (what the examples assume)

Whether you use **`workflow.yaml`** (`llms:` blocks) or Python agents, the same **routing idea** applies: DataRobot **LLM Gateway**, a **model deployment**, **NIM**, or an **external** provider via LiteLLM. You steer that with **environment variables** (and optional fields inside YAML for deployment-specific headers).

## Environment variables

Values are read from the process environment (and `.env` in the working directory when your runner loads it). Typical names:

| Variable | Role |
|---|---|
| `DATAROBOT_ENDPOINT` | Base API URL (default `https://app.datarobot.com/api/v2`). Used to build gateway and deployment URLs. |
| `DATAROBOT_API_TOKEN` | Token for DataRobot-hosted LLM routes. |
| `USE_DATAROBOT_LLM_GATEWAY` | When `true` (default), use the **DataRobot LLM Gateway**. |
| `LLM_DEPLOYMENT_ID` | When the gateway is off, use this **LLM deployment** chat endpoint. |
| `NIM_DEPLOYMENT_ID` | When the gateway is off and no LLM deployment id, use this **NIM** deployment. |
| `LLM_DEFAULT_MODEL` | Default model id (often `datarobot-deployed-llm`). |
| `DATAROBOT_GENAI_MAX_HISTORY_MESSAGES` | How many prior chat turns to include in history summaries. |

## Routing outcomes (four cases)

What you get depends on flags and ids:

1. **Gateway** — `USE_DATAROBOT_LLM_GATEWAY=true` (default): calls go through the DataRobot LLM Gateway derived from `DATAROBOT_ENDPOINT`.
2. **Deployment** — Gateway off **and** `LLM_DEPLOYMENT_ID` set: calls go to that deployment’s chat completions URL.
3. **NIM** — Gateway off, no LLM deployment id, **and** `NIM_DEPLOYMENT_ID` set: same URL style as a deployment, using the NIM id.
4. **External** — Gateway off and neither deployment id set: LiteLLM talks to external providers using **their** env vars (for example `OPENAI_API_KEY`). Model names should not rely on the `datarobot/` prefix in that mode.

If both `LLM_DEPLOYMENT_ID` and `NIM_DEPLOYMENT_ID` are set with the gateway off, **deployment wins**.

## In `workflow.yaml`

The e2e samples usually declare one named LLM and reference it from `workflow`:

```yaml
llms:
  datarobot_llm:
    _type: datarobot-llm-component
```

That component follows the same four outcomes using fields on the block (gateway flag, deployment ids, etc.) and the environment above. Per-stack notes: [nat/llm.md](nat/llm.md), [langgraph/llm.md](langgraph/llm.md), [llamaindex/llm.md](llamaindex/llm.md), [crewai/llm.md](crewai/llm.md).
