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

<p align="center">
  <a href="https://github.com/datarobot-oss/datarobot-genai">
    <img src="img/datarobot_logo.avif" width="600px" alt="DataRobot Logo"/>
  </a>
</p>
<p align="center">
    <span style="font-size: 1.5em; font-weight: bold; display: block;">DataRobot GenAI Library</span>
</p>

<p align="center">
  <a href="https://www.datarobot.com/">Homepage</a>
  ·
  <a href="https://pypi.org/project/datarobot-genai/">PyPI</a>
  ·
  <a href="https://docs.datarobot.com/en/docs/get-started/troubleshooting/general-help.html">Support</a>
</p>

Build agents with **LangGraph**, **LlamaIndex**, or **CrewAI**, or define them in a **NAT `workflow.yaml`** and run them with **DRAgent** (HTTP + AG-UI). Use the **examples under [`e2e-tests/dragent/`](https://github.com/datarobot-oss/datarobot-genai/tree/main/e2e-tests/dragent)** as the source of truth for configuration and how to run that stack&mdash;these docs follow those samples.

**What the samples demonstrate**

- In each sample, **`workflow.yaml`** names the LLM (`llms:`), picks the runner (`workflow._type:`), and optionally adds tools, MCP groups, and auth. DRAgent reads this file.
- In each sample, **`myagent.py`** (framework agents)&mdash;defines the graph, crew, or workflow. DRAgent registers it against the YAML.
- **Environment variables** for API token, endpoint, and LLM routing&mdash;the same ideas apply across stacks. See [LLM configuration](llm.md).

**AG-UI**&mdash;chat UIs and tests work with a stream of **AG-UI events** (run lifecycle, text, tools, steps). DRAgent serves those events over SSE; the examples walk through the flow end to end.

## Guides (what to edit in the examples)

Each guide covers the **interfaces in the repo**: `workflow.yaml` keys, env vars, and the Python file each sample ships.

| Integration | Overview | Agent and workflow surface | LLM options | Tools and MCP | Caveats |
|---|---|---|---|---|---|
| LangGraph | [langgraph/](langgraph/) | [langgraph/agent.md](langgraph/agent.md) | [LLM configuration](llm.md) | [langgraph/mcp.md](langgraph/mcp.md) | [langgraph/caveats.md](langgraph/caveats.md) |
| LlamaIndex | [llamaindex/](llamaindex/) | [llamaindex/agent.md](llamaindex/agent.md) | [LLM configuration](llm.md) | [llamaindex/mcp.md](llamaindex/mcp.md) | [llamaindex/caveats.md](llamaindex/caveats.md) |
| CrewAI | [crewai/](crewai/) | [crewai/agent.md](crewai/agent.md) | [LLM configuration](llm.md) | [crewai/mcp.md](crewai/mcp.md) | [crewai/caveats.md](crewai/caveats.md) |
| NAT + DRAgent | [nat/](nat/) | [nat/agent.md](nat/agent.md) | [nat/llm.md](nat/llm.md) | [nat/mcp.md](nat/mcp.md) | [nat/caveats.md](nat/caveats.md) |

For shared **LLM routing** (gateway vs deployment vs NIM vs external), see [LLM configuration](llm.md).
For native **primary/fallback failover** changes in an existing component, see [LLM provider fallback (router)](fallback.md).
For NAT automatic memory with Mem0, see [nat/memory.md](nat/memory.md).

**Note:** With NAT, edit **`workflow.yaml`** as the contract. Host workflows with DRAgent for the supported path. The same YAML can still load in process without DRAgent (legacy); target DRAgent for new work.

For a minimal custom agent without a framework wrapper, start from [`e2e-tests/dragent/base/myagent.py`](../e2e-tests/dragent/base/myagent.py) and its [`workflow.yaml`](../e2e-tests/dragent/base/workflow.yaml).

## DRAgent CLI

The standalone CLI runs and queries DRAgent workflows over NAT. See [docs/dragent/](dragent/) for `serve`, `run`, `query`, completion JSON, authentication, and debugging. For wiring OpenTelemetry tracing into the deployment **Data Exploration** tab, see [docs/dragent/tracing.md](dragent/tracing.md).

## Configuration reference

The examples and `workflow.yaml` expect the variables below; see [LLM configuration](llm.md) for the full table and details.

| Variable | Default | Description |
|---|---|---|
| `DATAROBOT_API_TOKEN` | — | DataRobot API token. |
| `DATAROBOT_ENDPOINT` | `https://app.datarobot.com/api/v2` | Base URL for DataRobot API requests. |
| `LLM_USE_DATAROBOT_LLM_GATEWAY` | `true` | Set to `true` to use the DataRobot LLM Gateway. |
| `LLM_DEPLOYMENT_ID` | — | Set this to target a specific LLM deployment when the gateway is off. |
| `LLM_NIM_DEPLOYMENT_ID` | — | Set this to target an NVIDIA NIM deployment when the gateway is off. |
| `LLM_DEFAULT_MODEL` | `datarobot-deployed-llm` | Default model name for runs. |
| `DATAROBOT_GENAI_MAX_HISTORY_MESSAGES` | `20` | Maximum number of prior messages the client keeps in history. |
| `AUTH_RESOLUTION_STRATEGY` | `http` | How `drtools` resolves secrets: `http` or `config`. See [drtools/auth.md](drtools/auth.md). |

## drtools

Agentic tools and credential/auth resolution for MCP servers and in-process agents. See [drtools/](drtools/).

## Troubleshooting and best practices

Symptom-based troubleshooting and recommended patterns live in the [documentation index](../README.md#troubleshooting). Framework-specific caveats are linked from the guides table above.

## License

Apache-2.0 — see [LICENSE](../LICENSE).
