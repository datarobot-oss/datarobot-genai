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

Build agents with **LangGraph**, **LlamaIndex**, or **CrewAI**, or define them in a **NAT `workflow.yaml`** and run them with **DRAgent** (HTTP + AG-UI). The docs are aligned with the **examples under [`e2e-tests/dragent/`](https://github.com/datarobot-oss/datarobot-genai/tree/main/e2e-tests/dragent)**&mdash;that folder is the source of truth for what you configure and what you run.

**What you see in practice**

- A **`workflow.yaml`** per sample&mdash;names the LLM (`llms:`), chooses the runner (`workflow._type:`), and optionally adds tools, MCP groups, and auth. DRAgent reads this file.
- A **`myagent.py`** (framework agents)&mdash;where you define the graph, crew, or workflow; DRAgent ties it to the YAML via registration.
- **Environment variables** for API token, endpoint, and LLM routing&mdash;same ideas everywhere. See [LLM configuration](llm.md).

**AG-UI**&mdash;chat UIs and tests consume a stream of **AG-UI events** (run lifecycle, text, tools, steps). DRAgent serves that over SSE; the examples show end-to-end behavior.

**Notebook**&mdash;[`Example.ipynb`](Example.ipynb) is an optional Jupyter walkthrough (credentials, listing gateway models, a **LangGraph** graph + `datarobot_agent_class_from_langgraph`, AG-UI `invoke`).

## Prerequisites

| Requirement | Details |
|---|---|
| Python | 3.11–3.13. |
| DataRobot account | API token and endpoint. |

```bash
# Set your DataRobot API token (replace the placeholder).
export DATAROBOT_API_TOKEN=YOUR_DATAROBOT_API_TOKEN
export DATAROBOT_ENDPOINT=https://app.datarobot.com/api/v2
```

Adjust `DATAROBOT_ENDPOINT` for your environment if it differs from the default.

## Installation

```bash
# Framework SDKs (see each guide for extras)
pip install "datarobot-genai[langgraph]"
pip install "datarobot-genai[llamaindex]"
pip install "datarobot-genai[crewai]"

# NAT workflow + DRAgent (YAML-driven agents)
pip install "datarobot-genai[dragent]"
```

## Guides (what to edit in the examples)

Each guide explains the **interfaces you see in the repo**: `workflow.yaml` keys, env vars, and the Python file the sample ships with.

| Integration | Overview | Agent & workflow surface | LLM options | Tools & MCP | Caveats |
|---|---|---|---|---|---|
| LangGraph | [langgraph/](langgraph/) | [langgraph/agent.md](langgraph/agent.md) | [LLM configuration](llm.md) | [langgraph/mcp.md](langgraph/mcp.md) | [langgraph/caveats.md](langgraph/caveats.md) |
| LlamaIndex | [llamaindex/](llamaindex/) | [llamaindex/agent.md](llamaindex/agent.md) | [LLM configuration](llm.md) | [llamaindex/mcp.md](llamaindex/mcp.md) | [llamaindex/caveats.md](llamaindex/caveats.md) |
| CrewAI | [crewai/](crewai/) | [crewai/agent.md](crewai/agent.md) | [LLM configuration](llm.md) | [crewai/mcp.md](crewai/mcp.md) | [crewai/caveats.md](crewai/caveats.md) |
| NAT + DRAgent | [nat/](nat/) | [nat/agent.md](nat/agent.md) | [nat/llm.md](nat/llm.md) | [nat/mcp.md](nat/mcp.md) | [nat/caveats.md](nat/caveats.md) |

Shared **LLM routing** (gateway vs deployment vs NIM vs external): [LLM configuration](llm.md).

**NAT note:** the contract you edit is **`workflow.yaml`**. DRAgent is the supported way to host those workflows. A legacy in-process Python path exists for loading the same YAML without DRAgent; new work should target DRAgent.

Minimal custom agent without a framework wrapper: [`e2e-tests/dragent/base/myagent.py`](../e2e-tests/dragent/base/myagent.py) and its [`workflow.yaml`](../e2e-tests/dragent/base/workflow.yaml).

## DRAgent CLI

Standalone CLI for running and querying DRAgent workflows via NAT. See [docs/dragent/](dragent/) for full usage (`serve`, `run`, `query`, completion JSON, authentication, debugging).

## Configuration reference

Environment variables the examples and `workflow.yaml` assume (full table): [LLM configuration](llm.md).

| Variable | Default | Description |
|---|---|---|
| `DATAROBOT_API_TOKEN` | — | Your DataRobot API token. |
| `DATAROBOT_ENDPOINT` | `https://app.datarobot.com/api/v2` | DataRobot API endpoint. |
| `USE_DATAROBOT_LLM_GATEWAY` | `true` | Use the DataRobot LLM Gateway when `true`. |
| `LLM_DEPLOYMENT_ID` | — | Route to a specific LLM deployment when the gateway is off. |
| `NIM_DEPLOYMENT_ID` | — | Route to an NVIDIA NIM deployment when the gateway is off. |
| `LLM_DEFAULT_MODEL` | `datarobot-deployed-llm` | Default model name. |
| `DATAROBOT_GENAI_MAX_HISTORY_MESSAGES` | `20` | Maximum prior messages in history. |

## License

Apache-2.0 — see [LICENSE](../LICENSE).
