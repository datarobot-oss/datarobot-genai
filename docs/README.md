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

Build AI agents with your favorite framework and run them on the DataRobot platform.
`datarobot-genai` provides thin integration layers for **LangGraph**, **LlamaIndex**, and **CrewAI**.

**AG-UI** — The library is built around the **Agent UI (AG-UI) protocol**: your agent’s `invoke()` takes `RunAgentInput` and **streams AG-UI events** (run lifecycle, streaming text, tool calls, steps/reasoning where applicable). That gives you a **single contract** for chat UIs, observability, and DataRobot-hosted frontends—no per-framework wire-up.

**Multi-agent, out of the box** — You are not limited to a single LLM turn. The same wrappers support **multi-step and multi-role workflows** from day one: LangGraph graphs with several nodes, LlamaIndex `AgentWorkflow` with handoffs, and CrewAI crews with sequential tasks—each mapped to the same AG-UI stream.

**Unified LLM layer** — Regardless of framework, you use the same pattern: **`get_llm()`** in `datarobot_genai.langgraph`, `llama_index`, or `crewai` returns that stack’s native client, but **routing is unified**: **LiteLLM** to the **DataRobot LLM Gateway**, a **deployment**, **NIM**, or an external model—selected from one **`Config`** / environment (`DATAROBOT_API_TOKEN`, `DATAROBOT_ENDPOINT`, `USE_DATAROBOT_LLM_GATEWAY`, `LLM_DEPLOYMENT_ID`, and related vars). All agent components therefore share a **single DataRobot-compatible LLM story** instead of ad hoc endpoints per integration.

You also get:

* **One-call wrapping** — `datarobot_agent_class_from_*` turns your graph, workflow, or crew into a **DataRobot-compatible agent class**.

## Prerequisites

| Requirement | Details |
|---|---|
| Python | 3.11 – 3.13 |
| DataRobot account | You need an API token and endpoint |

Set the following environment variables (or put them in a `.env` file):

```bash
export DATAROBOT_API_TOKEN="<your-api-token>"
export DATAROBOT_ENDPOINT="https://app.datarobot.com/api/v2"   # adjust for your environment
```

## Installation

Install the base package plus the extra for your framework of choice:

```bash
# LangGraph
pip install "datarobot-genai[langgraph]"

# LlamaIndex
pip install "datarobot-genai[llamaindex]"

# CrewAI
pip install "datarobot-genai[crewai]"

# Multiple frameworks at once
pip install "datarobot-genai[langgraph,llamaindex,crewai]"
```

## Quick overview

Every framework integration follows the same pattern—**one unified LLM setup, multi-agent optional, AG-UI always**:

1. **Get an LLM** — call `get_llm()` from `datarobot_genai.langgraph`, `llama_index`, or `crewai`. The same **DataRobot-aligned configuration** applies everywhere; you receive a **framework-native** chat client wired through the **unified LiteLLM layer**.
2. **Define your agent** — one agent or many: build a **LangGraph** `StateGraph`, a **LlamaIndex** `AgentWorkflow`, or a **CrewAI** `Crew` using the framework’s native APIs.
3. **Wrap it** — call `datarobot_agent_class_from_*` to get a `BaseAgent` subclass whose **`invoke()` streams AG-UI events** (text, tools, steps) to any compatible UI or platform.

## Framework guides

| Framework | Guide | Example | Extend the example |
|---|---|---|---|
| LangGraph | [docs/langgraph/](langgraph/) | [langgraph/agent_example.py](langgraph/agent_example.py) | [langgraph/AGENTS.md](langgraph/AGENTS.md) |
| LlamaIndex | [docs/llamaindex/](llamaindex/) | [llamaindex/agent_example.py](llamaindex/agent_example.py) | [llamaindex/AGENTS.md](llamaindex/AGENTS.md) |
| CrewAI | [docs/crewai/](crewai/) | [crewai/agent_example.py](crewai/agent_example.py) | [crewai/AGENTS.md](crewai/AGENTS.md) |

## Configuration reference

The library reads configuration from environment variables (via `datarobot-genai.core.config.Config`):

| Variable | Default | Description |
|---|---|---|
| `DATAROBOT_API_TOKEN` | — | Your DataRobot API token |
| `DATAROBOT_ENDPOINT` | `https://app.datarobot.com/api/v2` | DataRobot API endpoint |
| `USE_DATAROBOT_LLM_GATEWAY` | `true` | When `true`, route LLM calls through the DataRobot LLM Gateway |
| `LLM_DEPLOYMENT_ID` | — | Use a specific LLM deployment instead of the gateway |
| `NIM_DEPLOYMENT_ID` | — | Use an NVIDIA NIM deployment |
| `LLM_DEFAULT_MODEL` | `datarobot-deployed-llm` | Default model name passed to LiteLLM |
| `DATAROBOT_GENAI_MAX_HISTORY_MESSAGES` | `20` | Max prior messages included in chat history |

### LLM routing

`get_llm()` picks the backend automatically based on which variables are set:

1. `USE_DATAROBOT_LLM_GATEWAY=true` (default) → DataRobot LLM Gateway
2. `LLM_DEPLOYMENT_ID` is set → DataRobot deployment proxy
3. `NIM_DEPLOYMENT_ID` is set → NVIDIA NIM deployment
4. None of the above → external LiteLLM (reads provider keys from the environment)

## Architecture at a glance

```
┌─────────────────────────────────────────────┐
│  Multi-agent graph / workflow / crew        │  ← LangGraph, LlamaIndex, CrewAI
├─────────────────────────────────────────────┤
│  datarobot_agent_class_from_*()             │  ← thin wrapper
│  BaseAgent.invoke(RunAgentInput)            │
│       → async stream of AG-UI Events        │  ← AG-UI protocol
├─────────────────────────────────────────────┤
│  Unified LLM layer: get_llm() per stack      │  ← same Config / env for all
│       → framework-native LLM client         │  ← LiteLLM
├─────────────────────────────────────────────┤
│  DataRobot Gateway / Deployment / NIM / ext  │  ← DataRobot-compatible routing
└─────────────────────────────────────────────┘
```

## License

Apache-2.0 — see [LICENSE](../LICENSE).
