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
`datarobot-genai` provides thin integration layers for **LangGraph**, **LlamaIndex**, and **CrewAI** so you can:

* Use the **DataRobot LLM Gateway** as the model backend (no API keys to manage).
* Wrap your agent graph/workflow into a **DataRobot-compatible agent class** with one function call.
* Stream AG-UI events (text, tool calls, steps) back to the DataRobot frontend.

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

Every framework integration follows the same three-step pattern:

1. **Get an LLM** — call `get_llm()` from the framework sub-package. It reads your DataRobot credentials from the environment and returns a framework-native LLM object.
2. **Define your agent** — build your graph / workflow / crew using the framework's own API.
3. **Wrap it** — call the `datarobot_agent_class_from_*` helper to produce a `BaseAgent` subclass that streams AG-UI events.

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
│  Your agent code (LangGraph / LlamaIndex /  │
│  CrewAI graph definition)                   │
├─────────────────────────────────────────────┤
│  datarobot_agent_class_from_*()             │  ← thin wrapper
│  BaseAgent.invoke() → AG-UI event stream    │
├─────────────────────────────────────────────┤
│  get_llm() → framework-native LLM client   │  ← LiteLLM under the hood
├─────────────────────────────────────────────┤
│  DataRobot LLM Gateway / Deployment / NIM   │
└─────────────────────────────────────────────┘
```

## License

Apache-2.0 — see [LICENSE](../LICENSE).
