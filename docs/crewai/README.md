# CrewAI + DataRobot

**What you edit:** [`e2e-tests/dragent/crewai/myagent.py`](../../e2e-tests/dragent/crewai/myagent.py) (agents, tasks, crew, kickoff inputs) and [`workflow.yaml`](../../e2e-tests/dragent/crewai/workflow.yaml).

## Installation

```bash
pip install "datarobot-genai[crewai]"
```

## Guides

| Doc | Focus |
|---|---|
| [agent.md](agent.md) | YAML + `myagent.py` |
| [llm.md](llm.md) | Env + `llms:` |
| [mcp.md](mcp.md) | Injected tools on agents |
| [caveats.md](caveats.md) | Interface caveats |

[LLM configuration (shared)](../llm.md).

## Run the e2e sample

```bash
export DATAROBOT_API_TOKEN="<token>"
export DATAROBOT_ENDPOINT="https://app.datarobot.com/api/v2"
uv sync --group dragent-crewai
uv run --group dragent-crewai nat dragent run \
  --config_file dragent/crewai/workflow.yaml \
  --input "Say hello in one short sentence."
```

HTTP: [`e2e-tests/dragent/Taskfile.yaml`](../../e2e-tests/dragent/Taskfile.yaml).
