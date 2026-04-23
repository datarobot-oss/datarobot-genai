# LangGraph + DataRobot

**What you edit in the example:** [`e2e-tests/dragent/langgraph/myagent.py`](../../e2e-tests/dragent/langgraph/myagent.py) (graph + prompt) and [`e2e-tests/dragent/langgraph/workflow.yaml`](../../e2e-tests/dragent/langgraph/workflow.yaml) (LLM wiring + `workflow._type`). DRAgent runs the pair.

## Installation

```bash
pip install "datarobot-genai[langgraph]"
```

## Guides (what appears in the sample)

| Doc | Focus |
|---|---|
| [agent.md](agent.md) | `workflow.yaml` + what `myagent.py` defines |
| [mcp.md](mcp.md) | Extra tools merged into your graph (when MCP is enabled) |

Env reference: [LLM configuration (shared)](../llm.md).

## Run the e2e sample

From [`e2e-tests/`](../../e2e-tests/):

```bash
export DATAROBOT_API_TOKEN=YOUR_DATAROBOT_API_TOKEN
export DATAROBOT_ENDPOINT=https://app.datarobot.com/api/v2
uv sync --group dragent-langgraph
uv run --group dragent-langgraph nat dragent run \
  --config_file dragent/langgraph/workflow.yaml \
  --input "Say hello in one short sentence."
```

HTTP: [`e2e-tests/dragent/Taskfile.yaml`](../../e2e-tests/dragent/Taskfile.yaml).
