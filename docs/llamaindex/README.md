# LlamaIndex + DataRobot

**What you edit:** [`e2e-tests/dragent/llamaindex/myagent.py`](../../e2e-tests/dragent/llamaindex/myagent.py) (workflow + agents) and [`workflow.yaml`](../../e2e-tests/dragent/llamaindex/workflow.yaml).

## Installation

```bash
pip install "datarobot-genai[llamaindex]"
```

## Guides

| Doc | Focus |
|---|---|
| [agent.md](agent.md) | YAML + `myagent.py` in the sample |
| [mcp.md](mcp.md) | Injected tools vs per-agent tools |

[LLM configuration (shared)](../llm.md).

## Run the e2e sample

```bash
export DATAROBOT_API_TOKEN=YOUR_DATAROBOT_API_TOKEN
export DATAROBOT_ENDPOINT=https://app.datarobot.com/api/v2
uv sync --group dragent-llamaindex
uv run --group dragent-llamaindex nat dragent run \
  --config_file dragent/llamaindex/workflow.yaml \
  --input "Say hello in one short sentence."
```

HTTP: [`e2e-tests/dragent/Taskfile.yaml`](../../e2e-tests/dragent/Taskfile.yaml).
