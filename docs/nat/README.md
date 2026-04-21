# NAT + DRAgent (YAML workflows)

**What you edit:** a single **`workflow.yaml`** (plus optional small Python modules that register extra tools). **DRAgent** is the runner and HTTP front end: it loads that file and exposes AG-UI over SSE.

The canonical example is [`e2e-tests/dragent/nat/workflow.yaml`](../../e2e-tests/dragent/nat/workflow.yaml). The sections below match what you see there.

## Installation

```bash
pip install "datarobot-genai[dragent]"
```

## Guides (interfaces in the example)

| Topic | What it explains |
|---|---|
| [agent.md](agent.md) | Top-level **`workflow.yaml`**: `functions`, `workflow`, `llms`, how they connect |
| [llm.md](llm.md) | The **`llms:`** block and `_type` values (e.g. `datarobot-llm-component`) |
| [mcp.md](mcp.md) | **`function_groups`**, **`authentication`**, MCP tools in **`tool_names`** |

Shared env vars: [LLM configuration (shared)](../llm.md). Streaming behavior (NAT ≥ 1.6): [NAT 1.6 streaming in DRAgent](../nat-1.6-streaming.md).

## Run the example

From [`e2e-tests/`](../../e2e-tests/) (with credentials in the environment):

```bash
uv sync --group dragent-nat
uv run --group dragent-nat nat dragent run \
  --config_file dragent/nat/workflow.yaml \
  --input "Your prompt."
```

Serve HTTP: [`e2e-tests/dragent/Taskfile.yaml`](../../e2e-tests/dragent/Taskfile.yaml) (`run-nat`).

**Note:** Older samples sometimes loaded the same YAML through a small Python wrapper. **DRAgent is the path forward**; treat **`workflow.yaml` + `nat dragent`** as the supported interface.
