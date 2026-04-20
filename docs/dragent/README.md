# DRAgent CLI

`nat dragent` is a standalone CLI for running and querying DRAgent workflows built on NVIDIA NeMo Agent Toolkit (NAT).

## Installation

```bash
pip install "datarobot-genai[dragent]"
```

## Commands

### `nat dragent serve`

Start the DRAgent HTTP server locally. Serves the AG-UI `/generate/stream` endpoint.

```bash
nat dragent serve --config_file agent/workflow.yaml --port 8842
```

Key options:

| Flag | Description |
|---|---|
| `--config_file` | Path to the workflow YAML config. Falls back to `DRAGENT_CONFIG_FILE` env var. |
| `--port` | Port to bind the server to. Falls back to `AGENT_PORT` env var. |
| `--reload` | Enable auto-reload for development (`true`/`false`). |
| `--a2a` | Expose the agent via the Agent2Agent protocol (endpoints mounted under `/a2a/`). |
| `--override` | Override config values using dot notation (e.g., `--override llms.nim_llm.temperature 0.7`). |

### `nat dragent run`

Execute a workflow locally in-process (no server required).

```bash
nat dragent run --config_file agent/workflow.yaml --input "What is AI?"
```

Key options:

| Flag | Alias | Description |
|---|---|---|
| `--input` | `--user_prompt` | Prompt string to send to the workflow. |
| `--completion-json` | `--completion_json` | Path to a JSON file containing the chat completion payload. |
| `--config_file` | | Path to the workflow YAML config. Falls back to `DRAGENT_CONFIG_FILE` env var. |
| `--override` | | Override config values using dot notation. |

### `nat dragent query`

Query a running DRAgent server (local or deployed).

```bash
# Query a local server
nat dragent query --local --input "What is AI?"

# Query a DataRobot deployment
nat dragent query --deployment-id DEPLOYMENT_ID --input "What is AI?"

# Query with chat history from a completion JSON file
nat dragent query --local --completion-json completion.json
```

Key options:

| Flag | Alias | Description |
|---|---|---|
| `--local` | | Query localhost using `--port` or `AGENT_PORT` env var. |
| `--port` | | Port for `--local`. Falls back to `AGENT_PORT` env var. |
| `--deployment-id` | `--deployment_id` | DataRobot deployment ID (mutually exclusive with `--local`). |
| `--input` | `--user_prompt` | Prompt string. |
| `--completion-json` | `--completion_json` | Path to a JSON file containing the chat completion payload. Messages are extracted into an AG-UI payload. |
| `--show-payload` | | Print the AG-UI request payload before sending. |

## Completion JSON format

The `--completion-json` flag accepts a JSON file with an OpenAI-compatible chat completion structure. Messages are extracted and forwarded as an AG-UI payload:

```json
{
  "model": "datarobot-deployed-llm",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI is..."},
    {"role": "user", "content": "Tell me more"}
  ]
}
```

Only the `messages` array is used. Each message must have `role` and `content` keys.

## Authentication

When querying a DataRobot deployment, the CLI reads credentials from environment variables or flags:

| Source | Flag | Env var |
|---|---|---|
| API token | `--api-token` | `DATAROBOT_API_TOKEN` |
| API endpoint | `--base-url` | `DATAROBOT_ENDPOINT` |

For local queries, auth context headers are built from `SESSION_SECRET_KEY` and `DATAROBOT_USER_ID` if set.

## Debugging

Control NAT's log verbosity with the `NAT_LOG_LEVEL` environment variable:

```bash
export NAT_LOG_LEVEL=DEBUG
nat dragent serve --config_file agent/workflow.yaml --port 8842
```

Supported values: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Defaults to `INFO` if not set.
