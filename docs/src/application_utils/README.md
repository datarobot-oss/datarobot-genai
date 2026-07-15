# Application Utils

`datarobot-genai[application-utils]` provides building blocks for stateful agent
applications on top of the DataRobot Agentic Memory Service — a typed async ORM
and an AG-UI chat-history layer built on it.  Both use your existing
`DATAROBOT_ENDPOINT` / `DATAROBOT_API_TOKEN` credentials.

## Installation

```bash
pip install "datarobot-genai[application-utils]"
```

## Guides

| Doc | Focus |
|---|---|
| [memory_orm.md](memory_orm.md) | The Memory Service light ORM: `DRMemorySpace`, `DRSession`, `DREvent`, range-key encoding, optimistic concurrency, batch ops, typed (de)serialization. |
| [chat_history.md](chat_history.md) | The AG-UI chat-history layer: `Chat` / `Message` models, repositories, the `AGUIStorageAgent` state machine, disconnect survival and cancellation via `StreamPersistenceManager`, and the four extensibility points. |

## Environment variables

| Variable | Description |
|---|---|
| `DATAROBOT_ENDPOINT` | Full DataRobot API base URL, e.g. `https://app.datarobot.com/api/v2`. |
| `DATAROBOT_API_TOKEN` | DataRobot API bearer token. |

Both are resolved automatically; pass them as constructor arguments to override.
