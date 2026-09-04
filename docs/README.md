# DataRobot GenAI library documentation

Documentation for building agents with **LangGraph**, **LlamaIndex**, **CrewAI**, and **NAT** on DataRobot. The published site is at [datarobot-oss.github.io/datarobot-genai](https://datarobot-oss.github.io/datarobot-genai/).

Source files live under [`src/`](src/). Use the [e2e samples](../e2e-tests/dragent/) as the source of truth for configuration and runtime wiring.

## Table of contents

- [Overview](src/index.md)
- [LLM configuration](src/llm.md)
- [LLM provider fallback (router)](src/fallback.md)
- Framework guides
  - [LangGraph](src/langgraph/README.md)
  - [LlamaIndex](src/llamaindex/README.md)
  - [CrewAI](src/crewai/README.md)
  - [NAT + DRAgent](src/nat/README.md)
- [DRAgent CLI](src/dragent/README.md) and [tracing](src/dragent/tracing.md)
- [drtools](src/drtools/README.md) — agentic tools and auth resolution
- [Design notes](src/design/nat-1.6-streaming.md)
- [API reference](https://datarobot-oss.github.io/datarobot-genai/api/) (generated from source)

## Prerequisites

Install and runtime requirements before using the library:

| Requirement | Notes |
|---|---|
| Python 3.11–3.13 | Required for all installs |
| DataRobot account | Required for gateway, deployments, MCP, and platform features |
| `DATAROBOT_API_TOKEN` / `DATAROBOT_ENDPOINT` | See [configuration reference](src/index.md#configuration-reference) |
| Framework extra | Install `datarobot-genai[langgraph]`, `[llamaindex]`, `[crewai]`, or `[nat]` |

For local development, also install [`uv`](https://docs.astral.sh/uv/), [Task](https://taskfile.dev/), and [pre-commit](https://pre-commit.com/). See the [root README](../README.md#develop).

## Quick start

1. Install the extra that matches the target framework:

   ```bash
   pip install "datarobot-genai[langgraph]"
   ```

2. Export DataRobot credentials:

   ```bash
   export DATAROBOT_API_TOKEN=YOUR_DATAROBOT_API_TOKEN
   export DATAROBOT_ENDPOINT=https://app.datarobot.com/api/v2
   ```

3. Follow a framework guide or the [quickstart notebook](../e2e-tests/examples/quickstart.ipynb).

4. For the DRAgent HTTP stack, start from [`e2e-tests/dragent/`](../e2e-tests/dragent/) and [DRAgent CLI](src/dragent/README.md).

## Configuration

Core environment variables are listed in the [configuration reference](src/index.md#configuration-reference). LLM routing (gateway, deployment, NIM, external) is documented in [LLM configuration](src/llm.md). Tool and MCP auth uses [`AUTH_RESOLUTION_STRATEGY`](src/drtools/auth.md).

## Troubleshooting

Use the table below to find documentation for common symptoms:

| Symptom | Where to look |
|---|---|
| LLM auth or routing errors | [LLM configuration](src/llm.md), verify `DATAROBOT_API_TOKEN` and gateway flags |
| MCP tools missing at runtime | Framework [MCP guides](src/langgraph/mcp.md) — merge injected tools with local tools |
| DRAgent tracing empty | [DRAgent tracing](src/dragent/tracing.md) — OTLP env vars and `instrument()` in `register.py` |
| A2A auth failures | [A2A authentication](src/nat/a2a-auth.md) and [A2A client](src/nat/a2a-client.md) |
| NAT streaming or AG-UI mismatch | [NAT caveats](src/nat/caveats.md), [NAT 1.6 streaming design](src/design/nat-1.6-streaming.md) |
| E2E test matrix locally | [e2e-tests README](../e2e-tests/README.md) |

## Best practices

- Treat **`workflow.yaml` + `myagent.py`** as the supported contract for DRAgent-hosted agents.
- Prefer the **DataRobot LLM Gateway** when combining MCP with CrewAI ([caveats](src/crewai/caveats.md)).
- Use a **durable LangGraph checkpointer** for human-in-the-loop flows in production ([HITL guide](src/langgraph/hitl.md)).
- Enable **OpenTelemetry tracing** in both `workflow.yaml` and `register.py` for full span coverage ([tracing](src/dragent/tracing.md)).

## Related resources

- [Repository README](../README.md) — install, develop, publish
- [CONTRIBUTING](../CONTRIBUTING.md) — issues, changelog, versioning
- [CHANGELOG](../CHANGELOG.md) — release history
- [DataRobot product documentation](https://docs.datarobot.com/)
