<!--
  ~ Copyright 2026 DataRobot, Inc. and its affiliates.
  ~
  ~ Licensed under the Apache License, Version 2.0 (the "License");
  ~ you may not use this file except in compliance with the License.
  ~ You may obtain a copy of the License at
  ~
  ~     http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS,
  ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ~ See the License for the specific language governing permissions and
  ~ limitations under the License.
-->

# OpenTelemetry tracing

How to wire DRAgent spans and view tracing in the deployment's **Monitoring -> Data exploration** tab in DataRobot.

## What gets traced

Two independent span sources reach DataRobot, each wired through its own switch:

- **NAT lifecycle spans** — workflow runs, tool calls, and other `IntermediateStep`-derived events NAT emits as your `workflow.yaml` executes. Enabled by a block in `workflow.yaml` (see below).
- **Framework auto-instrumentor spans** — spans emitted by `opentelemetry-instrumentation-crewai`, `-langchain`, `-llamaindex`, and `-openai`. HTTP-client and OpenAI SDK spans are enabled by calling the core `instrument()`; framework spans are enabled by calling the matching framework `instrument()` (e.g. `datarobot_genai.langgraph.telemetry.instrument`) from your own code.
- **Mem0 memory spans** — `update_memory`, `search_memory`, and `delete_memory` spans emitted by the `dr_mem0_memory` NAT provider when `streaming_memory_agent` / `auto_memory_agent` store or retrieve long-term memory. Enabled automatically once the OTel SDK bootstrap from `instrument()` is active (same env vars as above); no extra YAML config.

When both the NAT exporter and SDK bootstrap are active, `datarobot_otelcollector` mirrors NAT span hierarchy into the OTel SDK context and the SDK bootstrap wraps the global `TracerProvider` so framework, HTTP, and memory spans nest under the active workflow trace instead of exporting as separate trees.

You generally want both NAT lifecycle and framework spans; mem0 spans appear automatically when memory is configured and tracing is enabled.

## `workflow.yaml`: enable the NAT exporter

Add a `general.telemetry.tracing` block. The exporter `_type: datarobot_otelcollector` is registered as a NAT plugin and discovered automatically when the `dragent` extra is installed.

```yaml
general:
  telemetry:
    tracing:
      otelcollector:
        _type: datarobot_otelcollector
        project: "<your-agent-name>"   # becomes the OTel service.name
```

Fields:

| Field | Required? | Default | Description |
|---|---|---|---|
| `project` | yes | — | OTel `service.name` for spans emitted by this workflow. |
| `endpoint` | no | `<DATAROBOT_(PUBLIC_)ENDPOINT>/otel/v1/traces` | Full OTLP/HTTP endpoint override. |
| `extra_headers` | no | `{}` | Additional headers; keys here win on collision with the DataRobot defaults. |
| `resource_attributes` | no | `{}` | Extra OTel resource attributes; keys here win on collision. |

The API key and entity id come from the environment (see below), not from this block. To override the entity for this workflow only, set `extra_headers` — e.g. `{X-DataRobot-Entity-Id: experiment_container-<use-case-id>}` for a use case. The API key still comes from the environment, so `extra_headers` alone does not authenticate a local run; use `OTEL_EXPORTER_OTLP_HEADERS` for that. Unknown keys in this block are silently ignored, so a misspelled one fails quietly.

Batch-tuning knobs (`batch_size`, `flush_interval`, `max_queue_size`, etc.) are inherited from NAT's `BatchConfigMixin`; defaults are fine for most agents.

## `register.py`: call `instrument()`

The core `instrument()` sets up HTTP-client, OpenAI SDK, and threading instrumentation plus the DataRobot OTel SDK bootstrap. The NAT exporter only carries NAT's own spans. To also route framework auto-instrumentor spans (CrewAI / LangChain / LlamaIndex) to DataRobot, call the matching framework `instrument()` at module-import time in your agent's `register.py`, before the framework constructs any agents:

```python
from datarobot_genai.core.telemetry.agent import instrument
from datarobot_genai.langgraph.telemetry import instrument as instrument_langgraph

instrument()  # HTTP clients + OpenAI SDK + OTel SDK bootstrap
instrument_langgraph()  # framework auto-instrumentor spans
```

The per-framework helpers live alongside each framework package:

| Framework | Import |
|---|---|
| CrewAI | `from datarobot_genai.crewai.telemetry import instrument` |
| LangChain / LangGraph | `from datarobot_genai.langgraph.telemetry import instrument` |
| LlamaIndex | `from datarobot_genai.llama_index.telemetry import instrument` |

All of these are idempotent — repeat calls are no-ops — and safe to keep in `register.py` during local development: when the DataRobot deployment environment variables below are not all set, the underlying `bootstrap_otel_provider_for_datarobot()` silently skips installing the SDK provider, so framework spans simply go nowhere instead of erroring.

## Required environment

The export endpoint and auth headers are configured through the standard OpenTelemetry env vars — this is the **primary** mechanism, and in practice every environment (deployments, notebooks, local, CI) relies on it. Both span paths (the NAT `datarobot_otelcollector` exporter and the `instrument()` SDK bootstrap) read them first.

| Variable | Description |
|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP/HTTP base URL; `/v1/traces` is appended. Point it at `<host>/otel` (not `<host>/otel/v1/traces`) to hit the DataRobot ingest path. |
| `OTEL_EXPORTER_OTLP_HEADERS` | Comma-separated `key=value` list sent as request headers, e.g. `X-DataRobot-Api-Key=<token>,X-DataRobot-Entity-Id=deployment-<id>`. Passed through as given; an entry that is not `key=value` is skipped. |

When the OTLP vars are not set, the runtime **falls back** to deriving the endpoint and headers from the DataRobot deployment env (populated for you inside a deployment):

| Fallback variable | Used for | Missing → |
|---|---|---|
| `DATAROBOT_API_TOKEN` | `X-DataRobot-Api-Key` header | Silent no-op; no spans reach DataRobot. |
| `MLOPS_DEPLOYMENT_ID`, `WORKLOAD_ID` or `DATAROBOT_USE_CASE_ID` | `X-DataRobot-Entity-Id`, auto-prefixed `deployment-`, `workload-` or `experiment_container-`. First one set wins, in that order. | Silent no-op; no spans reach DataRobot. |
| `DATAROBOT_ENDPOINT` (or `DATAROBOT_PUBLIC_API_ENDPOINT`) | endpoint base; `/otel/v1/traces` appended | Silent no-op; no spans reach DataRobot. |

Optionally set `OTEL_SERVICE_NAME` to override the resource `service.name` used by the SDK bootstrap (the NAT exporter uses `project` from the YAML instead). It does not affect routing: the ingest attributes spans by the `X-DataRobot-Entity-Id` header.

## Local tracing

Outside a DataRobot runtime there is no deployment or workload to attribute spans to, so name the
entity yourself through the standard OTLP variables. A use case is the natural target for local
runs; the ingest knows one as an `experiment_container`:

```python
from datarobot_genai.core.telemetry import trace_to_use_case
from datarobot_genai.langgraph.telemetry import instrument as instrument_langgraph

print(trace_to_use_case("My local runs"))  # reuses or creates that use case
instrument_langgraph()
```

`trace_to_use_case` picks the use case, calls `instrument()`, and reports where spans went. In an
agent's `register.py`, where the use case id comes from configuration rather than a name, set the
variable it reads and call `instrument()` yourself:

```python
import os

from datarobot_genai.core.telemetry.agent import instrument
from datarobot_genai.langgraph.telemetry import instrument as instrument_langgraph

os.environ["DATAROBOT_USE_CASE_ID"] = "<use-case-id>"

instrument()
instrument_langgraph()
```

`DATAROBOT_USE_CASE_ID` is read only when neither a deployment nor a workload id is set, and
`OTEL_EXPORTER_OTLP_HEADERS` outranks all three, so neither form overrides what a runtime already
configured. The endpoint and API key come from `DATAROBOT_(PUBLIC_)ENDPOINT` and
`DATAROBOT_API_TOKEN`, and `OTEL_SDK_DISABLED=true` stops this SDK path from exporting (the NAT
exporter in `workflow.yaml` has its own switch). For looking a use case up by name, or creating one,
see [`quickstart.ipynb`](https://github.com/datarobot-oss/datarobot-genai/blob/main/e2e-tests/examples/quickstart.ipynb).

View the traces with the [`dr xp` plugin](https://docs.datarobot.com/en/docs/agentic-ai/cli/experimentation-plugin.html):

```bash
dr xp --entity-id <use-case-id>
```

## Troubleshooting

- **Data Exploration tab is empty**: Confirm the export is configured — `OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_EXPORTER_OTLP_HEADERS` (primary) or the `DATAROBOT_*` fallback. Both span paths silently skip when neither supplies an endpoint and headers.
- **NAT lifecycle spans appear but framework spans don't**: the framework `instrument()` (e.g. `datarobot_genai.langgraph.telemetry.instrument`) was not called, or was called after the framework imported. Move the call to the top of `register.py`.
- **Framework or memory spans appear in a separate trace from workflow spans**: confirm `datarobot_otelcollector` is enabled in `workflow.yaml` and `instrument()` is called in `register.py` before the framework imports. The exporter bridges NAT context into the SDK and the bootstrap wraps the global `TracerProvider` so LangChain/LangGraph, HTTP `POST`, and memory spans share the active workflow trace.
- **Spans land on the wrong entity**: `OTEL_EXPORTER_OTLP_HEADERS` wins over the `DATAROBOT_*` fallback, so a stale value there redirects everything. The bootstrap logs the entity it resolved (`DataRobot OTel span processor installed → ... (entity_id=...)`).
