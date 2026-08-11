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
| `datarobot_api_key` | no | `DATAROBOT_API_TOKEN` env var | Sent as the `X-DataRobot-Api-Key` header. |
| `datarobot_entity_id` | no | `deployment-<MLOPS_DEPLOYMENT_ID>` | Sent as the `X-DataRobot-Entity-Id` header, in `<entity type>-<id>` form: `deployment-`, `workload-` or `experiment_container-` (a use case). |
| `extra_headers` | no | `{}` | Additional headers; keys here win on collision with the DataRobot defaults. |
| `resource_attributes` | no | `{}` | Extra OTel resource attributes; keys here win on collision. |

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
| `OTEL_EXPORTER_OTLP_HEADERS` | Comma-separated `key=value` list sent as request headers, e.g. `X-DataRobot-Api-Key=<token>,X-DataRobot-Entity-Id=deployment-<id>`. Used verbatim; |

When the OTLP vars are not set, the runtime **falls back** to deriving the endpoint and headers from the DataRobot deployment env (populated for you inside a deployment):

| Fallback variable | Used for | Missing → |
|---|---|---|
| `DATAROBOT_API_TOKEN` | `X-DataRobot-Api-Key` header | Silent no-op; no spans reach DataRobot. |
| `MLOPS_DEPLOYMENT_ID` or `WORKLOAD_ID` | `X-DataRobot-Entity-Id`, auto-prefixed `deployment-` or `workload-` (deployment wins) | Silent no-op; no spans reach DataRobot. |
| `DATAROBOT_ENDPOINT` (or `DATAROBOT_PUBLIC_API_ENDPOINT`) | endpoint base; `/otel/v1/traces` appended | Silent no-op; no spans reach DataRobot. |

Optionally set `OTEL_SERVICE_NAME` to override the resource `service.name` used by the SDK bootstrap (the NAT exporter uses `project` from the YAML instead). It does not affect routing: the ingest attributes spans by the `X-DataRobot-Entity-Id` header.

## Local tracing

Outside a DataRobot runtime there is no deployment or workload to attribute spans to, so name the
entity yourself through the standard OTLP variables. A use case is the natural target for local
runs; the ingest knows one as an `experiment_container`:

```python
import os

from datarobot_genai.core.telemetry.agent import instrument
from datarobot_genai.langgraph.telemetry import instrument as instrument_langgraph

host = os.environ["DATAROBOT_ENDPOINT"].removesuffix("/").removesuffix("/api/v2")
os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", f"{host}/otel")
os.environ.setdefault(
    "OTEL_EXPORTER_OTLP_HEADERS",
    f"X-DataRobot-Api-Key={os.environ['DATAROBOT_API_TOKEN']},"
    f"X-DataRobot-Entity-Id=experiment_container-<use-case-id>",
)

instrument()
instrument_langgraph()
```

`setdefault`, not assignment, so the same code works unchanged once deployed: inside a runtime the
platform supplies these variables, and a collector you configured yourself keeps winning too.
`OTEL_SDK_DISABLED=true` turns export off, and `datarobot_otel_provider_installed()` reports
whether spans will reach DataRobot, worth asserting in a local smoke test since a missing
variable is otherwise a silent no-op. For looking a use case up by name, or creating one, see
[`quickstart.ipynb`](https://github.com/datarobot-oss/datarobot-genai/blob/main/e2e-tests/examples/quickstart.ipynb).

View the traces with the [`dr xp` plugin](https://docs.datarobot.com/en/docs/agentic-ai/cli/experimentation-plugin.html):

```bash
dr xp --entity-id <use-case-id>
```

## Troubleshooting

- **Data Exploration tab is empty**: Confirm the export is configured — `OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_EXPORTER_OTLP_HEADERS` (primary) or the `DATAROBOT_*` fallback. Both span paths silently skip when neither supplies an endpoint and headers.
- **NAT lifecycle spans appear but framework spans don't**: the framework `instrument()` (e.g. `datarobot_genai.langgraph.telemetry.instrument`) was not called, or was called after the framework imported. Move the call to the top of `register.py`.
- **Framework or memory spans appear in a separate trace from workflow spans**: confirm `datarobot_otelcollector` is enabled in `workflow.yaml` and `instrument()` is called in `register.py` before the framework imports. The exporter bridges NAT context into the SDK and the bootstrap wraps the global `TracerProvider` so LangChain/LangGraph, HTTP `POST`, and memory spans share the active workflow trace.
- **Spans land on the wrong entity**: `OTEL_EXPORTER_OTLP_HEADERS` wins over the `DATAROBOT_*` fallback, so a stale value there redirects everything. The bootstrap logs the entity it resolved (`DataRobot OTel span processor installed → ... (entity_id=...)`).
