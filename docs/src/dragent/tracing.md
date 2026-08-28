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

### Auth and entity headers

The exporter always sends two DataRobot headers. They are **not** `workflow.yaml` fields — the exporter resolves them from the environment on every startup, then merges `extra_headers` over the top, so `extra_headers` is how you override either one.

| Header | Resolved from | Notes |
|---|---|---|
| `X-DataRobot-Api-Key` | `DATAROBOT_API_TOKEN` | Omitted (along with the entity header) when the token is missing. |
| `X-DataRobot-Entity-Id` | `MLOPS_DEPLOYMENT_ID` → `deployment-<id>`, or `WORKLOAD_ID` → `workload-<id>` | Deployment wins when both are set. The `deployment-` / `workload-` prefix is required by the ingest and added for you. Overrides are **not** validated locally, so an unprefixed value is sent as-is and the ingest drops the spans. |

Batch-tuning knobs (`batch_size`, `flush_interval`, `max_queue_size`, etc.) are inherited from NAT's `BatchConfigMixin`; defaults are fine for most agents.

## `register.py`: call `instrument()`

The core `instrument()` sets up HTTP-client, OpenAI SDK, and threading instrumentation plus the DataRobot OTel SDK bootstrap. The NAT exporter only carries NAT's own spans. To also route framework auto-instrumentor spans (CrewAI / LangChain / LlamaIndex) to DataRobot, call the matching framework `instrument()` at module-import time in your agent's `register.py`, before the framework constructs any agents:

```python
from datarobot_genai.core.telemetry.agent import instrument
from datarobot_genai.langgraph.telemetry import instrument as langgraph_instrument

instrument()  # HTTP clients + OpenAI SDK + OTel SDK bootstrap
langgraph_instrument()  # framework auto-instrumentor spans
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
| `MLOPS_DEPLOYMENT_ID` **or** `WORKLOAD_ID` | `X-DataRobot-Entity-Id` (auto-prefixed `deployment-<id>` / `workload-<id>`; deployment wins when both are set) | Silent no-op; no spans reach DataRobot. |
| `DATAROBOT_ENDPOINT` (or `DATAROBOT_PUBLIC_API_ENDPOINT`) | endpoint base; `/otel/v1/traces` appended | Silent no-op; no spans reach DataRobot. |

Two caveats regardless of which path supplies the endpoint/headers:

- The `instrument()` SDK bootstrap (framework / `datarobot_otel_conventions` spans) is gated on `MLOPS_DEPLOYMENT_ID` or `WORKLOAD_ID`. Outside a deployment, call `bootstrap_otel_provider_for_datarobot()` yourself (see [Local tracing](#local-tracing)).
- Optional: set `OTEL_SERVICE_NAME` to override the resource `service.name` used by the SDK bootstrap (the NAT exporter uses `project` from the YAML instead).

## Local tracing

A notebook or script has no deployment or workload to attribute spans to, so name a use case, which
the ingest addresses as an `experiment_container`. The endpoint still comes from `DATAROBOT_ENDPOINT`:

```python
from datarobot_genai.core.telemetry.datarobot_otel import bootstrap_otel_provider_for_datarobot

use_case_id = "<a use case id>"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = (
    f"X-DataRobot-Api-Key={os.environ['DATAROBOT_API_TOKEN']},"
    f"X-DataRobot-Entity-Id=experiment_container-{use_case_id}"
)
bootstrap_otel_provider_for_datarobot()
```

Then call `instrument()` and the framework `instrument()` as above. View the traces with the
[`dr xp` plugin](https://docs.datarobot.com/en/docs/agentic-ai/cli/experimentation-plugin.html),
after `dr auth login` in that terminal:

```bash
dr xp --entity-id <use-case-id>
```

## Moderation metrics

When your agent runs the `datarobot_moderation` middleware, `datarobot_dome` can also export
guardrail **metrics** over OTel, alongside the spans above. This works for agents running as
DataRobot **workloads**, not just deployments.

It is opt-in: set `OTEL_EXPORTER_OTLP_ENDPOINT` to the `<host>/otel` base. That single variable
switches on both halves — `datarobot_dome` creates its OTel metric session, and `instrument()`
installs the global `MeterProvider` those instruments record into. With the variable unset, both
stay off and nothing is exported.

| | |
|---|---|
| **Turn on with** | `OTEL_EXPORTER_OTLP_ENDPOINT=<host>/otel` (the same base the traces path uses) |
| **What is exported** | `datarobot.moderations.*` counters, gauges, and histograms — one per guard metric |
| **Attributed to** | the same entity as spans: `workload-<id>` or `deployment-<id>`, deployment winning when both env vars are set |
| **Export cadence** | `OTEL_METRIC_EXPORT_INTERVAL` (standard OTel variable; the SDK default is 60s) |
| **Ingest path** | `<host>/otel/v1/metrics`, appended by the OTLP exporter |

DataRobot **custom metrics** are a separate mechanism and remain deployment-only: they upload
through a deployment-scoped route that needs `MLOPS_DEPLOYMENT_ID`, and `datarobot_dome` has no
workload equivalent. In a workload, OTel metrics are the moderation-metrics path.

## Troubleshooting

- **Data Exploration tab is empty**: Confirm the export is configured — `OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_EXPORTER_OTLP_HEADERS` (primary) or the `DATAROBOT_*` fallback. Both span paths silently skip when neither supplies an endpoint and headers.
- **NAT lifecycle spans appear but framework spans don't**: the framework `instrument()` (e.g. `datarobot_genai.langgraph.telemetry.instrument`) was not called, or was called after the framework imported. Move the call to the top of `register.py`.
- **Framework or memory spans appear in a separate trace from workflow spans**: confirm `datarobot_otelcollector` is enabled in `workflow.yaml` and `instrument()` is called in `register.py` before the framework imports. The exporter bridges NAT context into the SDK and the bootstrap wraps the global `TracerProvider` so LangChain/LangGraph, HTTP `POST`, and memory spans share the active workflow trace.
- **Spans stop arriving after setting `extra_headers`**: an `X-DataRobot-Entity-Id` override must keep the `deployment-` / `workload-` / `experiment_container-` prefix. Nothing validates it locally, so a malformed value fails silently at the ingest.
- **Moderation metrics don't appear**: `OTEL_EXPORTER_OTLP_ENDPOINT` is not set. Both `datarobot_dome` and the `instrument()` meter-provider bootstrap read that variable and stay off without it — see [Moderation metrics](#moderation-metrics).
