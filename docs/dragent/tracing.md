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
- **Framework auto-instrumentor spans** — spans emitted by `opentelemetry-instrumentation-crewai`, `-langchain`, `-llamaindex`, and `-openai`. Enabled by calling `instrument(framework=...)` from your own code.
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
| `datarobot_entity_id` | no | `deployment-<MLOPS_DEPLOYMENT_ID>` | Sent as the `X-DataRobot-Entity-Id` header. Non-empty values must keep the `deployment-` prefix. |
| `extra_headers` | no | `{}` | Additional headers; keys here win on collision with the DataRobot defaults. |
| `resource_attributes` | no | `{}` | Extra OTel resource attributes; keys here win on collision. |

Batch-tuning knobs (`batch_size`, `flush_interval`, `max_queue_size`, etc.) are inherited from NAT's `BatchConfigMixin`; defaults are fine for most agents.

## `register.py`: call `instrument()`

The NAT exporter only carries NAT's own spans. To also route framework auto-instrumentor spans (CrewAI / LangChain / LlamaIndex / OpenAI) to DataRobot, call `instrument(framework=...)` at module-import time in your agent's `register.py`, before the framework constructs any agents:

```python
from datarobot_genai.core.telemetry.agent import instrument

instrument(framework="langgraph")  # "crewai" | "langgraph" | "llamaindex" | "nat" | None
```

Accepted `framework` values are `"crewai"`, `"langgraph"`, `"llamaindex"`, `"nat"`, or `None`. `"nat"` is shorthand for all three framework instrumentors. Passing `None` instruments only HTTP clients and the OpenAI SDK.

`instrument()` is idempotent — repeat calls are no-ops — and safe to keep in `register.py` during local development: when the DataRobot deployment environment variables below are not all set, the underlying `bootstrap_otel_provider_for_datarobot()` silently skips installing the SDK provider, so framework spans simply go nowhere instead of erroring.

## Required environment

The runtime reads the same environment variables from both sides (the NAT exporter and the SDK bootstrap). Inside a DataRobot deployment they are populated for you; locally you set them yourself.

| Variable | Description | Missing → |
|---|---|---|
| `DATAROBOT_API_TOKEN` | API token used as `X-DataRobot-Api-Key`. | Silent no-op; no spans reach DataRobot. |
| `MLOPS_DEPLOYMENT_ID` | Deployment ID; auto-prefixed to form `X-DataRobot-Entity-Id`. | Silent no-op; no spans reach DataRobot. |
| `DATAROBOT_ENDPOINT` (or `DATAROBOT_PUBLIC_API_ENDPOINT`) | Base URL; `/otel/v1/traces` is appended. | Silent no-op; no spans reach DataRobot. |

Optional override: set `OTEL_SERVICE_NAME` to override the resource `service.name` used by the SDK bootstrap (the NAT exporter uses `project` from the YAML instead).

## Verifying locally

The repo ships a minimal reproducer at [`e2e-tests/dragent/base/workflow-tracing.yaml`](../../e2e-tests/dragent/base/workflow-tracing.yaml). The companion test [`e2e-tests/dragent_tests/test_otel_tracing.py`](../../e2e-tests/dragent_tests/test_otel_tracing.py) spawns a real dragent subprocess against an in-process mock OTLP collector and asserts that POSTs reach `/otel/v1/traces` carrying the `X-DataRobot-Api-Key` and `X-DataRobot-Entity-Id` headers. Use it as a template for your own integration checks.

## Troubleshooting

- **Data Exploration tab is empty**: Confirm the three environment variables in the table above are set in the deployment. Both sides silently skip when any is missing.
- **NAT lifecycle spans appear but framework spans don't**: `instrument(framework=...)` was not called, or was called after the framework imported. Move the call to the top of `register.py`.
- **Framework or memory spans appear in a separate trace from workflow spans**: confirm `datarobot_otelcollector` is enabled in `workflow.yaml` and `instrument()` is called in `register.py` before the framework imports. The exporter bridges NAT context into the SDK and the bootstrap wraps the global `TracerProvider` so LangChain/LangGraph, HTTP `POST`, and memory spans share the active workflow trace.
- **`datarobot_entity_id must be of the form 'deployment-<id>'`**: You set `datarobot_entity_id` manually without the `deployment-` prefix. Either add the prefix or omit the field inside a deployment — it auto-derives from `MLOPS_DEPLOYMENT_ID`.
