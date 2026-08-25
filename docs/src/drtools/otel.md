# OTel observability tools (`drtools.otel`)

Seven read-only tools (`otel_*`) that let an agent inspect a DataRobot entity's
OpenTelemetry traces, logs, and metrics — the same data the platform's tracing
UI renders, sized for a model's context window instead of a browser.

Enable them with:

```bash
export ENABLE_OTEL_TOOLS=true
```

They ship in `datarobot-genai[drtools]` / `datarobot-genai[drmcp]` and are
registered under the `dr_otel` category (label **Observability**), a
standalone top-level category alongside `dr_documentation`, `dr_use_cases`,
and `dr_deployments`.

## Why this package exists

A raw OTel trace is not something you can hand to a model. Measured against
real agent traffic:

| Percentile | Tokens |
| --- | --- |
| p50 | 1,129 |
| p95 | **809,639** |
| max | **1,560,241** |

Span count does not predict this — a 37-span trace measured 1.56M tokens while
a 32-span trace measured 82k (an 86× spread per-span). So there is no safe
client-side guard based on the trace summary alone: every tool in this
package projects its response instead of passing the upstream payload
through.

`otel_trace_get`'s default `view="summary"` is the load-bearing example: on
that same 1.56M-token trace, the summary projection costs **4,936 tokens
(0.3%)** — a flat span tree with per-span `payload_chars`/`payload_fields`
accounting, but zero payload text. That is what makes a summary → drill-down
workflow possible at all: an agent sees *where the mass is* for about 133
tokens/span, then fetches only the one span's payload it actually needs.

## The tool set

| Tool | Endpoint | Purpose |
| --- | --- | --- |
| `otel_traces_list` | `GET traces/` | List traces for an entity — the default entry point. |
| `otel_trace_get` | `GET traces/{traceId}/` | One trace's span tree: `view="summary"` (default, no payloads) or `view="payloads"` (canonical attributes, capped and budgeted). |
| `otel_span_payload_get` | same endpoint, span-scoped | Drill-down escape hatch: one span's full payload text, windowed field by field. |
| `otel_logs_list` | `GET logs/` | List OTel log lines, filterable by level, trace/span ID, and text. |
| `otel_metrics_catalog_list` | `GET metrics/summary/` | Discover which metrics an entity emits. |
| `otel_metrics_values_get` | `GET metrics/values/` or `metrics/autocollectedValues/` | Current metric values — built-in platform metrics by default, or configured custom metrics via `source="configured"`. |
| `otel_entity_stats_get` | `GET /otel/stats/` | Preflight: does this entity have any OTel data, and can the caller read it? |

Every entity-scoped tool takes the same `entity_type` / `entity_id` pair:

```python
entity_type: Literal[
    "deployment", "use_case", "experiment_container", "custom_application",
    "workload", "execution_environment", "custom_job", "artifact",
]
entity_id: str  # 24-character hex ID
```

Entity resolution is deliberately out of scope for this package — resolve an
entity first with `workload_list`, `deployment_get_info`, or the user's own
`.env`, then pass it in here.

## The summary → drill-down flow

The intended path through the tool set, in the order an agent should reach
for these:

1. **`otel_traces_list`** — find the trace you care about (e.g.
   `status="error"`). Flat ~185 tokens/trace regardless of the underlying
   trace's size, because this endpoint hard-truncates `completion` to 512
   chars server-side and carries no span attributes at all.
2. **`otel_trace_get(view="summary")`** (the default) — the span tree for one
   trace, ~133 tokens/span, with `payload_chars` and `payload_fields` per span
   telling you where the payload mass actually is, without paying for any of
   it.
3. **`otel_span_payload_get`** — once you know which span matters, fetch its
   payload text directly. Every dropped duplicate/derived field is *named*,
   not silently discarded (`dropped_as_duplicate` / `dropped_semconv`), and
   `field_offset` pages within a single field — one measured attribute was
   740,000 characters, well past what any single call can return in full.

`otel_trace_get(view="payloads")` exists as a middle ground — canonical
semconv attributes only, each capped and budgeted across the whole span page
— but even that view can overshoot a 20k-token target on the worst traces
(measured 64,641 tokens / 4.1% on the largest trace). It is not the default,
and its own tool description says so: prefer `view="summary"` plus a targeted
`otel_span_payload_get` call over `view="payloads"`.

### Why dedup alone is not enough

64.5% of the worst measured trace's payload was byte-identical duplication —
Traceloop, NAT, OpenInference, and `gen_ai.*` semantic conventions each
independently write the same prompt/completion text. `canonical_attributes()`
(in `drtools/otel/truncation.py`) drops those duplicates and every
derived/non-canonical field, keeping the highest-precedence surviving copy:

1. the response's own `prompt` / `completion` fields (never dropped)
2. `gen_ai.*` (the OTel semantic conventions)
3. `openinference` `input.value` / `output.value`
4. `traceloop.*`
5. `nat.*`

**Even after perfect dedup, 423,300 tokens of genuinely unique text remain**
on that trace — still roughly 2× a context window. Dedup narrows the problem;
only summary-by-default plus a hard character budget actually bounds it. This
is why `apply_char_budget()` is a hard stop (spans are emitted in order until
the budget is spent, then dropped and reported via `spans_dropped`/
`spans_returned`), not a "try to fit, then give up" cap.

## Token budgets are character budgets

`drtools` may import only `drtools` and `drmcputils` — no `fastmcp`, no
tokenizer dependency (specifically no `tiktoken`). Every truncation limit in
this package is a **character** budget, using a measured conversion of
**3.1 chars/token** on this JSON shape. `otel_trace_get`'s `max_total_chars`
default of 60,000 is chosen to land near 20k tokens under that ratio. Every
budget is a plain `int` argument on the tool call, so an agent that finds a
default wrong for its use case can pass a different one on the next call.

> **PROVISIONAL:** `otel_trace_get`'s three numeric defaults —
> `span_limit=100`, `max_field_chars=2000`, `max_total_chars=60000` — were set
> from this proxy measurement, not from a run against Claude's own tokenizer
> or a real trace. Plan §9 step 9 (a manual run against a real agent
> deployment) may correct them after this release; check
> `src/datarobot_genai/drtools/otel/traces.py`'s `DEFAULT_TRACE_*` constants
> for the current values if this note has gone stale.

## 403s: configuration, not missing data

Every `/otel` route enforces its own combination of feature flags and
licenses, and a 403 from one of these tools almost always means
*configuration*, not "no access to this entity" or "no data here":

| Cause | Scope |
| --- | --- |
| `GENAI_EXPERIMENTATION` feature flag | Required by every `/otel` route **except** `otel_entity_stats_get` (see below). |
| `AGENTIC_PREDICTIVE_GOVERNANCE_BUILDER` seat license | Required by all trace routes (`otel_traces_list`, `otel_trace_get`, `otel_span_payload_get`). |
| `MMM_DISABLE_OTEL_TRACING_VIEWING_FOR_DEPLOYMENT_AND_CUSTOM_APPS` eng config | Kills trace viewing specifically for `deployment` and `custom_application` entities. |
| `MLOPS_ADVANCED_TRACE_ANALYSIS` | Phase-2 endpoints only (span histograms, anomaly scores, dependency graphs) — not shipped in this release. |
| `FUTURE_PUBLIC_API_DOCS` | Phase-2 `traces/attributes/` only — not shipped in this release. |

`otel_entity_stats_get` is the one exception: `OtelStatsController` requires
neither the feature flag nor the seat license, so it still answers on a
cluster where every other tool here would 403. Use it as a preflight check —
"does this entity have data, and can I read it" — before paging traces on a
cluster you're not sure is configured for it. One caveat baked into that
tool's design: a 403 from `otel_entity_stats_get` itself *is* reported as a
real permissions failure (`ToolError`, not an empty result), because unlike
the other tools this one has no configuration gate to blame it on.

## Connecting a coding harness to a local `drmcp` server

To let a coding agent (Claude Code, `dr opencode`, or any other MCP-aware
harness) call these tools against your own DataRobot account, run a `drmcp`
server locally and point the harness's MCP client at it.

**1. Start the server** with the OTel tools enabled:

```bash
export ENABLE_OTEL_TOOLS=true
export DATAROBOT_API_TOKEN=YOUR_DATAROBOT_API_TOKEN
export DATAROBOT_ENDPOINT=https://app.datarobot.com/api/v2
task drmcp-dev
```

This serves streamable-HTTP on `http://localhost:8080/mcp` by default
(`MCP_SERVER_PORT` / `MCP_SERVER_HOST` override the port/host; see
`MCPServerConfig` in `drmcp/core/config.py`). `task drmcp-test-dev-server-background`
(used by this repo's own acceptance suite) starts the same server on
`:8652` with every tool package — including OTel — pre-enabled, if you'd
rather reuse that instead of setting the flags above by hand.

**2. Point the harness at it.** Most MCP-aware coding harnesses read an
`.mcp.json` in the project root (or a user-level config) describing remote
servers. For a streamable-HTTP server like this one:

```json
{
  "mcpServers": {
    "drmcp-local": {
      "type": "http",
      "url": "http://localhost:8080/mcp",
      "headers": {
        "Authorization": "Bearer ${DATAROBOT_API_TOKEN}"
      }
    }
  }
}
```

The `Authorization: Bearer <token>` header is how `drtools`' default
`AUTH_RESOLUTION_STRATEGY=http` resolves credentials per request (see
[auth.md](auth.md)) — the harness sends it on every MCP call, and the server
never needs its own copy of your token. Check your harness's own docs for
exactly where it expects this file (some accept an equivalent `--mcp-server`
CLI flag or a `stdio` launcher instead of `http`); the `url`/`headers` shape
above is the part specific to this server.

**3. Confirm it's live.** Once connected, the harness's tool listing should
include `otel_traces_list`, `otel_trace_get`, `otel_span_payload_get`,
`otel_logs_list`, `otel_metrics_catalog_list`, `otel_metrics_values_get`, and
`otel_entity_stats_get`, along with every other package the server has
enabled.

For scripted/CI use rather than an interactive coding harness, the
`DR_MCP_SERVER_URL` environment variable is the equivalent knob this repo's
own test harness reads (`drmcp/test_utils/mcp_utils_ete.py`,
`test_interactive.py`) — point it at a running server's `/mcp` URL the same
way, e.g. `DR_MCP_SERVER_URL=http://localhost:8080/mcp`.

## See also

- [`drtools/README.md`](README.md) — package index.
- [`auth.md`](auth.md) — how `drtools` resolves credentials per request.
- `src/datarobot_genai/drtools/otel/` — source: `traces.py`, `logs.py`,
  `metrics.py`, `entity_stats.py`, `truncation.py`, `constants.py`.
