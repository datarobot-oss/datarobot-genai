# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic builders for the two OTel trace populations under test.

Built, not checked in: the real thing is a 4.76 MB blob, and a generator is
cheap, reproducible and CI-safe. No RNG at all — every string is produced by
cycling a fixed sentence pool and sliced to an exact length, so byte-identity and
every char count below are stable across runs, machines and Python versions.

Two populations, both drawn from the measurements in APP-6967:

* ``build_oversized_agent_trace`` — the population this ticket exists for. 12
  spans, 1,022,000 chars of payload (~330k tokens at the measured 3.1 chars per
  token), containing byte-identical Traceloop/gen_ai twin pairs and one single
  attribute value of 160,000 chars (~52k tokens on its own).
* ``build_small_trace`` — the other population, ~400 tokens end to end, to prove
  the summary projection does not mangle the easy case.

The char-count constants are the declared intent of the payload plan. Tests
assert the truncation module's accounting against them, so editing the plan means
editing the constant — which is the point.
"""

from typing import Any

# Measured on the real trace JSON in APP-6967: ~3.1 chars per token. Every token
# figure in this module and its tests is derived from this ratio rather than a
# tokenizer, because drtools may not take a tokenizer dependency.
CHARS_PER_TOKEN = 3.1

# Payload plan of the oversized fixture, as built below.
OVERSIZED_SPAN_COUNT = 12
OVERSIZED_TOTAL_PAYLOAD_CHARS = 1_022_000
OVERSIZED_CANONICAL_PAYLOAD_CHARS = 393_500
OVERSIZED_GIANT_ATTRIBUTE_CHARS = 160_000

OVERSIZED_TRACE_ID = "9f2c41ab7d0e4c8b93a5e17fd6b04c2a"

# Small-population ceiling. The fixture measures ~450 tokens end to end rather
# than exactly 400: SpanViewValidator makes trace_id, events and links required on
# every span, which costs ~50 chars a span that a real response also pays. Keeping
# the wire shape faithful is worth 50 tokens.
SMALL_TRACE_ID = "3b71d0e6c4a94f2280ab5c19e7d3f068"
SMALL_TRACE_MAX_TOKENS = 500

_AGENT_SERVICE = "deployment-6889f1c4a2b7d3e5f0a91c22"
_WORKLOAD_SERVICE = "workload-5f10cb73a9e24d8c1b06f4a7"
_SPAN_ID_BASE = 0x7F3A9C1D00000000
_START_TIME = 1756_339_201.418

_SENTENCES = (
    "The customer asked whether the Q3 renewal quote already includes the "
    "negotiated multi-year discount, and if not, what the corrected total is.",
    "Tool result: 14 rows returned from deployments_list, of which 3 are in an "
    "errored state and 1 has been stuck in provisioning for 41 minutes.",
    "I should check the accuracy drift report before recommending a retrain, "
    "because a data-quality issue upstream would explain both symptoms.",
    "Assistant: based on the prediction explanations, the top three drivers are "
    "tenure_months, monthly_charges and contract_type, in that order.",
    "Observation: the span named drum.chat.completions.stream carries the full "
    "completion, so the parent span's copy is redundant for this analysis.",
    "The retrieved document says feature discovery runs before Autopilot and may "
    "add derived features from secondary datasets registered to the use case.",
    "Error: HTTPError 429 from the LLM gateway after 3 retries with exponential "
    "backoff; falling back to the smaller model for this turn only.",
    "Plan: summarize the failing traces, group them by root_span_name, then pull "
    "the payload of the single most expensive span for a closer look.",
)


def build_oversized_agent_trace() -> dict[str, Any]:
    """Build the oversized agent trace: 12 spans, 1,022,000 payload chars.

    Deliberately carries every shape the truncation module has to survive:

    * a byte-identical Traceloop/gen_ai twin pair on the input side and another
      on the output side of one llm.chat span, plus an OpenInference twin of the
      same text and an indexed ``gen_ai.completion.0.content`` copy;
    * a tool span whose 160,000-char output is written four times over by four
      different instrumentations — the single-attribute case that no per-field cap
      alone can rescue;
    * derived families that are *not* duplicated anywhere (``nat.metadata``,
      ``input.value_obj``), so the semconv bucket is exercised separately from
      the duplicate bucket;
    * spans with no payload at all, and one ERROR span with a status message.
    """
    twin_prompt = _text("conversation-transcript", 40_000)
    twin_completion = _text("assistant-turn", 8_000)
    giant_tool_output = _text("crm-export", OVERSIZED_GIANT_ATTRIBUTE_CHARS)

    spans = [
        _span(
            index=0,
            parent=None,
            name="agent.run",
            kind="SERVER",
            duration=41.907,
            prompt=_text("user-question", 1_000),
            completion=_text("final-answer", 2_500),
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": "renewal-assistant",
                "gen_ai.usage.input_tokens": "38211",
                "gen_ai.usage.output_tokens": "1804",
            },
        ),
        _span(
            index=1,
            parent=0,
            name="agent.plan",
            kind="INTERNAL",
            duration=0.311,
            attributes={"nat.function.name": "plan", "nat.event.type": "SPAN_START"},
        ),
        _span(
            index=2,
            parent=0,
            name="llm.chat",
            kind="CLIENT",
            duration=8.204,
            prompt=twin_prompt,
            completion=twin_completion,
            attributes={
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.response.finish_reasons": "tool_calls",
                # Twins: same bytes, four different instrumentations.
                "gen_ai.task.input": twin_prompt,
                "traceloop.entity.input": twin_prompt,
                "input.value": twin_prompt,
                "gen_ai.task.output": twin_completion,
                "traceloop.entity.output": twin_completion,
                "gen_ai.completion.0.content": twin_completion,
                # Derived, but duplicated nowhere: the semconv bucket.
                "input.value_obj": _text("input-value-obj", 3_000),
                "nat.metadata": _text("nat-metadata", 1_500),
            },
        ),
        _span(
            index=3,
            parent=0,
            name="tool.execute",
            kind="INTERNAL",
            duration=12.663,
            completion=giant_tool_output,
            attributes={
                "gen_ai.tool.name": "crm_account_export",
                "gen_ai.tool.call.id": "call_7Kq1",
                "traceloop.entity.output": giant_tool_output,
                "gen_ai.task.output": giant_tool_output,
                "input.value": giant_tool_output,
            },
        ),
        _span(
            index=4,
            parent=0,
            name="tool.execute",
            kind="INTERNAL",
            duration=1.084,
            completion=_text("deployment-list", 18_000),
            attributes={"gen_ai.tool.name": "deployments_list"},
        ),
        _span(
            index=5,
            parent=0,
            name="llm.chat",
            kind="CLIENT",
            duration=6.771,
            prompt=_text("conversation-transcript-2", 55_000),
            completion=_text("assistant-turn-2", 9_000),
            attributes={"gen_ai.system": "openai", "gen_ai.request.model": "gpt-4o"},
        ),
        _span(
            index=6,
            parent=0,
            name="tool.execute",
            kind="INTERNAL",
            duration=0.492,
            completion=_text("drift-report", 9_500),
            attributes={"gen_ai.tool.name": "deployment_drift_get"},
        ),
        _span(
            index=7,
            parent=0,
            name="llm.chat",
            kind="CLIENT",
            status_code="ERROR",
            status_message="RateLimitError: 429 Too Many Requests after 3 retries",
            duration=30.008,
            prompt=_text("conversation-transcript-3", 60_000),
            completion=_text("truncated-assistant-turn", 500),
            attributes={"gen_ai.system": "openai", "gen_ai.request.model": "gpt-4o"},
        ),
        _span(
            index=8,
            parent=0,
            name="guardrail.check",
            kind="INTERNAL",
            duration=0.203,
            attributes={"datarobot.moderation.name": "prompt_injection", "score": "0.02"},
        ),
        _span(
            index=9,
            parent=4,
            name="db.query",
            kind="CLIENT",
            duration=0.148,
            attributes={"db.system": "postgresql", "db.operation.name": "SELECT"},
        ),
        _span(
            index=10,
            parent=3,
            name="http.post",
            kind="CLIENT",
            duration=11.902,
            attributes={"http.request.method": "POST", "http.response.status_code": "200"},
        ),
        _span(
            index=11,
            parent=0,
            name="agent.finalize",
            kind="INTERNAL",
            duration=2.117,
            completion=_text("final-report", 30_000),
            attributes={"nat.function.name": "finalize"},
        ),
    ]

    return _trace(
        trace_id=OVERSIZED_TRACE_ID,
        spans=spans,
        duration=41.907,
        root_span_name="agent.run",
        service_name=_AGENT_SERVICE,
        metrics={
            "prompt_guards": {"prompt_injection": {"average": 0.02, "count": 4}},
            "response_guards": {"toxicity": {"average": 0.0, "count": 4}},
        },
    )


def build_small_trace() -> dict[str, Any]:
    """Build the small non-agentic trace: 3 spans, no payload, ~400 tokens total."""
    spans = [
        _span(
            index=0,
            parent=None,
            name="GET /health",
            kind="SERVER",
            duration=0.104,
            service_name=_WORKLOAD_SERVICE,
            attributes={"http.response.status_code": "200"},
        ),
        _span(
            index=1,
            parent=0,
            name="db.query",
            kind="CLIENT",
            duration=0.041,
            service_name=_WORKLOAD_SERVICE,
            attributes={"db.system": "postgresql"},
        ),
        _span(
            index=2,
            parent=0,
            name="cache.get",
            kind="CLIENT",
            duration=0.002,
            service_name=_WORKLOAD_SERVICE,
            attributes={"cache.hit": "true"},
        ),
    ]

    return _trace(
        trace_id=SMALL_TRACE_ID,
        spans=spans,
        duration=0.104,
        root_span_name="GET /health",
        service_name=_WORKLOAD_SERVICE,
    )


def _text(label: str, target_chars: int) -> str:
    """Build exactly ``target_chars`` chars of stable, realistic-looking prose."""
    if target_chars <= 0:
        return ""
    lines: list[str] = []
    length = 0
    index = 0
    while length <= target_chars:
        line = f"[{label} {index:05d}] {_SENTENCES[index % len(_SENTENCES)]}"
        lines.append(line)
        length += len(line) + 1
        index += 1
    return "\n".join(lines)[:target_chars]


def _span(
    *,
    index: int,
    parent: int | None,
    name: str,
    kind: str,
    duration: float,
    attributes: dict[str, str],
    status_code: str = "OK",
    status_message: str | None = None,
    service_name: str = _AGENT_SERVICE,
    prompt: str | None = None,
    completion: str | None = None,
) -> dict[str, Any]:
    """Build one span in the shape SpanViewValidator serializes."""
    span: dict[str, Any] = {
        "trace_id": OVERSIZED_TRACE_ID,
        "span_id": f"{_SPAN_ID_BASE + index:016x}",
        "parent_span_id": None if parent is None else f"{_SPAN_ID_BASE + parent:016x}",
        "name": name,
        "status_code": status_code,
        "status_message": status_message,
        "kind": kind,
        "service_name": service_name,
        "duration": duration,
        "start_time": _START_TIME + index,
        "attributes": dict(attributes),
        "events": [],
        "links": [],
    }
    if prompt is not None:
        span["prompt"] = prompt
    if completion is not None:
        span["completion"] = completion
    return span


def _trace(
    *,
    trace_id: str,
    spans: list[dict[str, Any]],
    duration: float,
    root_span_name: str,
    service_name: str,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap spans in the trace-detail response envelope, pagination fields included."""
    for span in spans:
        span["trace_id"] = trace_id
    trace: dict[str, Any] = {
        "trace_id": trace_id,
        "span_count": len(spans),
        "duration": duration,
        "root_span_name": root_span_name,
        "root_service_name": service_name,
        "spans": spans,
        "count": len(spans),
        "offset": 0,
        "limit": 100,
        "total_count": len(spans),
        "next": None,
        "previous": None,
    }
    if metrics is not None:
        trace["metrics"] = metrics
    return trace
