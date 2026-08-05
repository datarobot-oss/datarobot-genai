# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import httpx
from ag_ui.core import EventType
from datarobot_genai.core.agents.verify import validate_sequence

from dragent_tests.helpers import AGENT
from dragent_tests.helpers import collect_ag_ui_events
from dragent_tests.helpers import collect_text
from dragent_tests.helpers import make_generate_payload
from dragent_tests.helpers import raise_if_nat_workflow_error_payload
from dragent_tests.helpers import stream_sse_responses
from dragent_tests.otel_helpers import SETUP_HTTP_SPAN_URLS
from dragent_tests.otel_helpers import MockOtelCollector
from dragent_tests.otel_helpers import assert_tracing_conventions


def test_generate_streaming(
    http_client: httpx.Client, otel_collector: MockOtelCollector
) -> None:
    """Concatenated text deltas produce a non-empty moderated stream that finishes cleanly."""
    # GIVEN: a payload that requests "Say 'hello world' and nothing else."
    prompt = "Say 'hello world' and nothing else."
    payload = make_generate_payload(prompt)

    # WHEN: the payload is streamed to the generate endpoint
    # (stream_sse_responses fails on bare NAT workflow_error JSON lines)
    sse_responses = stream_sse_responses(http_client, payload)

    # THEN: the response contains AG-UI events
    ag_ui_events = collect_ag_ui_events(sse_responses)

    # THEN: no AG-UI RUN_ERROR (workflow failures should already have been
    # raised by stream_sse_responses; this catches error events framed as SSE)
    run_errors = [e for e in ag_ui_events if e.type == EventType.RUN_ERROR]
    assert not run_errors, f"Unexpected RUN_ERROR in moderated stream: {run_errors!r}"

    # THEN: the events are a valid AG-UI sequence
    validate_sequence(ag_ui_events)

    # THEN: there are events with text
    full_text = collect_text(ag_ui_events)
    assert len(full_text) > 0, "Expected non-empty text response"

    # THEN: token-count guards attach serialized moderation metadata to at least one chunk
    assert any(
        chunk.datarobot_moderations for chunk in sse_responses
    ), "Expected streamed chunks to include datarobot_moderations when guards are configured"

    # THEN: the streaming run exported DataRobot Tracing-table spans
    # (gen_ai.prompt / gen_ai.completion) to the OTel ingest with DR auth headers.
    assert_tracing_conventions(
        otel_collector, prompt, framework=AGENT, ignore_span_urls=SETUP_HTTP_SPAN_URLS
    )
