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

"""E2E test: dragent OTel spans reach the DataRobot ingest with auth headers.

Boots the inline dragent subprocess against ``base/workflow.yaml`` (which
enables ``datarobot_otelcollector``), pointed at an in-process mock OTLP/HTTP
collector via env vars. After the subprocess exits, asserts that the mock saw
≥ 1 POST to ``/otel/v1/traces`` carrying the ``X-DataRobot-Api-Key`` and
``X-DataRobot-Entity-Id`` headers, then parses the OTLP payload and checks the
exported spans carry the expected attributes.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from dragent_tests.helpers import AGENT
from dragent_tests.helpers import AGENT_SUPPORTS_TOOL_CALLS_STREAMING
from dragent_tests.helpers import agent_dir
from dragent_tests.helpers import build_chat_completion
from dragent_tests.helpers import spawn_runner
from dragent_tests.helpers import workflow_file
from dragent_tests.mock_otel_collector import MockOtelCollector
from dragent_tests.test_tools import GENERATE_OBJECTID_PROMPT

OTLP_TRACES_PATH = "/otel/v1/traces"
OTLP_PATH = "/otel"
TEST_API_TOKEN = "test-token"
TEST_USE_CASE_ID = "test-use-case-id"
EXPECTED_ENTITY_ID = f"use-case-{TEST_USE_CASE_ID}"
OTEL_EXPORTER_OTLP_HEADERS = (
    f"X-DataRobot-Api-Key={TEST_API_TOKEN},X-DataRobot-Entity-Id={EXPECTED_ENTITY_ID}"
)

# Span attributes that map to the deployment Tracing table columns, per
# https://docs.datarobot.com/en/docs/agentic-ai/agentic-develop/agentic-tracing-code.html#map-spans-and-attributes-to-the-tracing-table
EXPECTED_SPAN_ATTRIBUTES = {
    "gen_ai.prompt",  # Prompt
    "gen_ai.completion",  # Completion
}

# Only expected for frameworks which idenify tool calls in AG-UI events.
# Not expected in: crewai, base, NAT in non-streaming mode
TOOL_NAME = "tool_name"


def _exported_span_attribute_keys(bodies: list[bytes]) -> set[str]:
    """Union of attribute keys across every span in the OTLP export bodies."""
    keys: set[str] = set()
    for body in bodies:
        request = ExportTraceServiceRequest()
        request.ParseFromString(body)
        for resource_spans in request.resource_spans:
            for scope_spans in resource_spans.scope_spans:
                for span in scope_spans.spans:
                    keys.update(attribute.key for attribute in span.attributes)
    return keys


@pytest.mark.parametrize("stream", [True, False])
def test_otel_spans_export_with_datarobot_headers(tmp_path: Path, stream: bool) -> None:
    """Spans leave the dragent process with DR auth headers and expected attributes."""
    # GIVEN: a mock OTLP collector and a chat completion request
    output_path = tmp_path / "output.json"
    chat_completion = build_chat_completion(GENERATE_OBJECTID_PROMPT, stream=stream)

    with MockOtelCollector() as collector:
        env = {
            **os.environ,
            # Use the setup close to Codespaces
            "OTEL_EXPORTER_OTLP_ENDPOINT": collector.endpoint + OTLP_PATH,
            "OTEL_EXPORTER_OTLP_HEADERS": OTEL_EXPORTER_OTLP_HEADERS,
            # TODO (BUZZOK-31396): only necessary because of the bootstrap hack
            # src/datarobot_genai/core/telemetry/agent.py
            "MLOPS_DEPLOYMENT_ID": "test-deployment-id",
        }

        # WHEN: the inline runner is executed against workflow.yaml
        result = spawn_runner(
            chat_completion=chat_completion,
            output_path=output_path,
            custom_model_dir=agent_dir(),
            config_file=workflow_file(),
            env=env,
        )

        assert result.returncode == 0, (
            f"runner failed (exit {result.returncode}).\n{result.stderr}"
        )

        # The bootstrap-installed BatchSpanProcessor flushes on subprocess
        # shutdown (see run_agent._flush_otel_tracer_provider); NAT's exporter
        # flushes via its own lifecycle. A bounded wait covers either path.
        collector.wait_for_requests(n=1, timeout=10.0)

    # THEN: at least one export hit /otel/v1/traces with the DR auth headers
    auth_requests = [
        req
        for req in collector.requests
        if req.path == OTLP_TRACES_PATH
        and req.headers.get("X-DataRobot-Api-Key") == TEST_API_TOKEN
        and req.headers.get("X-DataRobot-Entity-Id") == EXPECTED_ENTITY_ID
    ]
    assert auth_requests, (
        f"Expected ≥ 1 POST to {OTLP_TRACES_PATH} with DR auth headers; "
        f"captured {len(collector.requests)} request(s) at paths "
        f"{[r.path for r in collector.requests]} with header keys "
        f"{[sorted(r.headers.keys()) for r in collector.requests]}."
    )
    assert auth_requests[0].body, "Expected the OTLP request body to be non-empty."

    # THEN: the exported spans carry the expected attributes
    attribute_keys = _exported_span_attribute_keys([req.body for req in auth_requests])
    missing = EXPECTED_SPAN_ATTRIBUTES - attribute_keys
    assert not missing, (
        f"Exported spans missing expected attributes {sorted(missing)}; "
        f"got {sorted(attribute_keys)}."
    )

    if AGENT_SUPPORTS_TOOL_CALLS_STREAMING and not (stream and AGENT == "nat"):
        assert TOOL_NAME in attribute_keys, (
            f"Exported spans missing expected attribute {TOOL_NAME}; "
            f"got {sorted(attribute_keys)}."
        )
