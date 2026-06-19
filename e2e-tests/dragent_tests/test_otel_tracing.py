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

Boots the inline dragent subprocess against ``base/workflow-tracing.yaml`` (which
enables ``datarobot_otelcollector``), pointed at an in-process mock OTLP/HTTP
collector via env vars. After the subprocess exits, asserts that the mock saw
≥ 1 POST to ``/otel/v1/traces`` carrying the ``X-DataRobot-Api-Key`` and
``X-DataRobot-Entity-Id`` headers — the precise contract a real DataRobot OTel
ingest checks. This is the e2e equivalent of "deploy → chat → look at the
Tracing tab" that we have been verifying by hand.
"""

from __future__ import annotations

import os
from pathlib import Path

from dragent_tests.helpers import WORKFLOW_FILE
from dragent_tests.helpers import agent_dir
from dragent_tests.helpers import build_chat_completion
from dragent_tests.helpers import spawn_runner
from dragent_tests.mock_otel_collector import MockOtelCollector

OTLP_TRACES_PATH = "/otel/v1/traces"
TEST_API_TOKEN = "test-token"
TEST_DEPLOYMENT_ID = "test-deployment-id"
EXPECTED_ENTITY_ID = f"deployment-{TEST_DEPLOYMENT_ID}"


def test_otel_spans_export_with_datarobot_headers(tmp_path: Path) -> None:
    """Spans leave the dragent process with DR auth headers and the right path."""
    # GIVEN: a mock OTLP collector and a chat completion request
    output_path = tmp_path / "output.json"
    chat_completion = build_chat_completion()

    with MockOtelCollector() as collector:
        env = {
            **os.environ,
            "DATAROBOT_ENDPOINT": collector.endpoint,
            "DATAROBOT_API_TOKEN": TEST_API_TOKEN,
            "MLOPS_DEPLOYMENT_ID": TEST_DEPLOYMENT_ID,
        }

        # WHEN: the inline runner is executed against workflow-tracing.yaml
        result = spawn_runner(
            chat_completion=chat_completion,
            output_path=output_path,
            custom_model_dir=agent_dir(),
            config_file=WORKFLOW_FILE,
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
