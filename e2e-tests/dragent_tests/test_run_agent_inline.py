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

"""E2E test: spawn the dragent inline runner as a subprocess (mirrors run_agent.py).

The companion script :mod:`dragent.run_agent` plays the role of
``datarobot-user-models``'s ``run_agent.py``: it accepts a chat completion as
JSON on the command line, invokes ``execute_dragent_inline``, and writes the
final aggregated OpenAI ``ChatCompletion`` to disk for the test to read back.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from openai.types.chat import ChatCompletion

from dragent_tests.helpers import AGENT
from dragent_tests.helpers import E2E_ROOT
from dragent_tests.helpers import agent_dir
from dragent_tests.helpers import build_chat_completion
from dragent_tests.helpers import spawn_runner
from dragent_tests.helpers import workflow_file
from dragent_tests.otel_helpers import GEN_AI_PROMPT
from dragent_tests.otel_helpers import OTEL_EXPORTER_OTLP_ENDPOINT
from dragent_tests.otel_helpers import OTEL_EXPORTER_OTLP_HEADERS
from dragent_tests.otel_helpers import SETUP_HTTP_SPAN_URLS
from dragent_tests.otel_helpers import MockOtelCollector
from dragent_tests.otel_helpers import assert_tracing_conventions

RUNNER_SCRIPT = E2E_ROOT / "dragent" / "run_agent.py"

# Name of the span the runner opens around ``execute_dragent_inline`` (mirrors
# how ``datarobot-user-models``'s ``run_agent.py`` roots a trace before invoking
# the agent). See ``dragent/run_agent.py``.
RUN_AGENT_SPAN_NAME = "run_agent"

def test_run_agent_inline(tmp_path: Path, otel_collector: MockOtelCollector) -> None:
    """Inline run produces a valid ``ChatCompletion`` and exports Tracing spans.

    The inline path (``execute_dragent_inline``) is the production custom-model
    entrypoint. The runner subprocess is pointed at the shared mock collector
    and given the ``MLOPS_DEPLOYMENT_ID`` that unlocks the OTel SDK bootstrap,
    so the ``datarobot_otel_conventions`` spans (gen_ai.prompt /
    gen_ai.completion) flush before the subprocess exits — verified inline here
    rather than in a separate (and expensive) extra run.
    """
    # GIVEN: a prompt, an output path, and a runner pointed at the collector
    output_path = tmp_path / "output.json"
    prompt = "Say 'hello world' and nothing else."
    chat_completion = build_chat_completion(prompt)
    env = {
        **os.environ,
        "OTEL_EXPORTER_OTLP_ENDPOINT": OTEL_EXPORTER_OTLP_ENDPOINT,
        "OTEL_EXPORTER_OTLP_HEADERS": OTEL_EXPORTER_OTLP_HEADERS,
        # Unlocks the SDK TracerProvider bootstrap (see instrument()).
        "MLOPS_DEPLOYMENT_ID": "e2e-test",
        # Use LiteLLM's bundled cost map instead of fetching it from GitHub, so
        # the runtime bootstrap does not emit an outbound GET span (mirrors the
        # dragent server env). The DR version check / MCP discovery spans still
        # root their own trace, so SETUP_HTTP_SPAN_URLS below remains required.
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }

    # WHEN: the inline runner is executed as a subprocess
    result = spawn_runner(
        chat_completion=chat_completion,
        output_path=output_path,
        custom_model_dir=agent_dir(),
        config_file=workflow_file(),
        env=env,
    )

    # THEN: the runner exits cleanly and the file parses as a ChatCompletion
    assert result.returncode == 0, f"runner failed (exit {result.returncode}).\n{result.stderr}"
    assert output_path.exists(), f"Expected the runner to write the output file.\n{result.stderr}"

    result_text = output_path.read_text()
    payload = json.loads(result_text)
    completion = ChatCompletion.model_validate(payload)
    assert completion.choices, f"Expected at least one choice.\n{result.stderr}\n{result_text}"
    assert completion.choices[0].message.content, (
        f"Expected non-empty assistant message content.\n{result.stderr}\n{result_text}"
    )

    # THEN: token-count guards attach serialized moderation metadata to the completion
    assert getattr(completion, "datarobot_moderations", None), (
        "Expected datarobot_moderations on inline ChatCompletion when guards are configured"
    )

    # THEN: the inline run exported convention spans with the DR auth headers.
    # Setup-time HTTP client spans (LiteLLM cost map, DR version check, MCP
    # discovery) root their own trace and are ignored for the single-trace check.
    assert_tracing_conventions(
        otel_collector, prompt, framework=AGENT, ignore_span_urls=SETUP_HTTP_SPAN_URLS
    )

    # THEN: the ``run_agent`` span the runner opened around the inline call was
    # exported and shares the agent's trace. The inline path seeds NAT's
    # workflow_trace_id from this active span, so the datarobot_agent span (and
    # its children) join the trace run_agent started instead of a NAT-only one.
    run_agent_spans = otel_collector.wait_for_spans(
        lambda span: span.name == RUN_AGENT_SPAN_NAME
    )
    assert run_agent_spans, (
        f"Expected a {RUN_AGENT_SPAN_NAME!r} span exported from the inline runner; "
        f"observed span names {sorted({s.name for s in otel_collector.spans()})}."
    )

    agent_trace_ids = {
        span.trace_id
        for span in otel_collector.spans()
        if span.attributes.get(GEN_AI_PROMPT) == prompt
    }
    run_agent_trace_ids = {span.trace_id for span in run_agent_spans}
    assert run_agent_trace_ids == agent_trace_ids, (
        f"Expected the {RUN_AGENT_SPAN_NAME!r} span to share the agent's trace_id, but "
        f"run_agent trace(s)={sorted(t.hex() for t in run_agent_trace_ids)} vs "
        f"agent trace(s)={sorted(t.hex() for t in agent_trace_ids)}."
    )


def test_inline_runner_script_is_packaged_with_tests() -> None:
    """Guard rail: the runner script must live next to this test module."""
    # GIVEN: this test runs from the e2e-tests package
    # WHEN/THEN: the runner script must exist on disk under dragent_tests/
    assert RUNNER_SCRIPT.exists(), (
        f"Expected {RUNNER_SCRIPT} to exist alongside this test."
    )
