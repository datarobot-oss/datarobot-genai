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

from dragent_tests.helpers import E2E_ROOT
from dragent_tests.helpers import OTEL_EXPORTER_OTLP_ENDPOINT
from dragent_tests.helpers import OTEL_EXPORTER_OTLP_HEADERS
from dragent_tests.helpers import agent_dir
from dragent_tests.helpers import assert_tracing_conventions
from dragent_tests.helpers import build_chat_completion
from dragent_tests.helpers import spawn_runner
from dragent_tests.helpers import workflow_file
from dragent_tests.mock_otel_collector import MockOtelCollector

RUNNER_SCRIPT = E2E_ROOT / "dragent" / "run_agent.py"


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

    # THEN: the inline run exported convention spans with the DR auth headers
    assert_tracing_conventions(otel_collector, prompt)


def test_inline_runner_script_is_packaged_with_tests() -> None:
    """Guard rail: the runner script must live next to this test module."""
    # GIVEN: this test runs from the e2e-tests package
    # WHEN/THEN: the runner script must exist on disk under dragent_tests/
    assert RUNNER_SCRIPT.exists(), (
        f"Expected {RUNNER_SCRIPT} to exist alongside this test."
    )
