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
from pathlib import Path

import pytest
from openai.types.chat import ChatCompletion

from dragent_tests.helpers import ALL_TEST_CASES
from dragent_tests.helpers import E2E_ROOT
from dragent_tests.helpers import agent_dir
from dragent_tests.helpers import build_chat_completion
from dragent_tests.helpers import spawn_runner

RUNNER_SCRIPT = E2E_ROOT / "dragent" / "run_agent.py"
AGENT_DIR = agent_dir()
WORKFLOW_CONFIG = AGENT_DIR / "workflow.yaml"

if not ALL_TEST_CASES:
    pytest.skip(
        "Running minimal test set for non-LLM Gateway LLM, skipping inline runner tests",
        allow_module_level=True,
    )


def test_run_agent_inline(tmp_path: Path) -> None:
    """Inline run produces a valid ``ChatCompletion`` JSON file."""
    # GIVEN: a chat completion request and an output path
    output_path = tmp_path / "output.json"
    chat_completion = build_chat_completion()

    # WHEN: the inline runner is executed as a subprocess
    result = spawn_runner(chat_completion=chat_completion, output_path=output_path)

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


def test_inline_runner_script_is_packaged_with_tests() -> None:
    """Guard rail: the runner script must live next to this test module."""
    # GIVEN: this test runs from the e2e-tests package
    # WHEN/THEN: the runner script must exist on disk under dragent_tests/
    assert RUNNER_SCRIPT.exists(), (
        f"Expected {RUNNER_SCRIPT} to exist alongside this test."
    )
