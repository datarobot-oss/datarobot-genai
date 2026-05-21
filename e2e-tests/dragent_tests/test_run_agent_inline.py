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
import subprocess
import sys
from pathlib import Path

from openai.types.chat import ChatCompletion

from dragent_tests.helpers import AGENT

E2E_ROOT = Path(__file__).resolve().parent.parent
RUNNER_SCRIPT = E2E_ROOT / "dragent" / "run_agent.py"
RUNNER_MODULE = "dragent.run_agent"
AGENT_DIR = E2E_ROOT / "dragent" / (AGENT or "base")
WORKFLOW_CONFIG = AGENT_DIR / "workflow.yaml"


def build_chat_completion() -> dict[str, object]:
    return {
        "model": "unknown",
        "messages": [{"role": "user", "content": "Say 'hello world' and nothing else."}],
    }


def spawn_runner(
    *,
    chat_completion: dict[str, object],
    output_path: Path,
) -> subprocess.CompletedProcess[str]:
    # Invoke as ``python -m dragent.run_agent`` rather than as a script.
    # If we ran the script directly, Python would prepend ``e2e-tests/dragent/``
    # to ``sys.path``; that directory contains a local ``nat/`` subpackage (the
    # NAT e2e agent) that shadows the third-party ``nvidia-nat`` and breaks
    # ``import nat.data_models``. ``-m`` puts ``cwd`` (``e2e-tests/``) on
    # ``sys.path`` instead, which has no top-level ``nat/`` collision.
    return subprocess.run(
        [
            sys.executable,
            "-m",
            RUNNER_MODULE,
            "--chat_completion",
            json.dumps(chat_completion),
            "--custom_model_dir",
            str(AGENT_DIR),
            "--output_path",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(E2E_ROOT),
        env={**os.environ},
        check=False,
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
