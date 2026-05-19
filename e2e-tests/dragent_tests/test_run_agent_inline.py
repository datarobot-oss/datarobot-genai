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

The companion script :mod:`dragent_tests._run_agent_inline` plays the role of
``datarobot-user-models``'s ``run_agent.py``: it accepts a chat completion as
JSON on the command line, invokes ``execute_dragent_inline``, and writes the
result to a file. This test then reads that file back and validates it parses
as an OpenAI ``ChatCompletion`` (non-streaming) or list of
``ChatCompletionChunk`` (streaming).
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path

import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionChunk

from dragent_tests.helpers import AGENT

E2E_ROOT = Path(__file__).resolve().parent.parent
RUNNER_SCRIPT = Path(__file__).resolve().parent / "_run_agent_inline.py"
AGENT_DIR = E2E_ROOT / "dragent" / AGENT
WORKFLOW_CONFIG = AGENT_DIR / "workflow.yaml"


def build_chat_completion(*, stream: bool) -> dict[str, object]:
    uid = uuid.uuid4().hex[:8]
    return {
        "model": "datarobot-e2e",
        "messages": [{"role": "user", "content": "Say 'hello world' and nothing else."}],
        "stream": stream,
        # AG-UI thread/run ids piggyback for parity with the FastAPI /generate path.
        "thread_id": f"test-{uid}",
        "run_id": f"run-{uid}",
    }


def spawn_runner(
    *,
    chat_completion: dict[str, object],
    output_path: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(RUNNER_SCRIPT),
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

def test_run_agent_inline_non_streaming(tmp_path: Path) -> None:
    """Non-streaming inline run produces a valid ``ChatCompletion`` JSON file."""
    # GIVEN: a non-streaming chat completion request and an output path
    output_path = tmp_path / "output.json"
    chat_completion = build_chat_completion(stream=False)

    # WHEN: the inline runner is executed as a subprocess
    result = spawn_runner(chat_completion=chat_completion, output_path=output_path)

    # THEN: the runner exits cleanly and the file parses as a ChatCompletion
    assert result.returncode == 0, (
        f"runner failed (exit {result.returncode}).\n"
        f"stdout: {result.stdout[:1000]}\n"
        f"stderr: {result.stderr[:2000]}"
    )
    assert output_path.exists(), "Expected the runner to write the output file"

    payload = json.loads(output_path.read_text())
    completion = ChatCompletion.model_validate(payload)
    assert completion.choices, "Expected at least one choice"
    assert completion.choices[0].message.content, (
        "Expected non-empty assistant message content"
    )

@pytest.mark.skipif(AGENT == "nat", reason=(
    "NAT does not support chat completions with streaming: its output is not in the "
    "Chat Completions format"
))
def test_run_agent_inline_streaming(tmp_path: Path) -> None:
    """Streaming inline run produces a JSON array of ``ChatCompletionChunk``s."""
    # GIVEN: a streaming chat completion request
    output_path = tmp_path / "output.json"
    chat_completion = build_chat_completion(stream=True)

    # WHEN: the inline runner is executed as a subprocess
    result = spawn_runner(chat_completion=chat_completion, output_path=output_path)

    # THEN: the runner exits cleanly and the file parses as a list[ChatCompletionChunk]
    assert result.returncode == 0, (
        f"runner failed (exit {result.returncode}).\n"
        f"stdout: {result.stdout[:1000]}\n"
        f"stderr: {result.stderr[:2000]}"
    )
    assert output_path.exists(), "Expected the runner to write the output file"

    payload = json.loads(output_path.read_text())
    assert isinstance(payload, list), "Streaming mode must serialise to a JSON array"
    chunks = [ChatCompletionChunk.model_validate(item) for item in payload]
    assert chunks, "Expected at least one streaming chunk"

    concatenated = "".join(
        choice.delta.content or ""
        for chunk in chunks
        for choice in chunk.choices
    )
    assert concatenated, "Expected non-empty concatenated streamed content"


def test_inline_runner_script_is_packaged_with_tests() -> None:
    """Guard rail: the runner script must live next to this test module."""
    # GIVEN: this test runs from the e2e-tests package
    # WHEN/THEN: the runner script must exist on disk under dragent_tests/
    assert RUNNER_SCRIPT.exists(), (
        f"Expected {RUNNER_SCRIPT} to exist alongside this test."
    )
