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

"""E2E tests for ``nat dragent`` CLI commands."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from dragent_tests.helpers import FRAMEWORK

_E2E_ROOT = Path(__file__).resolve().parent.parent
_WORKFLOW_CONFIG = _E2E_ROOT / "dragent" / (FRAMEWORK or "langgraph") / "workflow.yaml"


@pytest.mark.skipif(not FRAMEWORK, reason="FRAMEWORK env var not set")
@pytest.mark.skipif(
    not _WORKFLOW_CONFIG.exists(),
    reason=f"Workflow config not found: {_WORKFLOW_CONFIG}",
)
def test_cli_run_produces_output() -> None:
    """``nat dragent run`` executes a workflow in-process and prints output."""
    result = subprocess.run(
        [
            "uv", "run",
            "nat", "dragent", "run",
            "--config_file", str(_WORKFLOW_CONFIG),
            "--input", "Say 'hello world' and nothing else.",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(_E2E_ROOT),
        env={**os.environ},
        check=False,
    )

    assert result.returncode == 0, (
        f"nat dragent run failed (exit {result.returncode}).\n"
        f"stdout: {result.stdout[:500]}\n"
        f"stderr: {result.stderr[:500]}"
    )
    output = result.stdout + result.stderr
    assert len(output.strip()) > 0, "Expected non-empty output from nat dragent run"


def test_cli_query_local_server() -> None:
    """``nat dragent query --local`` queries the background dragent server via CLI."""
    result = subprocess.run(
        [
            "uv", "run",
            "nat", "dragent", "query",
            "--local",
            "--input", "Say 'hello world' and nothing else.",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(_E2E_ROOT),
        env={**os.environ, "AGENT_PORT": "8080"},
        check=False,
    )

    assert result.returncode == 0, (
        f"nat dragent query --local failed (exit {result.returncode}).\n"
        f"stdout: {result.stdout[:500]}\n"
        f"stderr: {result.stderr[:500]}"
    )
    assert "hello" in result.stdout.lower(), "Expected 'hello' in query output"
