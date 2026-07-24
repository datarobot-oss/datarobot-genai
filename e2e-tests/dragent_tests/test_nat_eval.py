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

"""E2E tests for ``nat eval`` with DataRobot moderation evaluators."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from dragent_tests.helpers import AGENT
from dragent_tests.helpers import E2E_ROOT

NAT_EVAL_DIR = E2E_ROOT / "dragent" / "nat" / "eval"
NAT_AGENT_DIR = E2E_ROOT / "dragent" / "nat"
# ``nat eval`` writes here when run manually; the e2e test overrides to ``tmp_path``.
LEGACY_EVAL_OUTPUT_DIR = NAT_AGENT_DIR / ".tmp" / "nat-eval-e2e"

_AGENT_GOAL_ACCURACY_AVAILABLE = (
    importlib.util.find_spec("datarobot_dome.guards.agent_goal_accuracy") is not None
)

NAT_EVAL_CASES: list = [
    ("eval-config-faithfulness.yaml", "faithfulness"),
    ("eval-config-task-adherence.yaml", "task_adherence"),
    pytest.param(
        "eval-config-agent-goal-accuracy.yaml",
        "agent_goal_accuracy",
        marks=pytest.mark.skipif(
            not _AGENT_GOAL_ACCURACY_AVAILABLE,
            reason="datarobot-moderations does not yet ship agent_goal_accuracy guard",
        ),
    ),
    ("eval-config-agent-guideline-adherence.yaml", "agent_guideline_adherence"),
]


pytestmark = [
    # cases.py passes --timeout=60; workflow + LLM judge can take ~2 minutes.
    pytest.mark.timeout(120),
]


def _run_nat_eval(config_file: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "uv",
            "run",
            "nat",
            "eval",
            "--config_file",
            str(config_file),
            "--override",
            "eval.general.output.dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(NAT_AGENT_DIR),
        env={**os.environ},
        check=False,
    )


@pytest.mark.parametrize(("config_name", "evaluator_name"), NAT_EVAL_CASES)
def test_nat_eval_runs_workflow_and_scores_evaluator(
    tmp_path: Path,
    config_name: str,
    evaluator_name: str,
) -> None:
    """``nat eval`` runs the workflow on each dataset row, then scores with the evaluator."""
    config_file = NAT_EVAL_DIR / config_name
    output_dir = tmp_path / evaluator_name
    result = _run_nat_eval(config_file, output_dir)
    output = result.stdout + result.stderr

    try:
        assert result.returncode == 0, (
            f"nat eval failed for {evaluator_name} (exit {result.returncode}).\n"
            f"stdout: {result.stdout[:1000]}\n"
            f"stderr: {result.stderr[:1000]}"
        )
        assert "EVALUATION SUMMARY" in output
        assert evaluator_name in output.lower()
    finally:
        shutil.rmtree(LEGACY_EVAL_OUTPUT_DIR, ignore_errors=True)
