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

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from dragent_tests.helpers import AGENT
from dragent_tests.helpers import E2E_ROOT

NAT_EVAL_DIR = E2E_ROOT / "dragent" / "nat" / "eval"
NAT_AGENT_DIR = E2E_ROOT / "dragent" / "nat"
EVAL_CONFIG = NAT_EVAL_DIR / "eval-config.yaml"
# ``nat eval`` writes here when run manually; the e2e test overrides to ``tmp_path``.
LEGACY_EVAL_OUTPUT_DIR = NAT_AGENT_DIR / ".tmp" / "nat-eval-e2e"


pytestmark = [
    pytest.mark.skipif(AGENT != "nat", reason="nat eval is NAT-only"),
    # cases.py passes --timeout=60; ``nat eval`` cold start + LLM judge takes ~80s.
    pytest.mark.timeout(120),
]


def test_nat_eval_skip_workflow_scores_faithfulness(tmp_path: Path) -> None:
    """``nat eval --skip_workflow`` runs DataRobot faithfulness evaluator end-to-end."""
    output_dir = tmp_path / "nat-eval-e2e"
    result = subprocess.run(
        [
            "uv",
            "run",
            "nat",
            "eval",
            "--config_file",
            str(EVAL_CONFIG),
            "--skip_workflow",
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
    output = result.stdout + result.stderr

    try:
        assert result.returncode == 0, (
            f"nat eval --skip_workflow failed (exit {result.returncode}).\n"
            f"stdout: {result.stdout[:1000]}\n"
            f"stderr: {result.stderr[:1000]}"
        )
        assert "EVALUATION SUMMARY" in output
        assert "faithfulness" in output.lower()
    finally:
        # ``--skip_workflow`` disables NAT's pre-run cleanup; remove any prior manual-run dir.
        shutil.rmtree(LEGACY_EVAL_OUTPUT_DIR, ignore_errors=True)
