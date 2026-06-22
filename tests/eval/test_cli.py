# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the eval CLI entrypoints (run / generate / summarize).

These run entirely in-process with no LLM, no network, and no DataRobot
credentials — enough to prove each entrypoint parses its arguments and runs its
no-model codepaths.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from datarobot_genai.eval import cli

_PIPELINE_CFG: dict[str, Any] = {
    "benchmark": {
        "module": "datarobot_genai/eval/benchmarks/answer_quality.py",
        "name": "answer_quality",
    },
    "target": {"model_type": "chat", "model_id": "unknown"},
    "judge": None,
    "run": {},
}

_RESULTS: dict[str, Any] = {
    "run_id": "20260601_120000",
    "completed_at": "2026-06-01T12:00:00+00:00",
    "agent_endpoint": "http://localhost/v1",
    "pipeline": "answer_quality.yaml",
    "total_cases": 1,
    "summary": {
        "scored_cases": 1,
        "inconclusive_cases": 0,
        "mean_quality_score": 1.0,
        "pass_rate": 1.0,
        "good_case_pass_rate": 1.0,
        "bad_case_pass_rate": None,
        "nemo_aggregate": {},
    },
    "cases": [
        {
            "id": "c-001",
            "input": "hello",
            "expected_behavior": "good",
            "agent_response": "hi",
        }
    ],
}


# ---------------------------------------------------------------------------
# --help builds the argument parser and exits 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", [cli.run_main, cli.generate_main])
def test_help_exits_zero(entry: Any) -> None:
    with pytest.raises(SystemExit) as exc:
        entry(argv=["--help"])
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# generate --convert: real CSV -> JSON, no model involved
# ---------------------------------------------------------------------------


def test_generate_convert(tmp_path: Path) -> None:
    csv_path = tmp_path / "cases.csv"
    csv_path.write_text(
        "id,source,input,notes\n"
        "c-001,collected,What is 2+2?,arithmetic\n"
        "c-002,collected,Capital of France?,geography\n"
    )
    out_path = tmp_path / "cases.json"

    cli.generate_main(
        argv=["--convert", str(csv_path), "--output", str(out_path)],
        repo_root=tmp_path,
    )

    cases = json.loads(out_path.read_text())
    assert [c["id"] for c in cases] == ["c-001", "c-002"]


# ---------------------------------------------------------------------------
# summarize: pretty-print a results file — the eval display codepath
# ---------------------------------------------------------------------------


def test_summarize_prints_results(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    results_path = tmp_path / "eval_results.json"
    results_path.write_text(json.dumps(_RESULTS))

    cli.summarize_main(argv=[str(results_path)])  # must not raise

    assert capsys.readouterr().out.strip()  # produced some summary output


def test_summarize_missing_file_exits_1(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.summarize_main(argv=[str(tmp_path / "nope.json")])
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# run --dry-run: exercises run_main wiring without a network call
# ---------------------------------------------------------------------------


def test_run_dry_run_exits_zero(tmp_path: Path) -> None:
    with (
        patch("datarobot_genai.eval.eval.validate_inputs", return_value=[]),
        patch("datarobot_genai.eval.eval.load_pipeline", return_value=_PIPELINE_CFG),
        pytest.raises(SystemExit) as exc,
    ):
        cli.run_main(
            argv=[
                "--endpoint",
                "http://localhost/v1",
                "--pipeline",
                "answer_quality.yaml",
                "--dry-run",
            ],
            repo_root=tmp_path,
        )
    assert exc.value.code == 0
