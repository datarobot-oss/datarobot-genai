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
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from datarobot_genai.eval.eval import EvalRunner

_PIPELINE_CFG: dict[str, Any] = {
    "benchmark": {
        "module": "datarobot_genai/eval/benchmarks/answer_quality.py",
        "name": "answer_quality",
    },
    "target": {"model_type": "chat", "model_id": "unknown"},
    "judge": {
        "url": "https://judge.example.com",
        "model_id": "gpt-4o",
        "api_key_name": "KEY",
    },
    "run": {},
}

_NORMALIZED_RESULTS: dict[str, Any] = {
    "run_id": "20260601_120000",
    "completed_at": "2026-06-01T12:00:00+00:00",
    "agent_endpoint": "http://localhost/v1",
    "pipeline": "test.yaml",
    "total_cases": 1,
    "summary": {
        "scored_cases": 1,
        "inconclusive_cases": 0,
        "mean_score": 1.0,
        "pass_rate": 1.0,
        "good_case_pass_rate": 1.0,
        "bad_case_pass_rate": None,
        "nemo_aggregate": {},
    },
    "cases": [],
}


def _make_runner(tmp_path: Path, dataset_path: Path | None = None) -> EvalRunner:
    if dataset_path is None:
        p = tmp_path / "cases.json"
        p.write_text(json.dumps([{"id": "c-001", "input": "hello", "expected_behavior": "good"}]))
        dataset_path = p
    return EvalRunner(
        endpoint="http://localhost/v1",
        pipeline="test.yaml",
        dataset=str(dataset_path),
        repo_root=tmp_path,
    )


# ---------------------------------------------------------------------------
# Validation failure → exit 1
# ---------------------------------------------------------------------------


def test_run_returns_1_on_validation_failure(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    with patch(
        "datarobot_genai.eval.eval.validate_inputs",
        return_value=["Endpoint not reachable"],
    ):
        assert runner.run() == 1


def test_run_writes_failed_status_on_validation_failure(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    with patch(
        "datarobot_genai.eval.eval.validate_inputs",
        return_value=["Endpoint not reachable", "Dataset not found"],
    ):
        runner.run()
    status = json.loads((tmp_path / "output" / "eval_status.json").read_text())
    assert status["status"] == "failed"
    assert "Endpoint not reachable" in status["error"]
    assert "Dataset not found" in status["error"]


# ---------------------------------------------------------------------------
# Dry run → exit 0, no side-effects
# ---------------------------------------------------------------------------


def test_run_dry_run_returns_0(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    with (
        patch("datarobot_genai.eval.eval.validate_inputs", return_value=[]),
        patch("datarobot_genai.eval.eval.load_pipeline", return_value=_PIPELINE_CFG),
    ):
        assert runner.run(dry_run=True) == 0


def test_run_dry_run_does_not_write_output(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    with (
        patch("datarobot_genai.eval.eval.validate_inputs", return_value=[]),
        patch("datarobot_genai.eval.eval.load_pipeline", return_value=_PIPELINE_CFG),
    ):
        runner.run(dry_run=True)
    assert not (tmp_path / "output").exists()


# ---------------------------------------------------------------------------
# BYOB failure → exit 2, status = failed
# ---------------------------------------------------------------------------


def test_run_returns_2_on_byob_failure(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    with (
        patch("datarobot_genai.eval.eval.validate_inputs", return_value=[]),
        patch("datarobot_genai.eval.eval.load_pipeline", return_value=_PIPELINE_CFG),
        patch("datarobot_genai.eval.eval.preflight_judge"),
        patch(
            "datarobot_genai.eval.eval.run_byob",
            side_effect=RuntimeError("runner crashed"),
        ),
    ):
        assert runner.run() == 2


def test_run_writes_failed_status_on_byob_error(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    with (
        patch("datarobot_genai.eval.eval.validate_inputs", return_value=[]),
        patch("datarobot_genai.eval.eval.load_pipeline", return_value=_PIPELINE_CFG),
        patch("datarobot_genai.eval.eval.preflight_judge"),
        patch(
            "datarobot_genai.eval.eval.run_byob",
            side_effect=RuntimeError("runner crashed"),
        ),
    ):
        runner.run()
    status = json.loads((tmp_path / "output" / "eval_status.json").read_text())
    assert status["status"] == "failed"
    assert "runner crashed" in status["error"]


# ---------------------------------------------------------------------------
# Normalization failure → exit 3
# ---------------------------------------------------------------------------


def test_run_returns_3_on_normalization_failure(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    with (
        patch("datarobot_genai.eval.eval.validate_inputs", return_value=[]),
        patch("datarobot_genai.eval.eval.load_pipeline", return_value=_PIPELINE_CFG),
        patch("datarobot_genai.eval.eval.preflight_judge"),
        patch("datarobot_genai.eval.eval.run_byob"),
        patch(
            "datarobot_genai.eval.eval.normalize_output",
            side_effect=ValueError("bad output"),
        ),
    ):
        assert runner.run() == 3


# ---------------------------------------------------------------------------
# Happy path → exit 0, results + status written
# ---------------------------------------------------------------------------


def test_run_happy_path_returns_0(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    with (
        patch("datarobot_genai.eval.eval.validate_inputs", return_value=[]),
        patch("datarobot_genai.eval.eval.load_pipeline", return_value=_PIPELINE_CFG),
        patch("datarobot_genai.eval.eval.preflight_judge"),
        patch("datarobot_genai.eval.eval.run_byob"),
        patch(
            "datarobot_genai.eval.eval.normalize_output",
            return_value=_NORMALIZED_RESULTS,
        ),
    ):
        assert runner.run() == 0


def test_run_happy_path_writes_results(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    with (
        patch("datarobot_genai.eval.eval.validate_inputs", return_value=[]),
        patch("datarobot_genai.eval.eval.load_pipeline", return_value=_PIPELINE_CFG),
        patch("datarobot_genai.eval.eval.preflight_judge"),
        patch("datarobot_genai.eval.eval.run_byob"),
        patch(
            "datarobot_genai.eval.eval.normalize_output",
            return_value=_NORMALIZED_RESULTS,
        ),
    ):
        runner.run()
    results = json.loads((tmp_path / "output" / "eval_results.json").read_text())
    assert results["total_cases"] == 1


def test_run_happy_path_status_complete(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    with (
        patch("datarobot_genai.eval.eval.validate_inputs", return_value=[]),
        patch("datarobot_genai.eval.eval.load_pipeline", return_value=_PIPELINE_CFG),
        patch("datarobot_genai.eval.eval.preflight_judge"),
        patch("datarobot_genai.eval.eval.run_byob"),
        patch(
            "datarobot_genai.eval.eval.normalize_output",
            return_value=_NORMALIZED_RESULTS,
        ),
    ):
        runner.run()
    status = json.loads((tmp_path / "output" / "eval_status.json").read_text())
    assert status["status"] == "complete"
    assert status["error"] is None


# ---------------------------------------------------------------------------
# Last-resort guard → any unhandled failure still leaves status = failed
# ---------------------------------------------------------------------------


def test_run_writes_failed_status_on_unexpected_error(tmp_path: Path) -> None:
    # load_pipeline raising is not caught by any inner branch; the top-level
    # guard must still flip the status to "failed" rather than let a traceback
    # escape with no status written.
    runner = _make_runner(tmp_path)
    with (
        patch("datarobot_genai.eval.eval.validate_inputs", return_value=[]),
        patch(
            "datarobot_genai.eval.eval.load_pipeline",
            side_effect=RuntimeError("boom"),
        ),
    ):
        assert runner.run() == 1
    status = json.loads((tmp_path / "output" / "eval_status.json").read_text())
    assert status["status"] == "failed"
    assert "boom" in status["error"]


def test_run_not_left_running_when_results_write_fails(tmp_path: Path) -> None:
    # After status is flipped to "running", a failure serializing/writing the
    # results (here: a non-JSON-serializable normalized result) must not leave
    # the status stuck at "running" — the guard flips it to failed. The small
    # status payload is plain strings, so it still serializes and persists.
    unserializable = {"total_cases": 1, "summary": {}, "extra": {1, 2, 3}}
    runner = _make_runner(tmp_path)
    with (
        patch("datarobot_genai.eval.eval.validate_inputs", return_value=[]),
        patch("datarobot_genai.eval.eval.load_pipeline", return_value=_PIPELINE_CFG),
        patch("datarobot_genai.eval.eval.preflight_judge"),
        patch("datarobot_genai.eval.eval.run_byob"),
        patch(
            "datarobot_genai.eval.eval.normalize_output",
            return_value=unserializable,
        ),
    ):
        assert runner.run() == 1
    status = json.loads((tmp_path / "output" / "eval_status.json").read_text())
    assert status["status"] == "failed"


# ---------------------------------------------------------------------------
# EvalRunner path derivation
# ---------------------------------------------------------------------------


def test_runner_paths_derived_from_repo_root(tmp_path: Path) -> None:
    runner = EvalRunner("http://x", "p.yaml", "d.json", repo_root=tmp_path)
    assert runner.pipelines_dir == tmp_path / "user_pipelines"
    assert runner.output_dir == tmp_path / "output"
