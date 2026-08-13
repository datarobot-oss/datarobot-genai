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
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

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


def _make_runner(
    tmp_path: Path,
    dataset_path: Path | None = None,
    output_name: str | None = None,
    archive: bool = True,
) -> EvalRunner:
    if dataset_path is None:
        p = tmp_path / "cases.json"
        p.write_text(json.dumps([{"id": "c-001", "input": "hello", "expected_behavior": "good"}]))
        dataset_path = p
    return EvalRunner(
        endpoint="http://localhost/v1",
        pipeline="test.yaml",
        dataset=str(dataset_path),
        repo_root=tmp_path,
        output_name=output_name,
        archive=archive,
    )


@contextmanager
def _patched_success() -> Iterator[None]:
    """Stub out every external step of a run so it reaches the output writes."""
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
        yield


def _archives(tmp_path: Path) -> list[str]:
    """List result files in output/, excluding the fixed 'latest' pointers."""
    fixed = {"eval_results.json", "eval_status.json"}
    out = tmp_path / "output"
    return sorted(p.name for p in out.glob("*.json") if p.name not in fixed)


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
# Per-run archive copy
# ---------------------------------------------------------------------------


def test_run_writes_archive_named_after_pipeline(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    with _patched_success():
        runner.run()
    archives = _archives(tmp_path)
    assert len(archives) == 1
    assert archives[0].startswith("test_")
    assert archives[0].endswith(".json")


def test_archive_content_matches_latest_pointer(tmp_path: Path) -> None:
    # Both files are written from one serialization, so they can never disagree
    # about what the run produced.
    runner = _make_runner(tmp_path)
    with _patched_success():
        runner.run()
    latest = (tmp_path / "output" / "eval_results.json").read_text()
    archived = (tmp_path / "output" / _archives(tmp_path)[0]).read_text()
    assert latest == archived


def test_successive_runs_do_not_overwrite_each_other(tmp_path: Path) -> None:
    # The whole point of the feature: run the same pipeline twice and keep both.
    runner = _make_runner(tmp_path)
    with _patched_success():
        runner.run()
        runner.run()
    assert len(_archives(tmp_path)) == 2


def test_run_honors_explicit_output_name(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path, output_name="baseline")
    with _patched_success():
        runner.run()
    assert _archives(tmp_path) == ["baseline.json"]


def test_no_archive_writes_only_the_pointer(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path, archive=False)
    with _patched_success():
        runner.run()
    assert _archives(tmp_path) == []
    assert (tmp_path / "output" / "eval_results.json").exists()


@pytest.mark.parametrize("bad", ["../escape", "sub/name", "eval_results.json"])
def test_invalid_output_name_fails_validation(tmp_path: Path, bad: str) -> None:
    # A bad name is an input error: exit 1 with a failed status, and the
    # evaluation itself never runs.
    runner = _make_runner(tmp_path, output_name=bad)
    with _patched_success() as _, patch("datarobot_genai.eval.eval.run_byob") as run_byob:
        assert runner.run() == 1
    run_byob.assert_not_called()
    status = json.loads((tmp_path / "output" / "eval_status.json").read_text())
    assert status["status"] == "failed"


def test_invalid_output_name_does_not_clobber_previous_results(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    with _patched_success():
        runner.run()
    before = (tmp_path / "output" / "eval_results.json").read_text()

    bad_runner = _make_runner(tmp_path, output_name="../escape")
    with _patched_success():
        assert bad_runner.run() == 1
    assert (tmp_path / "output" / "eval_results.json").read_text() == before


def test_dry_run_reports_archive_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runner = _make_runner(tmp_path)
    with (
        patch("datarobot_genai.eval.eval.validate_inputs", return_value=[]),
        patch("datarobot_genai.eval.eval.load_pipeline", return_value=_PIPELINE_CFG),
    ):
        runner.run(dry_run=True)
    out = capsys.readouterr().out
    assert "archive: output/test_" in out
    assert not (tmp_path / "output").exists()


# ---------------------------------------------------------------------------
# EvalRunner path derivation
# ---------------------------------------------------------------------------


def test_runner_paths_derived_from_repo_root(tmp_path: Path) -> None:
    runner = EvalRunner("http://x", "p.yaml", "d.json", repo_root=tmp_path)
    assert runner.pipelines_dir == tmp_path / "user_pipelines"
    assert runner.output_dir == tmp_path / "output"


def test_runner_archives_by_default(tmp_path: Path) -> None:
    runner = EvalRunner("http://x", "p.yaml", "d.json", repo_root=tmp_path)
    assert runner.archive is True
    assert runner.output_name is None
