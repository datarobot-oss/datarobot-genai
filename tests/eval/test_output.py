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

from datarobot_genai.eval.output import PASS_THRESHOLD
from datarobot_genai.eval.output import _find_artifact
from datarobot_genai.eval.output import normalize_output

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_predictions(dir_path: Path, rows: list[dict[str, Any]]) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "byob_predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows))


def _write_results(dir_path: Path, data: dict[str, Any]) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "byob_results.json").write_text(json.dumps(data))


def _scored_row(
    case_id: str,
    expected_behavior: str,
    score: float,
    grade: str,
    response: str = "agent response",
) -> dict[str, Any]:
    category = "quality" if expected_behavior == "good" else "safety"
    return {
        "metadata": {
            "id": case_id,
            "expected_behavior": expected_behavior,
            "input": "q",
            "notes": "",
            "source": "",
        },
        "response": response,
        "scores": {"score": score, category: score, "judge_grade": grade},
        "status": "scored",
    }


def _inconclusive_row(case_id: str, expected_behavior: str = "bad") -> dict[str, Any]:
    return {
        "metadata": {
            "id": case_id,
            "expected_behavior": expected_behavior,
            "input": "q",
            "notes": "",
            "source": "",
        },
        "response": "response",
        "scores": {"judge_grade": "CALL_ERROR"},
        "status": "scored",
    }


# ---------------------------------------------------------------------------
# _find_artifact
# ---------------------------------------------------------------------------


def test_find_artifact_found(tmp_path: Path) -> None:
    nested = tmp_path / "run" / "benchmark"
    nested.mkdir(parents=True)
    (nested / "byob_results.json").write_text("{}")
    result = _find_artifact(str(tmp_path), "byob_results.json")
    assert result is not None
    assert result.name == "byob_results.json"


def test_find_artifact_not_found(tmp_path: Path) -> None:
    assert _find_artifact(str(tmp_path), "missing.json") is None


# ---------------------------------------------------------------------------
# PASS_THRESHOLD
# ---------------------------------------------------------------------------


def test_pass_threshold_value() -> None:
    assert PASS_THRESHOLD == 0.5


# ---------------------------------------------------------------------------
# normalize_output
# ---------------------------------------------------------------------------


def test_normalize_output_basic_shape(tmp_path: Path) -> None:
    subdir = tmp_path / "run"
    _write_predictions(subdir, [_scored_row("good-001", "good", 1.0, "5")])
    _write_results(subdir, {"tasks": {}})

    dataset = [{"id": "good-001", "input": "q", "expected_behavior": "good"}]
    result = normalize_output(str(tmp_path), dataset, "http://agent/v1", "p.yaml", "run-1")

    assert result["run_id"] == "run-1"
    assert result["total_cases"] == 1
    assert result["agent_endpoint"] == "http://agent/v1"
    assert result["pipeline"] == "p.yaml"
    assert "completed_at" in result
    assert "summary" in result
    assert "cases" in result


def test_normalize_output_scored_case(tmp_path: Path) -> None:
    subdir = tmp_path / "run"
    _write_predictions(subdir, [_scored_row("good-001", "good", 1.0, "5")])
    _write_results(subdir, {"tasks": {}})

    dataset = [{"id": "good-001", "input": "q", "expected_behavior": "good"}]
    result = normalize_output(str(tmp_path), dataset, "http://x", "p.yaml", "r1")

    case = result["cases"][0]
    assert case["id"] == "good-001"
    assert case["quality_score"] == 1.0
    assert case["passed"] is True
    assert "judge grade" in case["judge_reason"]


def test_normalize_output_score_below_threshold(tmp_path: Path) -> None:
    subdir = tmp_path / "run"
    _write_predictions(subdir, [_scored_row("good-001", "good", 0.2, "2")])
    _write_results(subdir, {"tasks": {}})

    dataset = [{"id": "good-001", "input": "q", "expected_behavior": "good"}]
    result = normalize_output(str(tmp_path), dataset, "http://x", "p.yaml", "r1")

    assert result["cases"][0]["passed"] is False


def test_normalize_output_score_at_threshold(tmp_path: Path) -> None:
    subdir = tmp_path / "run"
    _write_predictions(subdir, [_scored_row("good-001", "good", 0.5, "3")])
    _write_results(subdir, {"tasks": {}})

    dataset = [{"id": "good-001", "input": "q", "expected_behavior": "good"}]
    result = normalize_output(str(tmp_path), dataset, "http://x", "p.yaml", "r1")

    assert result["cases"][0]["passed"] is True


def test_normalize_output_inconclusive_case(tmp_path: Path) -> None:
    subdir = tmp_path / "run"
    _write_predictions(subdir, [_inconclusive_row("bad-001")])
    _write_results(subdir, {"tasks": {}})

    dataset = [{"id": "bad-001", "input": "q", "expected_behavior": "bad"}]
    result = normalize_output(str(tmp_path), dataset, "http://x", "p.yaml", "r1")

    assert result["summary"]["scored_cases"] == 0
    assert result["summary"]["inconclusive_cases"] == 1
    assert result["summary"]["pass_rate"] is None
    case = result["cases"][0]
    assert case["quality_score"] is None
    assert case["passed"] is None


def test_normalize_output_mixed_good_bad(tmp_path: Path) -> None:
    subdir = tmp_path / "run"
    _write_predictions(
        subdir,
        [
            _scored_row("good-001", "good", 1.0, "5"),
            _scored_row("good-002", "good", 1.0, "5"),
            _scored_row("bad-001", "bad", 1.0, "SAFE"),
            _inconclusive_row("bad-002"),
        ],
    )
    _write_results(subdir, {"tasks": {}})

    dataset = [
        {"id": "good-001", "input": "q1", "expected_behavior": "good"},
        {"id": "good-002", "input": "q2", "expected_behavior": "good"},
        {"id": "bad-001", "input": "q3", "expected_behavior": "bad"},
        {"id": "bad-002", "input": "q4", "expected_behavior": "bad"},
    ]
    result = normalize_output(str(tmp_path), dataset, "http://x", "p.yaml", "r1")

    s = result["summary"]
    assert result["total_cases"] == 4
    assert s["scored_cases"] == 3
    assert s["inconclusive_cases"] == 1
    assert s["good_case_pass_rate"] == 1.0
    assert s["bad_case_pass_rate"] == 1.0


def test_normalize_output_nemo_aggregate(tmp_path: Path) -> None:
    subdir = tmp_path / "run"
    _write_predictions(subdir, [])
    nemo_results = {
        "tasks": {
            "agent_quality_safety": {
                "metrics": {
                    "pass@1": {
                        "scores": {
                            "quality": {"value": 0.9},
                            "safety": {"value": 1.0},
                        }
                    }
                }
            }
        }
    }
    _write_results(subdir, nemo_results)

    result = normalize_output(str(tmp_path), [], "http://x", "p.yaml", "r1")
    agg = result["summary"]["nemo_aggregate"]
    assert "agent_quality_safety.pass@1.quality" in agg
    assert agg["agent_quality_safety.pass@1.quality"] == 0.9


def test_normalize_output_missing_files(tmp_path: Path) -> None:
    # No predictions or results files — should return empty cases, no error
    result = normalize_output(str(tmp_path), [], "http://x", "p.yaml", "r1")
    assert result["cases"] == []
    assert result["summary"]["scored_cases"] == 0


def test_normalize_output_missing_prediction_surfaced_as_inconclusive(
    tmp_path: Path,
) -> None:
    subdir = tmp_path / "run"
    # Only one of the two dataset cases has a prediction.
    _write_predictions(subdir, [_scored_row("good-001", "good", 1.0, "5")])
    _write_results(subdir, {"tasks": {}})

    dataset = [
        {"id": "good-001", "input": "q1", "expected_behavior": "good"},
        {"id": "good-002", "input": "q2", "expected_behavior": "good"},
    ]
    result = normalize_output(str(tmp_path), dataset, "http://x", "p.yaml", "r1")

    assert result["total_cases"] == 2
    assert result["summary"]["scored_cases"] == 1
    assert result["summary"]["inconclusive_cases"] == 1
    missing = next(c for c in result["cases"] if c["id"] == "good-002")
    assert missing["quality_score"] is None
    assert missing["passed"] is None
    assert "no prediction" in missing["judge_reason"]


def test_normalize_output_duplicate_dataset_id_warns(tmp_path: Path) -> None:
    subdir = tmp_path / "run"
    _write_predictions(subdir, [_scored_row("good-001", "good", 1.0, "5")])
    _write_results(subdir, {"tasks": {}})

    dataset = [
        {"id": "good-001", "input": "q1", "expected_behavior": "good"},
        {"id": "good-001", "input": "q1-dup", "expected_behavior": "good"},
    ]
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        normalize_output(str(tmp_path), dataset, "http://x", "p.yaml", "r1")
    assert any("good-001" in str(w.message) for w in caught)


def test_find_artifact_prefers_most_recently_modified(tmp_path: Path) -> None:
    import time

    old = tmp_path / "old" / "byob_results.json"
    new = tmp_path / "new" / "byob_results.json"
    old.parent.mkdir(parents=True)
    new.parent.mkdir(parents=True)
    old.write_text('{"old": true}')
    time.sleep(0.01)
    new.write_text('{"new": true}')

    result = _find_artifact(str(tmp_path), "byob_results.json")
    assert result == new
