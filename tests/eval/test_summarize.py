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

import pytest

from datarobot_genai.eval.summarize import ResultsSummarizer


def _write_results(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(json.dumps(data))
    return path


def _minimal_results(
    run_id: str = "20260601_120000",
    total_cases: int = 2,
    scored: int = 2,
    inconclusive: int = 0,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "completed_at": "2026-06-01T12:00:00+00:00",
        "agent_endpoint": "http://localhost/v1",
        "pipeline": "test.yaml",
        "total_cases": total_cases,
        "summary": {
            "scored_cases": scored,
            "inconclusive_cases": inconclusive,
            "mean_quality_score": 1.0,
            "pass_rate": 1.0,
            "good_case_pass_rate": 1.0,
            "bad_case_pass_rate": 1.0,
            "nemo_aggregate": {},
        },
        "cases": [
            {
                "id": "good-001",
                "input": "question",
                "expected_behavior": "good",
                "agent_response": "answer",
                "quality_score": 1.0,
                "judge_reason": "judge grade: 5",
                "passed": True,
                "answer_match_score": None,
                "notes": "",
                "source": "",
            }
        ],
    }


# ---------------------------------------------------------------------------
# _resolve
# ---------------------------------------------------------------------------


def test_resolve_accepts_direct_file(tmp_path: Path) -> None:
    p = _write_results(tmp_path / "eval_results.json", _minimal_results())
    s = ResultsSummarizer(p)
    assert s.results_path == p


def test_resolve_accepts_directory(tmp_path: Path) -> None:
    _write_results(tmp_path / "eval_results.json", _minimal_results())
    s = ResultsSummarizer(tmp_path)
    assert s.results_path == tmp_path / "eval_results.json"


def test_resolve_raises_when_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No eval_results.json"):
        ResultsSummarizer(tmp_path)


# ---------------------------------------------------------------------------
# print_summary
# ---------------------------------------------------------------------------


def test_print_summary_shows_run_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_results(tmp_path / "eval_results.json", _minimal_results(run_id="run-42"))
    ResultsSummarizer(tmp_path).print_summary()
    out = capsys.readouterr().out
    assert "run-42" in out
    assert "http://localhost/v1" in out
    assert "test.yaml" in out


def test_print_summary_shows_scores(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_results(tmp_path / "eval_results.json", _minimal_results())
    ResultsSummarizer(tmp_path).print_summary()
    out = capsys.readouterr().out
    assert "1.0" in out
    assert "Pass rate" in out


def test_print_summary_shows_per_case_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_results(tmp_path / "eval_results.json", _minimal_results())
    ResultsSummarizer(tmp_path).print_summary()
    out = capsys.readouterr().out
    assert "good-001" in out
    assert "Per-case" in out


def test_print_summary_shows_nemo_aggregate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = _minimal_results()
    data["summary"]["nemo_aggregate"] = {"agent_quality_safety.pass@1.quality": 0.95}
    _write_results(tmp_path / "eval_results.json", data)
    ResultsSummarizer(tmp_path).print_summary()
    out = capsys.readouterr().out
    assert "NeMo Aggregate" in out
    assert "0.95" in out


def test_print_summary_no_nemo_section_when_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_results(tmp_path / "eval_results.json", _minimal_results())
    ResultsSummarizer(tmp_path).print_summary()
    out = capsys.readouterr().out
    assert "NeMo Aggregate" not in out
