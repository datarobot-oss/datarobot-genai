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
"""Unit tests for the response_acceptability judge benchmark.

The judge call is stubbed, so these pin how the examples reach the judge prompt
and how judge results are shaped, without a network.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from datarobot_genai.eval.benchmarks import response_acceptability


def _sample(**metadata: Any) -> SimpleNamespace:
    return SimpleNamespace(response="agent says hi", target=None, metadata=metadata, config={})


@pytest.fixture
def judge_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_judge_score(sample: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"judge_grade": "4", "judge_score": 0.8}

    monkeypatch.setattr(response_acceptability, "judge_score", fake_judge_score)
    return calls


def test_rating_is_scored_and_explained(judge_calls: list[dict[str, Any]]) -> None:
    result = response_acceptability.score_sample(_sample(input="Q?", acceptable_responses="Paris"))
    assert result["score"] == 0.8
    assert result["judge_grade"] == "4"
    assert result["reason"] == f"judge rated 4/5: {response_acceptability.RUBRIC['4']}"
    assert judge_calls[0]["question"] == "Q?"
    assert judge_calls[0]["score_mapping"]["3"] == 0.6


def test_examples_are_listed_as_reference(judge_calls: list[dict[str, Any]]) -> None:
    response_acceptability.score_sample(_sample(acceptable_responses=["Paris", "It's Paris."]))
    assert judge_calls[0]["reference"] == "- Paris\n- It's Paris."


def test_template_renders_with_every_placeholder() -> None:
    rendered = response_acceptability._TEMPLATE.format(
        question="Q", reference="R", response="A", criteria="C"
    )
    assert "1 - contradicts" in rendered
    assert "GRADE: 5" in rendered


def test_unacceptable_examples_and_notes_reach_criteria(
    judge_calls: list[dict[str, Any]],
) -> None:
    response_acceptability.score_sample(
        _sample(acceptable_responses=["Paris"], unacceptable_responses=["Lyon"], notes="Be strict")
    )
    criteria = judge_calls[0]["criteria"]
    assert "Unacceptable example answers:" in criteria
    assert "- Lyon" in criteria
    assert "Additional guidance: Be strict" in criteria


def test_criteria_omits_optional_sections() -> None:
    criteria = response_acceptability.build_criteria([], "")
    assert criteria == ""


def test_missing_examples_is_inconclusive_without_judge_call(
    judge_calls: list[dict[str, Any]],
) -> None:
    result = response_acceptability.score_sample(_sample(acceptable_responses=[]))
    assert "score" not in result
    assert "cannot score" in result["reason"]
    assert judge_calls == []


def test_judge_error_is_inconclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        response_acceptability,
        "judge_score",
        lambda sample, **kwargs: {"judge_grade": "CALL_ERROR", "judge_score": 0.0},
    )
    result = response_acceptability.score_sample(_sample(acceptable_responses=["Paris"]))
    assert result == {"judge_grade": "CALL_ERROR"}
