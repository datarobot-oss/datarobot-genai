# Copyright 2026 DataRobot, Inc. and its affiliates.
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

from __future__ import annotations

from collections.abc import Callable
from unittest import mock

import pytest

pytest.importorskip("datarobot_dome")

from nat.data_models.evaluator import EvalInputItem

from datarobot_genai.dragent.eval.agent_goal_accuracy import AgentGoalAccuracyNatEvaluator
from datarobot_genai.dragent.eval.agent_goal_accuracy import score_agent_goal_accuracy_item
from datarobot_genai.dragent.eval.faithfulness import FaithfulnessNatEvaluator
from datarobot_genai.dragent.eval.faithfulness import score_faithfulness_item
from datarobot_genai.dragent.eval.guideline_adherence import GuidelineAdherenceNatEvaluator
from datarobot_genai.dragent.eval.guideline_adherence import score_guideline_adherence_item
from datarobot_genai.dragent.eval.task_adherence import TaskAdherenceNatEvaluator
from datarobot_genai.dragent.eval.task_adherence import score_task_adherence_item


@pytest.mark.asyncio
async def test_score_agent_goal_accuracy_item_delegates_to_guard_helper(
    make_eval_item: Callable[..., EvalInputItem],
) -> None:
    item = make_eval_item(input_obj="task", output_obj="done")
    scorer = mock.Mock()
    with mock.patch(
        "datarobot_genai.dragent.eval.agent_goal_accuracy.calculate_agent_goal_accuracy",
        new=mock.AsyncMock(return_value=1.0),
    ) as mock_calc:
        result = await score_agent_goal_accuracy_item(scorer, item)

    mock_calc.assert_awaited_once_with(scorer, "task", None, "done")
    assert result.id == item.id
    assert result.score == 1.0
    assert result.reasoning["metric"] == "agent_goal_accuracy"
    assert result.reasoning["used_pipeline_interactions"] is False


@pytest.mark.asyncio
async def test_score_agent_goal_accuracy_item_uses_pipeline_interactions(
    make_eval_item: Callable[..., EvalInputItem],
) -> None:
    item = make_eval_item(
        full_dataset_entry={"pipeline_interactions": '{"steps": []}'},
    )
    scorer = mock.Mock()
    with mock.patch(
        "datarobot_genai.dragent.eval.agent_goal_accuracy.calculate_agent_goal_accuracy",
        new=mock.AsyncMock(return_value=0.5),
    ) as mock_calc:
        result = await score_agent_goal_accuracy_item(scorer, item)

    mock_calc.assert_awaited_once_with(scorer, "question", '{"steps": []}', "answer")
    assert result.reasoning["used_pipeline_interactions"] is True


@pytest.mark.asyncio
async def test_agent_goal_accuracy_nat_evaluator_delegates(
    make_eval_item: Callable[..., EvalInputItem],
) -> None:
    item = make_eval_item()
    scorer = mock.Mock()
    evaluator = AgentGoalAccuracyNatEvaluator(scorer=scorer, max_concurrency=1)
    expected = mock.Mock()

    with mock.patch(
        "datarobot_genai.dragent.eval.agent_goal_accuracy.score_agent_goal_accuracy_item",
        new=mock.AsyncMock(return_value=expected),
    ) as mock_score:
        result = await evaluator.evaluate_item(item)

    mock_score.assert_awaited_once_with(scorer, item)
    assert result is expected


@pytest.mark.asyncio
async def test_score_faithfulness_item_delegates_to_guard_helper(
    make_eval_item: Callable[..., EvalInputItem],
) -> None:
    item = make_eval_item(
        input_obj="q",
        output_obj="answer",
        full_dataset_entry={"context": ["doc"]},
    )
    evaluator = mock.Mock()
    with mock.patch(
        "datarobot_genai.dragent.eval.faithfulness.calculate_faithfulness",
        return_value=1.0,
    ) as mock_calc:
        result = await score_faithfulness_item(evaluator, item)

    mock_calc.assert_called_once_with(evaluator, "q", "answer", ["doc"])
    assert result.score == 1.0
    assert result.reasoning["metric"] == "faithfulness"
    assert result.reasoning["context_count"] == 1


@pytest.mark.asyncio
async def test_faithfulness_nat_evaluator_delegates(
    make_eval_item: Callable[..., EvalInputItem],
) -> None:
    item = make_eval_item()
    evaluator = FaithfulnessNatEvaluator(evaluator=mock.Mock(), max_concurrency=1)
    expected = mock.Mock()

    with mock.patch(
        "datarobot_genai.dragent.eval.faithfulness.score_faithfulness_item",
        new=mock.AsyncMock(return_value=expected),
    ) as mock_score:
        result = await evaluator.evaluate_item(item)

    mock_score.assert_awaited_once()
    assert result is expected


@pytest.mark.asyncio
async def test_score_task_adherence_item_delegates_to_guard_helper(
    make_eval_item: Callable[..., EvalInputItem],
) -> None:
    item = make_eval_item(input_obj="task", output_obj="done")
    scorer = mock.Mock()
    with mock.patch(
        "datarobot_genai.dragent.eval.task_adherence.calculate_task_adherence",
        return_value=0.8,
    ) as mock_calc:
        result = await score_task_adherence_item(scorer, item)

    mock_calc.assert_called_once_with(scorer, "task", "", "done")
    assert result.score == 0.8
    assert result.reasoning["metric"] == "task_adherence"


@pytest.mark.asyncio
async def test_task_adherence_nat_evaluator_delegates(
    make_eval_item: Callable[..., EvalInputItem],
) -> None:
    item = make_eval_item()
    evaluator = TaskAdherenceNatEvaluator(scorer=mock.Mock(), max_concurrency=1)
    expected = mock.Mock()

    with mock.patch(
        "datarobot_genai.dragent.eval.task_adherence.score_task_adherence_item",
        new=mock.AsyncMock(return_value=expected),
    ) as mock_score:
        result = await evaluator.evaluate_item(item)

    mock_score.assert_awaited_once()
    assert result is expected


@pytest.mark.asyncio
async def test_score_guideline_adherence_item_delegates_to_guard_helper(
    make_eval_item: Callable[..., EvalInputItem],
) -> None:
    item = make_eval_item(input_obj="task", output_obj="done")
    scorer = mock.Mock()
    with mock.patch(
        "datarobot_genai.dragent.eval.guideline_adherence.calculate_agent_guideline_adherence",
        new=mock.AsyncMock(return_value=True),
    ) as mock_calc:
        result = await score_guideline_adherence_item(scorer, item)

    mock_calc.assert_awaited_once_with(scorer, "task", "done", citations=None)
    assert result.score == 1.0
    assert result.reasoning["metric"] == "agent_guideline_adherence"
    assert result.reasoning["passing"] is True


@pytest.mark.asyncio
async def test_score_guideline_adherence_item_passes_citations(
    make_eval_item: Callable[..., EvalInputItem],
) -> None:
    item = make_eval_item(full_dataset_entry={"context": ["ctx"]})
    scorer = mock.Mock()
    with mock.patch(
        "datarobot_genai.dragent.eval.guideline_adherence.calculate_agent_guideline_adherence",
        new=mock.AsyncMock(return_value=False),
    ) as mock_calc:
        result = await score_guideline_adherence_item(scorer, item)

    mock_calc.assert_awaited_once_with(scorer, "question", "answer", citations=["ctx"])
    assert result.score == 0.0
    assert result.reasoning["passing"] is False


@pytest.mark.asyncio
async def test_guideline_adherence_nat_evaluator_delegates(
    make_eval_item: Callable[..., EvalInputItem],
) -> None:
    item = make_eval_item()
    evaluator = GuidelineAdherenceNatEvaluator(scorer=mock.Mock(), max_concurrency=1)
    expected = mock.Mock()

    with mock.patch(
        "datarobot_genai.dragent.eval.guideline_adherence.score_guideline_adherence_item",
        new=mock.AsyncMock(return_value=expected),
    ) as mock_score:
        result = await evaluator.evaluate_item(item)

    mock_score.assert_awaited_once()
    assert result is expected
