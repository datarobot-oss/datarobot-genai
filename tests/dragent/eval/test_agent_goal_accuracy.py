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
from typing import Any
from unittest import mock

import pytest

pytest.importorskip("datarobot_dome")

from nat.data_models.evaluator import EvalInputItem

from datarobot_genai.dragent.eval.agent_goal_accuracy import AgentGoalAccuracyNatEvaluator
from datarobot_genai.dragent.eval.agent_goal_accuracy import (
    DataRobotAgentGoalAccuracyEvaluatorConfig,
)
from datarobot_genai.dragent.eval.agent_goal_accuracy import (
    register_dr_agent_goal_accuracy_evaluator,
)
from datarobot_genai.dragent.eval.agent_goal_accuracy import score_agent_goal_accuracy_item
from datarobot_genai.dragent.eval.scorer_factory import build_agent_goal_accuracy_scorer


async def _collect_registered_evaluator_infos(
    register_fn: Any, config: Any, builder: Any
) -> list[Any]:
    async with register_fn(config, builder) as info:
        return [info]


def test_evaluator_config_discriminator_tag() -> None:
    config = DataRobotAgentGoalAccuracyEvaluatorConfig(llm_name="judge")
    assert config.type == "agent_goal_accuracy"


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
async def test_register_dr_agent_goal_accuracy_evaluator(mock_eval_builder: mock.Mock) -> None:
    config = DataRobotAgentGoalAccuracyEvaluatorConfig(llm_name="judge")
    scorer = mock.Mock()

    with mock.patch(
        "datarobot_genai.dragent.eval.agent_goal_accuracy.build_agent_goal_accuracy_scorer",
        new=mock.AsyncMock(return_value=scorer),
    ):
        infos = await _collect_registered_evaluator_infos(
            register_dr_agent_goal_accuracy_evaluator,
            config,
            mock_eval_builder,
        )

    assert len(infos) == 1
    assert infos[0].config is config
    assert "agent goal accuracy" in infos[0].description.lower()
    assert callable(infos[0].evaluate_fn)


@pytest.mark.asyncio
async def test_build_agent_goal_accuracy_scorer(mock_eval_builder: mock.Mock) -> None:
    pytest.importorskip(
        "datarobot_dome.guards.agent_goal_accuracy",
        reason="requires datarobot-moderations>=11.2.45",
    )
    from langchain_openai import ChatOpenAI

    mock_eval_builder.get_llm.return_value = ChatOpenAI(model="gpt-4o-mini", api_key="secret")

    scorer = await build_agent_goal_accuracy_scorer(mock_eval_builder, "judge_llm")

    assert scorer.model == "openai/gpt-4o-mini"
    assert scorer.completion_kwargs["api_key"] == "secret"
