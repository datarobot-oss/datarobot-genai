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

from typing import Any
from unittest import mock

import pytest

pytest.importorskip("datarobot_dome")

from datarobot_genai.dragent.eval.faithfulness import DataRobotFaithfulnessEvaluatorConfig
from datarobot_genai.dragent.eval.faithfulness import register_dr_faithfulness_evaluator
from datarobot_genai.dragent.eval.guideline_adherence import (
    DataRobotGuidelineAdherenceEvaluatorConfig,
)
from datarobot_genai.dragent.eval.guideline_adherence import (
    register_dr_guideline_adherence_evaluator,
)
from datarobot_genai.dragent.eval.task_adherence import DataRobotTaskAdherenceEvaluatorConfig
from datarobot_genai.dragent.eval.task_adherence import register_dr_task_adherence_evaluator


async def _collect_registered_evaluator_infos(
    register_fn: Any, config: Any, builder: Any
) -> list[Any]:
    async with register_fn(config, builder) as info:
        return [info]


@pytest.mark.parametrize(
    ("config_cls", "expected_type"),
    [
        (DataRobotFaithfulnessEvaluatorConfig, "faithfulness"),
        (DataRobotTaskAdherenceEvaluatorConfig, "task_adherence"),
        (DataRobotGuidelineAdherenceEvaluatorConfig, "agent_guideline_adherence"),
    ],
)
def test_evaluator_config_discriminator_tags(config_cls: type, expected_type: str) -> None:
    if config_cls is DataRobotGuidelineAdherenceEvaluatorConfig:
        config = config_cls(llm_name="judge", agent_guideline="Be concise.")
    else:
        config = config_cls(llm_name="judge")

    assert config.type == expected_type


@pytest.mark.asyncio
async def test_register_dr_faithfulness_evaluator(mock_eval_builder: mock.Mock) -> None:
    config = DataRobotFaithfulnessEvaluatorConfig(llm_name="judge")
    evaluator = mock.Mock()

    with mock.patch(
        "datarobot_genai.dragent.eval.faithfulness.build_faithfulness_evaluator",
        new=mock.AsyncMock(return_value=evaluator),
    ):
        infos = await _collect_registered_evaluator_infos(
            register_dr_faithfulness_evaluator,
            config,
            mock_eval_builder,
        )

    assert len(infos) == 1
    assert infos[0].config is config
    assert "faithfulness" in infos[0].description.lower()


@pytest.mark.asyncio
async def test_register_dr_task_adherence_evaluator(mock_eval_builder: mock.Mock) -> None:
    config = DataRobotTaskAdherenceEvaluatorConfig(llm_name="judge")
    scorer = mock.Mock()

    with mock.patch(
        "datarobot_genai.dragent.eval.task_adherence.build_task_adherence_scorer",
        new=mock.AsyncMock(return_value=scorer),
    ):
        infos = await _collect_registered_evaluator_infos(
            register_dr_task_adherence_evaluator,
            config,
            mock_eval_builder,
        )

    assert len(infos) == 1
    assert infos[0].config is config
    assert "task adherence" in infos[0].description.lower()


@pytest.mark.asyncio
async def test_register_dr_guideline_adherence_evaluator(mock_eval_builder: mock.Mock) -> None:
    config = DataRobotGuidelineAdherenceEvaluatorConfig(
        llm_name="judge",
        agent_guideline="Be professional.",
    )
    scorer = mock.Mock()

    with mock.patch(
        "datarobot_genai.dragent.eval.guideline_adherence.build_guideline_adherence_scorer",
        new=mock.AsyncMock(return_value=scorer),
    ):
        infos = await _collect_registered_evaluator_infos(
            register_dr_guideline_adherence_evaluator,
            config,
            mock_eval_builder,
        )

    assert len(infos) == 1
    assert infos[0].config is config
    assert "guideline adherence" in infos[0].description.lower()


def test_register_module_imports_all_evaluators() -> None:
    from datarobot_genai.dragent.eval import register

    assert register.__all__ == [
        "_agent_goal_accuracy",
        "_faithfulness",
        "_guideline_adherence",
        "_task_adherence",
    ]
