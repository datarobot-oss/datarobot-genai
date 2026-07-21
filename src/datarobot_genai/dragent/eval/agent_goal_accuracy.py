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
"""``nat eval`` plugin: in-process agent goal accuracy (DataRobot moderations scorer)."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from datarobot_dome.guard_helpers import calculate_agent_goal_accuracy
from datarobot_dome.guards.agent_goal_accuracy import AgentGoalAccuracyEvaluator
from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvalInputItem
from nat.data_models.evaluator import EvaluatorLLMConfig
from nat.plugins.eval.data_models.evaluator_io import EvalOutputItem
from nat.plugins.eval.evaluator.base_evaluator import BaseEvaluator

from datarobot_genai.dragent.eval.common import coerce_text
from datarobot_genai.dragent.eval.common import interactions_json_from_eval_item
from datarobot_genai.dragent.eval.scorer_factory import build_agent_goal_accuracy_scorer


class DataRobotAgentGoalAccuracyEvaluatorConfig(EvaluatorLLMConfig, name="dr_agent_goal_accuracy"):  # type: ignore[call-arg]
    """Agent goal accuracy evaluator using the DataRobot moderations LLM judge."""


async def score_agent_goal_accuracy_item(
    scorer: AgentGoalAccuracyEvaluator,
    item: EvalInputItem,
) -> EvalOutputItem:
    prompt = coerce_text(item.input_obj)
    response = coerce_text(item.output_obj)
    interactions = interactions_json_from_eval_item(item)
    score = await calculate_agent_goal_accuracy(scorer, prompt, interactions, response)
    return EvalOutputItem(
        id=item.id,
        score=score,
        reasoning={
            "metric": "agent_goal_accuracy",
            "used_pipeline_interactions": interactions is not None,
        },
    )


class AgentGoalAccuracyNatEvaluator(BaseEvaluator):
    def __init__(self, scorer: AgentGoalAccuracyEvaluator, max_concurrency: int = 4):
        super().__init__(max_concurrency=max_concurrency, tqdm_desc="Agent goal accuracy")
        self._scorer = scorer

    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        return await score_agent_goal_accuracy_item(self._scorer, item)


@register_evaluator(config_type=DataRobotAgentGoalAccuracyEvaluatorConfig)
async def register_dr_agent_goal_accuracy_evaluator(
    config: DataRobotAgentGoalAccuracyEvaluatorConfig, builder: EvalBuilder
) -> AsyncGenerator[EvaluatorInfo]:
    scorer = await build_agent_goal_accuracy_scorer(builder, config.llm_name)
    evaluator = AgentGoalAccuracyNatEvaluator(
        scorer=scorer, max_concurrency=builder.get_max_concurrency()
    )
    yield EvaluatorInfo(
        config=config,
        evaluate_fn=evaluator.evaluate,
        description="DataRobot agent goal accuracy (in-process, no NeMo Evaluator microservice)",
    )
