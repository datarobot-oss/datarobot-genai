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
"""``nat eval`` plugin: in-process task adherence (DataRobot moderations DeepEval scorer)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from datarobot_dome.guard_helpers import calculate_task_adherence
from deepeval.metrics import TaskCompletionMetric
from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvalInputItem
from nat.data_models.evaluator import EvaluatorLLMConfig
from nat.plugins.eval.data_models.evaluator_io import EvalOutputItem
from nat.plugins.eval.evaluator.base_evaluator import BaseEvaluator

from datarobot_genai.dragent.eval.common import coerce_text
from datarobot_genai.dragent.eval.common import interactions_json_from_eval_item
from datarobot_genai.dragent.eval.scorer_factory import build_task_adherence_scorer


class DataRobotTaskAdherenceEvaluatorConfig(EvaluatorLLMConfig, name="task_adherence"):  # type: ignore[call-arg]
    """Task adherence evaluator using the DataRobot moderations DeepEval judge."""


async def score_task_adherence_item(
    scorer: TaskCompletionMetric,
    item: EvalInputItem,
) -> EvalOutputItem:
    prompt = coerce_text(item.input_obj)
    response = coerce_text(item.output_obj)
    interactions = interactions_json_from_eval_item(item) or ""
    score = await asyncio.to_thread(
        calculate_task_adherence,
        scorer,
        prompt,
        interactions,
        response,
    )
    return EvalOutputItem(
        id=item.id,
        score=score,
        reasoning={"metric": "task_adherence"},
    )


class TaskAdherenceNatEvaluator(BaseEvaluator):
    def __init__(self, scorer: TaskCompletionMetric, max_concurrency: int = 4):
        super().__init__(max_concurrency=max_concurrency, tqdm_desc="Task adherence")
        self._scorer = scorer

    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        return await score_task_adherence_item(self._scorer, item)


@register_evaluator(config_type=DataRobotTaskAdherenceEvaluatorConfig)
async def register_dr_task_adherence_evaluator(
    config: DataRobotTaskAdherenceEvaluatorConfig, builder: EvalBuilder
) -> AsyncGenerator[EvaluatorInfo]:
    scorer = await build_task_adherence_scorer(builder, config.llm_name)
    evaluator = TaskAdherenceNatEvaluator(
        scorer=scorer, max_concurrency=builder.get_max_concurrency()
    )
    yield EvaluatorInfo(
        config=config,
        evaluate_fn=evaluator.evaluate,
        description="DataRobot task adherence (in-process OOTB guard scorer)",
    )
