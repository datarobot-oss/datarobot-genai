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
"""``nat eval`` plugin: in-process faithfulness (DataRobot moderations OOTB scorer)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from datarobot_dome.guard_helpers import calculate_faithfulness
from llama_index.core.evaluation import FaithfulnessEvaluator
from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvalInputItem
from nat.data_models.evaluator import EvaluatorLLMConfig
from nat.plugins.eval.data_models.evaluator_io import EvalOutputItem
from nat.plugins.eval.evaluator.base_evaluator import BaseEvaluator

from datarobot_genai.dragent.eval.common import citations_from_eval_item
from datarobot_genai.dragent.eval.common import coerce_text
from datarobot_genai.dragent.eval.scorer_factory import build_faithfulness_evaluator


class DataRobotFaithfulnessEvaluatorConfig(EvaluatorLLMConfig, name="faithfulness"):  # type: ignore[call-arg]
    """Faithfulness evaluator using the DataRobot moderations LlamaIndex judge."""


async def score_faithfulness_item(
    evaluator: FaithfulnessEvaluator,
    item: EvalInputItem,
) -> EvalOutputItem:
    prompt = coerce_text(item.input_obj)
    response = coerce_text(item.output_obj)
    context = citations_from_eval_item(item)
    score = await asyncio.to_thread(
        calculate_faithfulness,
        evaluator,
        prompt,
        response,
        context,
    )
    return EvalOutputItem(
        id=item.id,
        score=score,
        reasoning={
            "metric": "faithfulness",
            "context_count": len(context),
        },
    )


class FaithfulnessNatEvaluator(BaseEvaluator):
    def __init__(self, evaluator: FaithfulnessEvaluator, max_concurrency: int = 4):
        super().__init__(max_concurrency=max_concurrency, tqdm_desc="Faithfulness")
        self._evaluator = evaluator

    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        return await score_faithfulness_item(self._evaluator, item)


@register_evaluator(config_type=DataRobotFaithfulnessEvaluatorConfig)
async def register_dr_faithfulness_evaluator(
    config: DataRobotFaithfulnessEvaluatorConfig, builder: EvalBuilder
) -> AsyncGenerator[EvaluatorInfo]:
    evaluator = await build_faithfulness_evaluator(builder, config.llm_name)
    nat_evaluator = FaithfulnessNatEvaluator(
        evaluator=evaluator, max_concurrency=builder.get_max_concurrency()
    )
    yield EvaluatorInfo(
        config=config,
        evaluate_fn=nat_evaluator.evaluate,
        description="DataRobot faithfulness (in-process OOTB guard scorer)",
    )
