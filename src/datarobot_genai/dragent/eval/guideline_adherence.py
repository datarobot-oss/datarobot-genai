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
"""``nat eval`` plugin: in-process guideline adherence (DataRobot moderations OOTB scorer)."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from datarobot_dome.guard_helpers import calculate_agent_guideline_adherence
from llama_index.core.evaluation import GuidelineEvaluator
from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvalInputItem
from nat.data_models.evaluator import EvaluatorLLMConfig
from nat.plugins.eval.data_models.evaluator_io import EvalOutputItem
from nat.plugins.eval.evaluator.base_evaluator import BaseEvaluator
from pydantic import Field

from datarobot_genai.dragent.eval.common import citations_from_eval_item
from datarobot_genai.dragent.eval.common import coerce_text
from datarobot_genai.dragent.eval.scorer_factory import build_guideline_adherence_scorer


class DataRobotGuidelineAdherenceEvaluatorConfig(EvaluatorLLMConfig, name="dr_guideline_adherence"):  # type: ignore[call-arg]
    """Guideline adherence evaluator using the DataRobot moderations LlamaIndex judge."""

    agent_guideline: str = Field(
        description="Natural-language guideline the agent response must follow.",
    )


async def score_guideline_adherence_item(
    scorer: GuidelineEvaluator,
    item: EvalInputItem,
) -> EvalOutputItem:
    prompt = coerce_text(item.input_obj)
    response = coerce_text(item.output_obj)
    citations = citations_from_eval_item(item) or None
    passing = await calculate_agent_guideline_adherence(
        scorer,
        prompt,
        response,
        citations=citations,
    )
    return EvalOutputItem(
        id=item.id,
        score=float(passing),
        reasoning={
            "metric": "guideline_adherence",
            "passing": passing,
        },
    )


class GuidelineAdherenceNatEvaluator(BaseEvaluator):
    def __init__(self, scorer: GuidelineEvaluator, max_concurrency: int = 4):
        super().__init__(max_concurrency=max_concurrency, tqdm_desc="Guideline adherence")
        self._scorer = scorer

    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        return await score_guideline_adherence_item(self._scorer, item)


@register_evaluator(config_type=DataRobotGuidelineAdherenceEvaluatorConfig)
async def register_dr_guideline_adherence_evaluator(
    config: DataRobotGuidelineAdherenceEvaluatorConfig, builder: EvalBuilder
) -> AsyncGenerator[EvaluatorInfo]:
    scorer = await build_guideline_adherence_scorer(
        builder, config.llm_name, config.agent_guideline
    )
    evaluator = GuidelineAdherenceNatEvaluator(
        scorer=scorer, max_concurrency=builder.get_max_concurrency()
    )
    yield EvaluatorInfo(
        config=config,
        evaluate_fn=evaluator.evaluate,
        description="DataRobot agent guideline adherence (in-process OOTB guard scorer)",
    )
