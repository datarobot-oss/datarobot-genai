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
"""Build moderation OOTB scorers from NAT ``EvalBuilder`` judge LLMs.

Each custom NAT evaluator plugin should stay thin (config + ``evaluate_item``).
This module centralizes how a workflow ``llm_name`` is resolved and mapped onto the
same scorer types used by ``moderation_config.yaml`` OOTB guards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from datarobot_dome.guards.agent_goal_accuracy import AgentGoalAccuracyEvaluator

from datarobot_genai.dragent.eval.litellm_target import langchain_chat_model_to_litellm
from datarobot_genai.dragent.eval.litellm_target import wrap_langchain_judge_for_llamaindex

if TYPE_CHECKING:
    from deepeval.metrics import TaskCompletionMetric
    from llama_index.core.evaluation import FaithfulnessEvaluator
    from llama_index.core.evaluation import GuidelineEvaluator
    from nat.builder.builder import EvalBuilder


@dataclass(frozen=True)
class LitellmJudgeTarget:
    """``litellm.acompletion`` model id and kwargs for in-process LLM judges."""

    model: str
    completion_kwargs: dict[str, Any]


async def resolve_langchain_judge_llm(builder: EvalBuilder, llm_name: str) -> object:
    """Resolve ``llm_name`` from the eval/workflow config to a LangChain chat model."""
    from nat.builder.framework_enum import LLMFrameworkEnum

    return await builder.get_llm(llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)


async def build_litellm_judge_target(builder: EvalBuilder, llm_name: str) -> LitellmJudgeTarget:
    """Map the workflow judge LLM to a litellm target (agent goal accuracy, etc.)."""
    llm = await resolve_langchain_judge_llm(builder, llm_name)
    model, completion_kwargs = langchain_chat_model_to_litellm(llm)
    return LitellmJudgeTarget(model=model, completion_kwargs=completion_kwargs)


async def build_agent_goal_accuracy_scorer(
    builder: EvalBuilder, llm_name: str
) -> AgentGoalAccuracyEvaluator:
    target = await build_litellm_judge_target(builder, llm_name)
    return AgentGoalAccuracyEvaluator(
        model=target.model,
        completion_kwargs=target.completion_kwargs,
    )


async def build_faithfulness_evaluator(
    builder: EvalBuilder, llm_name: str
) -> FaithfulnessEvaluator:
    """LlamaIndex faithfulness judge (same wiring as ``OOTBFaithfulnessGuard``)."""
    from datarobot_dome._import_utils import require_extra

    llm = await resolve_langchain_judge_llm(builder, llm_name)

    try:
        from llama_index.core import Settings
        from llama_index.core.evaluation import FaithfulnessEvaluator
    except ImportError as e:
        raise require_extra("llama-index-core", "llm-eval", e) from e

    llamaindex_llm = wrap_langchain_judge_for_llamaindex(llm)
    Settings.llm = llamaindex_llm
    Settings.embed_model = cast(Any, None)
    return FaithfulnessEvaluator()


async def build_task_adherence_scorer(builder: EvalBuilder, llm_name: str) -> TaskCompletionMetric:
    """DeepEval task completion judge (same wiring as ``OOTBTaskAdherenceGuard``)."""
    from datarobot_dome._import_utils import require_extra

    llm = await resolve_langchain_judge_llm(builder, llm_name)

    try:
        from datarobot_dome._deepeval_adapter import ModerationDeepEvalLLM
    except ImportError as e:
        raise require_extra("deepeval", "llm-eval", e) from e
    try:
        from deepeval.metrics import TaskCompletionMetric
    except ImportError as e:
        raise require_extra("deepeval", "llm-eval", e) from e

    deepeval_llm = ModerationDeepEvalLLM(llm)
    return TaskCompletionMetric(model=deepeval_llm, include_reason=True)


async def build_guideline_adherence_scorer(
    builder: EvalBuilder,
    llm_name: str,
    agent_guideline: str,
) -> GuidelineEvaluator:
    """LlamaIndex guideline judge (same wiring as ``OOTBAgentGuidelineAdherence``)."""
    from datarobot_dome._import_utils import require_extra

    if not agent_guideline.strip():
        raise ValueError("agent_guideline is required for guideline adherence evaluation")

    llm = await resolve_langchain_judge_llm(builder, llm_name)

    try:
        from llama_index.core.evaluation import GuidelineEvaluator
        from llama_index.core.evaluation.guideline import EvaluationData
        from llama_index.core.output_parsers import PydanticOutputParser
    except ImportError as e:
        raise require_extra("llama-index-core", "llm-eval", e) from e

    class _FixedPydanticOutputParser(PydanticOutputParser):
        def format(self, query: str) -> str:
            return query + "\n\n" + self.get_format_string(escape_json=False)

    langchain_llm = wrap_langchain_judge_for_llamaindex(llm)
    return GuidelineEvaluator(
        llm=langchain_llm,
        guidelines=agent_guideline,
        output_parser=_FixedPydanticOutputParser(output_cls=EvaluationData),
    )
