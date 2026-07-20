# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A NeMo Agent Toolkit custom evaluator that scores workflow outputs with DataRobot moderations.

Wired into a workflow's ``eval.evaluators`` block as ``_type: moderations``. ``nat eval`` runs each
dataset row through the workflow, then hands the output (and trajectory) here for scoring by the
moderation guards configured in ``moderation_config``.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

from datarobot_dome.api import ModerationPipeline
from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvalInputItem
from nat.data_models.evaluator import EvaluatorBaseConfig
from nat.data_models.intermediate_step import IntermediateStepType
from nat.plugins.eval.data_models.evaluator_io import EvalOutputItem
from nat.plugins.eval.evaluator.base_evaluator import BaseEvaluator
from pydantic import Field

# ``pipeline_interactions`` (the trajectory-aware path agent_goal_accuracy needs) was added to the
# ModerationPipeline API in a later datarobot-moderations release. Feature-detect it so the
# evaluator still runs on older installs, falling back to a single prompt/response pair there.
_SUPPORTS_PIPELINE_INTERACTIONS = (
    "pipeline_interactions"
    in inspect.signature(ModerationPipeline.evaluate_response_async).parameters
)


class ModerationsEvaluatorConfig(EvaluatorBaseConfig, name="moderations"):
    """Config for the moderations evaluator (``_type: moderations``)."""

    moderation_config: str = Field(description="Path to the moderation guard-config YAML.")
    metric: str = Field(
        default="agent_goal_accuracy",
        description="Key in EvaluationResult.metrics to report as the per-item score.",
    )


def _native(value: Any) -> Any:
    """Coerce numpy scalars (guard scores arrive as numpy dtypes) to JSON-serialisable natives."""
    return value.item() if hasattr(value, "item") else value


def _message_text(value: Any) -> str:
    """Best-effort text extraction from an IntermediateStep's ``data.output``."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    content = getattr(value, "content", None)
    return content if isinstance(content, str) else str(value)


def _interactions_from_trajectory(item: EvalInputItem) -> dict[str, Any]:
    """Rebuild the ``MultiTurnSample`` wire dict that ``nat eval`` does not carry natively.

    ``nat eval`` exposes NAT ``IntermediateStep``s, not the dragent chat-completion's assembled
    ``pipeline_interactions``. The ``agent_goal_accuracy`` guard needs the latter, so we map the
    steps into the plain ``{"user_input": [{"content", "type"}]}`` wire form the guard deserialises
    (via ragas / the vendored ``MultiTurnSample``). We emit the dict directly rather than import a
    version-specific message class. Ordering honours the ragas rule that a tool turn must follow an
    AI turn, and the conversation always ends on the final answer.
    """
    turns: list[dict[str, Any]] = [{"content": str(item.input_obj), "type": "human"}]
    seen_ai = False
    for step in item.trajectory:
        text = _message_text(step.data.output if step.data else None)
        if step.event_type == IntermediateStepType.LLM_END:
            turns.append({"content": text, "type": "ai"})
            seen_ai = True
        elif step.event_type == IntermediateStepType.TOOL_END and seen_ai:
            turns.append({"content": text, "type": "tool"})
    if turns[-1]["type"] != "ai":
        turns.append({"content": str(item.output_obj), "type": "ai"})
    return {"user_input": turns}


def _pipeline_interactions(item: EvalInputItem) -> str:
    """Prefer a real ``pipeline_interactions`` on the dataset entry; else convert the trajectory.

    A future ``nat dragent eval`` could thread dragent's own ``pipeline_interactions`` through
    ``full_dataset_entry``; when it does, we pass it straight through instead of reconstructing.
    """
    entry = item.full_dataset_entry if isinstance(item.full_dataset_entry, dict) else {}
    raw = entry.get("pipeline_interactions")
    if raw is None:
        raw = _interactions_from_trajectory(item)
    return raw if isinstance(raw, str) else json.dumps(raw)


class ModerationsEvaluator(BaseEvaluator):
    """Scores each item by running the configured moderation guards over its output."""

    def __init__(self, pipeline: ModerationPipeline, metric: str, max_concurrency: int):
        super().__init__(max_concurrency=max_concurrency, tqdm_desc="Moderations")
        self._pipeline = pipeline
        self._metric = metric

    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        kwargs: dict[str, Any] = {}
        if _SUPPORTS_PIPELINE_INTERACTIONS:
            kwargs["pipeline_interactions"] = _pipeline_interactions(item)
        result, _latency, _df = await self._pipeline.evaluate_response_async(
            response=str(item.output_obj),
            prompt=str(item.input_obj),
            **kwargs,
        )
        return EvalOutputItem(
            id=item.id,
            score=_native(result.metrics.get(self._metric)),
            reasoning={
                "blocked": bool(result.blocked),
                "trajectory_used": _SUPPORTS_PIPELINE_INTERACTIONS,
                "metrics": {key: _native(value) for key, value in result.metrics.items()},
            },
        )


@register_evaluator(config_type=ModerationsEvaluatorConfig)
async def register_moderations_evaluator(config: ModerationsEvaluatorConfig, builder: EvalBuilder):
    # Built here rather than at import: ModerationPipeline.from_yaml eagerly verifies DataRobot
    # credentials over the network, which must not run at module-import (config-parse) time.
    pipeline = ModerationPipeline.from_yaml(config.moderation_config)
    evaluator = ModerationsEvaluator(pipeline, config.metric, builder.get_max_concurrency())
    yield EvaluatorInfo(
        config=config,
        evaluate_fn=evaluator.evaluate,
        description="DataRobot moderations evaluator",
    )
