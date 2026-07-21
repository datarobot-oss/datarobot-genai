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

from unittest import mock

import pytest

pytest.importorskip("datarobot_dome")

from datarobot_genai.dragent.eval.scorer_factory import build_agent_goal_accuracy_scorer
from datarobot_genai.dragent.eval.scorer_factory import build_faithfulness_evaluator
from datarobot_genai.dragent.eval.scorer_factory import build_guideline_adherence_scorer
from datarobot_genai.dragent.eval.scorer_factory import build_litellm_judge_target
from datarobot_genai.dragent.eval.scorer_factory import build_task_adherence_scorer
from datarobot_genai.dragent.eval.scorer_factory import resolve_langchain_judge_llm


@pytest.mark.asyncio
async def test_resolve_langchain_judge_llm(mock_eval_builder: mock.Mock) -> None:
    chat_llm = object()
    mock_eval_builder.get_llm.return_value = chat_llm

    result = await resolve_langchain_judge_llm(mock_eval_builder, "judge_llm")

    assert result is chat_llm
    mock_eval_builder.get_llm.assert_awaited_once()


@pytest.mark.asyncio
async def test_build_litellm_judge_target(mock_eval_builder: mock.Mock) -> None:
    from langchain_openai import ChatOpenAI

    mock_eval_builder.get_llm.return_value = ChatOpenAI(model="gpt-4o-mini", api_key="secret")

    target = await build_litellm_judge_target(mock_eval_builder, "judge_llm")

    assert target.model == "openai/gpt-4o-mini"
    assert target.completion_kwargs["api_key"] == "secret"


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


@pytest.mark.asyncio
async def test_build_faithfulness_evaluator(mock_eval_builder: mock.Mock) -> None:
    pytest.importorskip("llama_index.core.evaluation")
    from langchain_openai import ChatOpenAI

    mock_eval_builder.get_llm.return_value = ChatOpenAI(model="gpt-4o-mini", api_key="secret")

    evaluator = await build_faithfulness_evaluator(mock_eval_builder, "judge_llm")

    assert evaluator is not None


@pytest.mark.asyncio
async def test_build_task_adherence_scorer(mock_eval_builder: mock.Mock) -> None:
    pytest.importorskip("deepeval.metrics")
    from langchain_openai import ChatOpenAI

    mock_eval_builder.get_llm.return_value = ChatOpenAI(model="gpt-4o-mini", api_key="secret")

    scorer = await build_task_adherence_scorer(mock_eval_builder, "judge_llm")

    assert scorer is not None


@pytest.mark.asyncio
async def test_build_guideline_adherence_scorer(mock_eval_builder: mock.Mock) -> None:
    pytest.importorskip("llama_index.core.evaluation")
    from langchain_openai import ChatOpenAI

    mock_eval_builder.get_llm.return_value = ChatOpenAI(model="gpt-4o-mini", api_key="secret")

    scorer = await build_guideline_adherence_scorer(
        mock_eval_builder,
        "judge_llm",
        "Be professional.",
    )

    assert scorer is not None


@pytest.mark.asyncio
async def test_build_guideline_adherence_scorer_requires_guideline(
    mock_eval_builder: mock.Mock,
) -> None:
    with pytest.raises(ValueError, match="agent_guideline is required"):
        await build_guideline_adherence_scorer(mock_eval_builder, "judge_llm", "   ")
