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

import json
from unittest import mock

import pytest

pytest.importorskip("datarobot_dome")

from datarobot_dome.guards.agent_goal_accuracy import AgentGoalAccuracyEvaluator
from nat.data_models.evaluator import EvalInputItem

from datarobot_genai.dragent.eval.agent_goal_accuracy import score_agent_goal_accuracy_item
from datarobot_genai.dragent.eval.common import citations_from_eval_item
from datarobot_genai.dragent.eval.common import interactions_json_from_eval_item
from datarobot_genai.dragent.eval.faithfulness import score_faithfulness_item
from datarobot_genai.dragent.eval.guideline_adherence import score_guideline_adherence_item
from datarobot_genai.dragent.eval.litellm_target import langchain_chat_model_to_litellm
from datarobot_genai.dragent.eval.task_adherence import score_task_adherence_item


def test_interactions_json_from_eval_item_string_column() -> None:
    item = EvalInputItem(
        id="1",
        input_obj="hello",
        expected_output_obj="",
        full_dataset_entry={"pipeline_interactions": '{"user_input": []}'},
    )
    assert interactions_json_from_eval_item(item) == '{"user_input": []}'


def test_interactions_json_from_eval_item_dict_column() -> None:
    payload = {"user_input": [{"type": "human", "content": "hi"}]}
    item = EvalInputItem(
        id="2",
        input_obj="hello",
        expected_output_obj="",
        full_dataset_entry={"pipelineInteractions": payload},
    )
    assert json.loads(interactions_json_from_eval_item(item) or "") == payload


def test_citations_from_eval_item_context_list() -> None:
    item = EvalInputItem(
        id="3",
        input_obj="q",
        expected_output_obj="",
        full_dataset_entry={"context": ["a", "b"]},
    )
    assert citations_from_eval_item(item) == ["a", "b"]


def test_citations_from_eval_item_citation_ctolumns() -> None:
    item = EvalInputItem(
        id="4",
        input_obj="q",
        expected_output_obj="",
        full_dataset_entry={"CITATION_CONTENT_0": "ctx1"},
    )
    assert citations_from_eval_item(item) == ["ctx1"]


@pytest.mark.asyncio
async def test_score_agent_goal_accuracy_item_delegates_to_guard_helper() -> None:
    item = EvalInputItem(
        id="row-1",
        input_obj="task",
        expected_output_obj="",
        output_obj="done",
        full_dataset_entry={},
    )
    scorer = mock.Mock(spec=AgentGoalAccuracyEvaluator)
    with mock.patch(
        "datarobot_genai.dragent.eval.agent_goal_accuracy.calculate_agent_goal_accuracy",
        new=mock.AsyncMock(return_value=1.0),
    ) as mock_calc:
        result = await score_agent_goal_accuracy_item(scorer, item)

    mock_calc.assert_awaited_once_with(scorer, "task", None, "done")
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_score_faithfulness_item_delegates_to_guard_helper() -> None:
    item = EvalInputItem(
        id="row-2",
        input_obj="q",
        expected_output_obj="",
        output_obj="answer",
        full_dataset_entry={"context": ["doc"]},
    )
    evaluator = mock.Mock()
    with mock.patch(
        "datarobot_genai.dragent.eval.faithfulness.calculate_faithfulness",
        return_value=1.0,
    ) as mock_calc:
        result = await score_faithfulness_item(evaluator, item)

    mock_calc.assert_called_once()
    assert result.score == 1.0
    assert result.reasoning["context_count"] == 1


@pytest.mark.asyncio
async def test_score_task_adherence_item_delegates_to_guard_helper() -> None:
    item = EvalInputItem(
        id="row-3",
        input_obj="task",
        expected_output_obj="",
        output_obj="done",
        full_dataset_entry={},
    )
    scorer = mock.Mock()
    with mock.patch(
        "datarobot_genai.dragent.eval.task_adherence.calculate_task_adherence",
        return_value=0.8,
    ) as mock_calc:
        result = await score_task_adherence_item(scorer, item)

    mock_calc.assert_called_once()
    assert result.score == 0.8


@pytest.mark.asyncio
async def test_score_guideline_adherence_item_delegates_to_guard_helper() -> None:
    item = EvalInputItem(
        id="row-4",
        input_obj="task",
        expected_output_obj="",
        output_obj="done",
        full_dataset_entry={},
    )
    scorer = mock.Mock()
    with mock.patch(
        "datarobot_genai.dragent.eval.guideline_adherence.calculate_agent_guideline_adherence",
        new=mock.AsyncMock(return_value=True),
    ) as mock_calc:
        result = await score_guideline_adherence_item(scorer, item)

    mock_calc.assert_awaited_once()
    assert result.score == 1.0


def test_langchain_chat_model_to_litellm_openai() -> None:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", api_key="secret")
    model, kwargs = langchain_chat_model_to_litellm(llm)
    assert model == "openai/gpt-4o-mini"
    assert kwargs["api_key"] == "secret"


def test_langchain_chat_model_to_litellm_chat_litellm() -> None:
    from langchain_litellm import ChatLiteLLM

    llm = ChatLiteLLM(
        model="datarobot/anthropic/claude-3",
        api_key="token",
        api_base="https://app.datarobot.com",
        extra_headers={"X-Custom": "1"},
        model_kwargs={"extra_body": {"reasoning": {"enabled": False}}},
    )
    model, kwargs = langchain_chat_model_to_litellm(llm)
    assert model == "datarobot/anthropic/claude-3"
    assert kwargs["api_key"] == "token"
    assert kwargs["api_base"] == "https://app.datarobot.com"
    assert kwargs["extra_headers"] == {"X-Custom": "1"}
    assert kwargs["extra_body"] == {"reasoning": {"enabled": False}}
