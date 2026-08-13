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

import pytest

pytest.importorskip("datarobot_dome")

from nat.data_models.evaluator import EvalInputItem

from datarobot_genai.dragent.eval.common import citations_from_eval_item
from datarobot_genai.dragent.eval.common import coerce_text
from datarobot_genai.dragent.eval.common import interactions_json_from_eval_item


def test_coerce_text_none_and_string() -> None:
    assert coerce_text(None) == ""
    assert coerce_text("hello") == "hello"


def test_coerce_text_non_string() -> None:
    assert coerce_text(42) == "42"


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


def test_interactions_json_from_eval_item_missing() -> None:
    item = EvalInputItem(
        id="3",
        input_obj="hello",
        expected_output_obj="",
        full_dataset_entry={},
    )
    assert interactions_json_from_eval_item(item) is None


def test_interactions_json_from_eval_item_non_dict_entry() -> None:
    item = EvalInputItem(
        id="4",
        input_obj="hello",
        expected_output_obj="",
        full_dataset_entry="not-a-dict",
    )
    assert interactions_json_from_eval_item(item) is None


def test_citations_from_eval_item_context_list() -> None:
    item = EvalInputItem(
        id="5",
        input_obj="q",
        expected_output_obj="",
        full_dataset_entry={"context": ["a", "b"]},
    )
    assert citations_from_eval_item(item) == ["a", "b"]


def test_citations_from_eval_item_context_scalar() -> None:
    item = EvalInputItem(
        id="6",
        input_obj="q",
        expected_output_obj="",
        full_dataset_entry={"context": "single-doc"},
    )
    assert citations_from_eval_item(item) == ["single-doc"]


def test_citations_from_eval_item_citation_columns() -> None:
    item = EvalInputItem(
        id="7",
        input_obj="q",
        expected_output_obj="",
        full_dataset_entry={
            "CITATION_CONTENT_1": "ctx1",
            "CITATION_CONTENT_0": "ctx0",
            "other": "ignored",
        },
    )
    assert citations_from_eval_item(item) == ["ctx1", "ctx0"]


def test_citations_from_eval_item_empty_entry() -> None:
    item = EvalInputItem(
        id="8",
        input_obj="q",
        expected_output_obj="",
        full_dataset_entry={},
    )
    assert citations_from_eval_item(item) == []
