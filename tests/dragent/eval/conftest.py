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

from collections.abc import Callable
from typing import Any
from unittest import mock

import pytest

pytest.importorskip("datarobot_dome")

from nat.data_models.evaluator import EvalInputItem


@pytest.fixture
def make_eval_item() -> Callable[..., EvalInputItem]:
    def _make(**overrides: Any) -> EvalInputItem:
        defaults: dict[str, Any] = {
            "id": "row-1",
            "input_obj": "question",
            "expected_output_obj": "",
            "output_obj": "answer",
            "full_dataset_entry": {},
        }
        defaults.update(overrides)
        return EvalInputItem(**defaults)

    return _make


@pytest.fixture
def mock_eval_builder() -> mock.Mock:
    builder = mock.Mock()
    builder.get_max_concurrency.return_value = 2
    builder.get_llm = mock.AsyncMock()
    return builder
