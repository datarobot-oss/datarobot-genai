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

from collections.abc import Iterator

import pytest

from datarobot_genai.dragent.frontends.tool_call_registry import pop_tool_call
from datarobot_genai.dragent.frontends.tool_call_registry import register_tool_call
from datarobot_genai.dragent.frontends.tool_call_registry import reset


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:
    reset()
    yield
    reset()


def test_pop_returns_none_when_nothing_registered() -> None:
    assert pop_tool_call("missing") is None


def test_register_then_pop_round_trip() -> None:
    register_tool_call("planner", "tc-1")
    assert pop_tool_call("planner") == "tc-1"
    assert pop_tool_call("planner") is None


def test_pops_in_fifo_order_per_name() -> None:
    register_tool_call("planner", "tc-1")
    register_tool_call("planner", "tc-2")

    assert pop_tool_call("planner") == "tc-1"
    assert pop_tool_call("planner") == "tc-2"
    assert pop_tool_call("planner") is None


def test_isolates_per_name() -> None:
    register_tool_call("planner", "tc-1")
    register_tool_call("writer", "tc-2")

    assert pop_tool_call("writer") == "tc-2"
    assert pop_tool_call("planner") == "tc-1"


def test_reset_clears_pending() -> None:
    register_tool_call("planner", "tc-1")
    reset()
    assert pop_tool_call("planner") is None
