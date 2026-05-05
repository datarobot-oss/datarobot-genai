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

from datarobot_genai.dragent.frontends.tool_call_registry import bind_tool_call
from datarobot_genai.dragent.frontends.tool_call_registry import pop_tool_call
from datarobot_genai.dragent.frontends.tool_call_registry import register_tool_call
from datarobot_genai.dragent.frontends.tool_call_registry import reset


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:
    reset()
    yield
    reset()


def test_pop_returns_none_when_nothing_registered() -> None:
    assert pop_tool_call("nat-uuid-missing") is None


def test_bind_returns_none_when_no_pending_for_name() -> None:
    # Internal NAT functions that were not dispatched as a tool by the LLM
    # arrive with a name that has no registered tool_call_id.
    assert bind_tool_call("internal-helper", "nat-uuid-1") is None


def test_register_then_bind_then_pop_round_trip() -> None:
    register_tool_call("planner", "tc-1")

    assert bind_tool_call("planner", "nat-uuid-1") == "tc-1"
    assert pop_tool_call("nat-uuid-1") == "tc-1"
    assert pop_tool_call("nat-uuid-1") is None


def test_binds_in_dispatch_order_per_name() -> None:
    # The LLM dispatches two same-name calls; FUNCTION_START fires for each
    # in dispatch order, so registering and binding must pair them as
    # tc-1->nat-1 and tc-2->nat-2 regardless of completion order.
    register_tool_call("planner", "tc-1")
    register_tool_call("planner", "tc-2")

    assert bind_tool_call("planner", "nat-1") == "tc-1"
    assert bind_tool_call("planner", "nat-2") == "tc-2"
    assert bind_tool_call("planner", "nat-3") is None


def test_pop_by_uuid_tolerates_out_of_order_completion() -> None:
    register_tool_call("planner", "tc-1")
    register_tool_call("planner", "tc-2")
    bind_tool_call("planner", "nat-1")
    bind_tool_call("planner", "nat-2")

    # Second dispatch finishes first.
    assert pop_tool_call("nat-2") == "tc-2"
    assert pop_tool_call("nat-1") == "tc-1"


def test_isolates_per_name() -> None:
    register_tool_call("planner", "tc-1")
    register_tool_call("writer", "tc-2")

    assert bind_tool_call("writer", "nat-w") == "tc-2"
    assert bind_tool_call("planner", "nat-p") == "tc-1"
    assert pop_tool_call("nat-p") == "tc-1"
    assert pop_tool_call("nat-w") == "tc-2"


def test_reset_clears_pending() -> None:
    register_tool_call("planner", "tc-1")
    bind_tool_call("planner", "nat-1")
    reset()

    assert pop_tool_call("nat-1") is None
    assert bind_tool_call("planner", "nat-1") is None
