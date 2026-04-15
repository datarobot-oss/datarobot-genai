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

from ag_ui.core import RunAgentInput
from ag_ui.core import UserMessage
from langgraph.types import Interrupt

from datarobot_genai.langgraph.hitl import LANGGRAPH_RESUME_STATE_KEY
from datarobot_genai.langgraph.hitl import extract_langgraph_resume
from datarobot_genai.langgraph.hitl import interrupts_to_ag_ui_value


def test_extract_langgraph_resume_from_state() -> None:
    inp = RunAgentInput(
        thread_id="t",
        run_id="r",
        state={LANGGRAPH_RESUME_STATE_KEY: "human says yes"},
        messages=[UserMessage(content="x", id="m")],
        tools=[],
        context=[],
        forwarded_props={},
    )
    assert extract_langgraph_resume(inp) == "human says yes"


def test_extract_langgraph_resume_from_forwarded_props() -> None:
    inp = RunAgentInput(
        thread_id="t",
        run_id="r",
        state={},
        messages=[UserMessage(content="x", id="m")],
        tools=[],
        context=[],
        forwarded_props={LANGGRAPH_RESUME_STATE_KEY: {"ok": True}},
    )
    assert extract_langgraph_resume(inp) == {"ok": True}


def test_extract_langgraph_resume_absent() -> None:
    inp = RunAgentInput(
        thread_id="t",
        run_id="r",
        state={},
        messages=[UserMessage(content="x", id="m")],
        tools=[],
        context=[],
        forwarded_props={},
    )
    assert extract_langgraph_resume(inp) is None


def test_interrupts_to_ag_ui_value() -> None:
    intr = Interrupt(value={"question": "ok?"}, id="id-1")
    out = interrupts_to_ag_ui_value((intr,))
    assert out["kind"] == "langgraph_interrupt"
    assert out["interrupts"] == [{"id": "id-1", "value": {"question": "ok?"}}]
