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

"""Human-in-the-loop (HITL) helpers for LangGraph agents.

LangGraph uses :func:`langgraph.types.interrupt` inside nodes and
:class:`langgraph.types.Command` with ``resume=...`` to continue. Interrupts
require a checkpointer when compiling the graph.

Clients resume by sending the next request with ``state["langgraph_resume"]`` set
to the resume value (or a mapping of interrupt id → value). See
:class:`datarobot_genai.langgraph.agent.LangGraphAgent`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ag_ui.core import RunAgentInput
from langgraph.types import Interrupt

# RunAgentInput.state / forwarded_props key for Command(resume=...)
LANGGRAPH_RESUME_STATE_KEY = "langgraph_resume"


def extract_langgraph_resume(run_agent_input: RunAgentInput) -> Any | None:
    """Return the LangGraph resume payload if the run is a HITL continuation.

    Checks ``run_agent_input.state`` and ``run_agent_input.forwarded_props`` for
    :data:`LANGGRAPH_RESUME_STATE_KEY`.
    """
    state = run_agent_input.state
    if isinstance(state, Mapping) and LANGGRAPH_RESUME_STATE_KEY in state:
        return state[LANGGRAPH_RESUME_STATE_KEY]
    forwarded = run_agent_input.forwarded_props
    if isinstance(forwarded, Mapping) and LANGGRAPH_RESUME_STATE_KEY in forwarded:
        return forwarded[LANGGRAPH_RESUME_STATE_KEY]
    return None


def interrupts_to_ag_ui_value(interrupts: tuple[Interrupt, ...]) -> dict[str, Any]:
    """Serialize LangGraph :class:`~langgraph.types.Interrupt` tuples for AG-UI."""
    serialized: list[dict[str, Any]] = []
    for intr in interrupts:
        serialized.append(
            {
                "id": intr.id,
                "value": intr.value,
            }
        )
    return {
        "kind": "langgraph_interrupt",
        "interrupts": serialized,
    }
