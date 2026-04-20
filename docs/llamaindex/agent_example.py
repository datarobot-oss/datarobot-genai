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

"""Minimal LlamaIndex agent running on DataRobot.

Two-agent pipeline: a planner creates a bullet-point outline, then hands off
to a writer that turns it into a short paragraph.  Uses the DataRobot LLM
Gateway for model calls.

Prerequisites
-------------
pip install "datarobot-genai[llamaindex]"

Environment variables
---------------------
DATAROBOT_API_TOKEN  – your DataRobot API token
DATAROBOT_ENDPOINT   – e.g. https://app.datarobot.com/api/v2

How to extend this example
--------------------------
See AGENTS.md next to this file.
"""

import asyncio
import uuid
from typing import Any

from ag_ui.core import Message
from ag_ui.core import RunAgentInput
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.agent.workflow import FunctionAgent

from datarobot_genai.core.agents import make_system_prompt
from datarobot_genai.llama_index.agent import datarobot_agent_class_from_llamaindex
from datarobot_genai.llama_index.llm import get_llm

# ---------------------------------------------------------------------------
# 1. Get a DataRobot-backed LLM
# ---------------------------------------------------------------------------
llm = get_llm()

# ---------------------------------------------------------------------------
# 2. Define LlamaIndex agents and workflow
# ---------------------------------------------------------------------------
planner = FunctionAgent(
    name="planner",
    description="Creates short bullet-point outlines",
    system_prompt=make_system_prompt(
        "You are a content planner. Given a topic, produce a short "
        "bullet-point outline with 3-5 key points."
    ),
    llm=llm,
    can_handoff_to=["writer"],
)

writer = FunctionAgent(
    name="writer",
    description="Writes concise responses based on outlines",
    system_prompt=make_system_prompt(
        "You are a concise writer. Using the planner's outline, "
        "write a short response in 2-3 sentences."
    ),
    llm=llm,
)

agents = [planner, writer]
workflow = AgentWorkflow(agents=agents, root_agent="planner")


# ---------------------------------------------------------------------------
# 3. Response extractor
# ---------------------------------------------------------------------------
def extract_response_text(result_state: Any, events: list[Any]) -> str:
    for event in reversed(events):
        resp = getattr(event, "response", None)
        if resp is not None:
            content = getattr(resp, "content", None)
            if content:
                return str(content)
    return ""


# ---------------------------------------------------------------------------
# 4. Wrap into a DataRobot agent class
# ---------------------------------------------------------------------------
MyAgent = datarobot_agent_class_from_llamaindex(workflow, agents, extract_response_text)


# ---------------------------------------------------------------------------
# 5. Run it locally
# ---------------------------------------------------------------------------
async def main() -> None:
    agent = MyAgent(llm=llm)

    run_input = RunAgentInput(
        thread_id=str(uuid.uuid4()),
        run_id=str(uuid.uuid4()),
        messages=[Message(role="user", content="Benefits of Python for data science")],
    )

    async for event, _interactions, usage in agent.invoke(run_input):
        print(event)

    print(f"\nToken usage: {usage}")


if __name__ == "__main__":
    asyncio.run(main())
