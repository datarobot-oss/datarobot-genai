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

from typing import Any

from datarobot_genai.llama_index.agent import datarobot_agent_class_from_llamaindex
from datarobot_genai.llama_index.llm import get_llm
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool

from dragent.tool import generate_objectid

generate_objectid_tool = FunctionTool.from_defaults(fn=generate_objectid, name="generate_objectid")

llm = get_llm(model_name="datarobot/azure-openai-gpt-5-codex")

planner = FunctionAgent(
    name="planner",
    description="Outlines",
    system_prompt="Outline the topic in 1 bullet. Hand off to writer.",
    llm=llm,
    tools=[generate_objectid_tool],
    can_handoff_to=["writer"],
)
writer = FunctionAgent(
    name="writer",
    description="Writes",
    system_prompt="Reply in 1 short sentence using the outline.",
    llm=llm,
    tools=[generate_objectid_tool],
)
agents = [planner, writer]
workflow = AgentWorkflow(agents=agents, root_agent="planner")


def extract_response_text(result_state: Any, events: list[Any]) -> str:
    for event in reversed(events):
        resp = getattr(event, "response", None)
        if resp is not None:
            content = getattr(resp, "content", None)
            if content:
                return str(content)
    return ""


MyAgent = datarobot_agent_class_from_llamaindex(workflow, agents, extract_response_text)
