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

from crewai import Agent
from crewai import Crew
from crewai import Task
from crewai.tools import tool
from datarobot_genai.crewai.agent import datarobot_agent_class_from_crew
from datarobot_genai.crewai.llm import get_llm
from datarobot_genai.drtools.calculator import calculator

from dragent.tool import generate_objectid

generate_objectid_tool = tool(generate_objectid)
calculator_tool = tool(calculator)

llm = get_llm(model_name="datarobot/azure-openai-gpt-5-codex")

agent_planner = Agent(
    role="Planner",
    goal=(
        "Call any required tool for {topic}. Reply with only the tool's result, "
        "or 1 brief line if no tool is needed."
    ),
    backstory="Replies with tool result only, or 1 brief line.",
    llm=llm,
    tools=[generate_objectid_tool, calculator_tool],
)

agent_writer = Agent(
    role="Writer",
    goal="Reply about {topic} with only the tool's result, or 1 brief line.",
    backstory="Replies with tool result only, or 1 brief line.",
    llm=llm,
    tools=[generate_objectid_tool, calculator_tool],
)

agents = [agent_planner, agent_writer]

task_planner = Task(
    description=(
        "Topic: {topic}. History: {chat_history}. Call any required tool. "
        "Reply with only the tool's result, or 1 brief line if no tool is needed."
    ),
    expected_output="Tool result, or 1 brief line.",
    agent=agent_planner,
)

task_writer = Task(
    description=(
        "Topic: {topic}. Call any required tool. "
        "Reply with only the tool's result, or 1 brief line."
    ),
    expected_output="Tool result, or 1 brief line.",
    agent=agent_writer,
)

tasks = [task_planner, task_writer]

crew = Crew(agents=agents, tasks=tasks, stream=True)

kickoff_inputs = lambda user_prompt_content: {
    "topic": user_prompt_content,
    "chat_history": "",
}

MyAgent = datarobot_agent_class_from_crew(crew, agents, tasks, kickoff_inputs)
