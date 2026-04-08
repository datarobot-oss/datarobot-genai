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
from crewai.llm import LLM
from crewai.tools import tool
from datarobot_genai.core.agents import make_system_prompt
from datarobot_genai.crewai.agent import datarobot_agent_class_from_crew
from datarobot_genai.drtools.calculator import calculator

from dragent.tool import generate_objectid

generate_objectid_tool = tool(generate_objectid)
calculator_tool = tool(calculator)

llm = LLM(model="datarobot/azure-openai-gpt-5-codex", is_litellm=True)

agent_planner = Agent(
    role="Content Planner",
    goal="Create a short bullet-point outline with 3-5 key points about: {topic}.",
    backstory=make_system_prompt(
        "You are a content planner. Given a topic, produce a short bullet-point "
        "outline with 3-5 key points. No paragraphs, no explanations — just the list. "
    ),
    llm=llm,
    tools=[generate_objectid_tool, calculator_tool],
)

agent_writer = Agent(
    role="Content Writer",
    goal="Write a 2-3 sentence response based on the planner's outline about: {topic}.",
    backstory=make_system_prompt(
        "You are a concise writer. Using the planner's outline, write a short response "
        "in 2-3 sentences. "
    ),
    llm=llm,
    tools=[generate_objectid_tool, calculator_tool],
)

agents = [agent_planner, agent_writer]

task_planner = Task(
    description=(
        "Create a short outline about: {topic}. "
        "Prior conversation context (may be empty): {chat_history}. "
        "Execute tool calls if requested instead of this task."
    ),
    expected_output=("A bullet-point outline with 3-5 key points. Or the result of a tool call."),
    agent=agent_planner,
)

task_writer = Task(
    description=(
        "Using the planner's outline, write a short response about: {topic}. "
        "Prior conversation context (may be empty): {chat_history}. "
        "Execute tool calls if requested instead of this task."
    ),
    expected_output="A concise 2-3 sentence response. Or the result of a tool call.",
    agent=agent_writer,
)

tasks = [task_planner, task_writer]

crew = Crew(agents=agents, tasks=tasks, stream=True)

kickoff_inputs = lambda user_prompt_content: {  # noqa: E731
    "topic": user_prompt_content,
    "chat_history": "",
}

agent_class = datarobot_agent_class_from_crew(crew, agents, tasks, kickoff_inputs)
