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

"""Minimal CrewAI agent running on DataRobot.

Two-agent crew: a planner creates a bullet-point outline, then a writer turns
it into a short paragraph.  Uses the DataRobot LLM Gateway for model calls.

Prerequisites
-------------
pip install "datarobot-genai[crewai]"

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

from ag_ui.core import Message
from ag_ui.core import RunAgentInput
from crewai import Agent
from crewai import Crew
from crewai import Task

from datarobot_genai.core.agents import make_system_prompt
from datarobot_genai.crewai.agent import datarobot_agent_class_from_crew
from datarobot_genai.crewai.llm import get_llm

# ---------------------------------------------------------------------------
# 1. Get a DataRobot-backed LLM
# ---------------------------------------------------------------------------
llm = get_llm()

# ---------------------------------------------------------------------------
# 2. Define CrewAI agents and tasks
# ---------------------------------------------------------------------------
agent_planner = Agent(
    role="Content Planner",
    goal="Create a short bullet-point outline with 3-5 key points about: {topic}.",
    backstory=make_system_prompt(
        "You are a content planner. Given a topic, produce a short "
        "bullet-point outline with 3-5 key points."
    ),
    llm=llm,
)

agent_writer = Agent(
    role="Content Writer",
    goal="Write a 2-3 sentence response based on the planner's outline about: {topic}.",
    backstory=make_system_prompt(
        "You are a concise writer. Using the planner's outline, "
        "write a short response in 2-3 sentences."
    ),
    llm=llm,
)

agents = [agent_planner, agent_writer]

task_planner = Task(
    description="Create a short outline about: {topic}.",
    expected_output="A bullet-point outline with 3-5 key points.",
    agent=agent_planner,
)

task_writer = Task(
    description="Using the planner's outline, write a short response about: {topic}.",
    expected_output="A concise 2-3 sentence response.",
    agent=agent_writer,
)

tasks = [task_planner, task_writer]

crew = Crew(agents=agents, tasks=tasks, stream=True)

# ---------------------------------------------------------------------------
# 3. Wrap into a DataRobot agent class
# ---------------------------------------------------------------------------


def kickoff_inputs(user_prompt: str) -> dict[str, str]:
    return {
        "topic": user_prompt,
        "chat_history": "",
    }


MyAgent = datarobot_agent_class_from_crew(crew, agents, tasks, kickoff_inputs)


# ---------------------------------------------------------------------------
# 4. Run it locally
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
