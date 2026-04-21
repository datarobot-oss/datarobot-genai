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

"""Minimal LangGraph agent running on DataRobot.

Two-agent pipeline: a planner creates a bullet-point outline, then a writer
turns it into a short paragraph.  Uses the DataRobot LLM Gateway for model
calls.

Prerequisites
-------------
pip install "datarobot-genai[langgraph]"

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
from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph

from datarobot_genai.core.agents import make_system_prompt
from datarobot_genai.langgraph.agent import datarobot_agent_class_from_langgraph
from datarobot_genai.langgraph.llm import get_llm

# ---------------------------------------------------------------------------
# 1. Get a DataRobot-backed LLM
# ---------------------------------------------------------------------------
llm = get_llm()

# ---------------------------------------------------------------------------
# 2. Define the LangGraph workflow via a factory function
# ---------------------------------------------------------------------------
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. {chat_history}"),
        ("user", "{topic}"),
    ]
)


def graph_factory(
    llm: BaseChatModel,
    tools: list[BaseTool],
    verbose: bool = False,
) -> StateGraph[MessagesState]:
    planner = create_agent(
        llm,
        tools=tools,
        system_prompt=make_system_prompt(
            "You are a content planner. Given a topic, produce a short "
            "bullet-point outline with 3-5 key points."
        ),
        name="planner",
        debug=verbose,
    )

    writer = create_agent(
        llm,
        tools=tools,
        system_prompt=make_system_prompt(
            "You are a concise writer. Using the planner's outline, "
            "write a short response in 2-3 sentences."
        ),
        name="writer",
        debug=verbose,
    )

    graph = StateGraph(MessagesState)
    graph.add_node("planner", planner)
    graph.add_node("writer", writer)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "writer")
    graph.add_edge("writer", END)
    return graph


# ---------------------------------------------------------------------------
# 3. Wrap into a DataRobot agent class
# ---------------------------------------------------------------------------
MyAgent = datarobot_agent_class_from_langgraph(graph_factory, prompt_template)


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
