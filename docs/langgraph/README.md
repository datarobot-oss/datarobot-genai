# LangGraph + DataRobot

Build a multi-agent LangGraph workflow and run it as a DataRobot agent.

## Installation

```bash
pip install "datarobot-genai[langgraph]"
```

## Key imports

```python
from datarobot_genai.langgraph.llm import get_llm
from datarobot_genai.langgraph.agent import datarobot_agent_class_from_langgraph
from datarobot_genai.core.agents import make_system_prompt
```

## How it works

1. **`get_llm()`** returns a LangChain `BaseChatModel` backed by the DataRobot LLM Gateway (via LiteLLM). It reads `DATAROBOT_API_TOKEN` and `DATAROBOT_ENDPOINT` from the environment.

2. **Define your graph** using the standard LangGraph `StateGraph` API. Write a *graph factory* function with the signature:

   ```python
   def graph_factory(
       llm: BaseChatModel,
       tools: list[BaseTool],
       verbose: bool,
   ) -> StateGraph[MessagesState]:
       ...
   ```

   The factory receives the LLM, any platform-injected tools, and a verbosity flag. Return an **uncompiled** `StateGraph` — the wrapper compiles it at invocation time.

3. **`datarobot_agent_class_from_langgraph(graph_factory, prompt_template)`** produces a `LangGraphAgent` subclass (inheriting from `BaseAgent`) that:
   - Compiles and runs your graph on every call.
   - Streams AG-UI events (text messages, tool calls, run lifecycle) to the DataRobot frontend.
   - Automatically tracks token usage.

## Chat history

History injection is **opt-in**. If your `ChatPromptTemplate` declares a `{chat_history}` input variable, prior conversation turns are automatically included. Otherwise no history is passed.

```python
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. {chat_history}"),
    ("user", "{topic}"),
])
```

## Example

See [agent_example.py](agent_example.py) for a complete, runnable two-agent (planner + writer) workflow.

```bash
# Set environment
export DATAROBOT_API_TOKEN="<token>"
export DATAROBOT_ENDPOINT="https://app.datarobot.com/api/v2"

python docs/langgraph/agent_example.py
```

## Developing from this example

See [AGENTS.md](AGENTS.md) for how to extend the graph, tools, prompts, and integration points.
