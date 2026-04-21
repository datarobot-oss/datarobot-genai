# CrewAI + DataRobot

Build a CrewAI crew and run it as a DataRobot agent.

## Installation

```bash
pip install "datarobot-genai[crewai]"
```

## Key imports

```python
from datarobot_genai.crewai.llm import get_llm
from datarobot_genai.crewai.agent import datarobot_agent_class_from_crew
from datarobot_genai.core.agents import make_system_prompt
```

## How it works

1. **`get_llm()`** returns a CrewAI `LLM` instance backed by the DataRobot LLM Gateway (via LiteLLM). It reads `DATAROBOT_API_TOKEN` and `DATAROBOT_ENDPOINT` from the environment.

2. **Define your crew** using the standard CrewAI `Agent`, `Task`, and `Crew` API. Define agents with roles, goals, and backstories; tasks with descriptions and expected outputs.

3. **`datarobot_agent_class_from_crew(crew, agents, tasks, kickoff_inputs)`** produces a `CrewAIAgent` subclass that:
   - Runs `crew.kickoff_async()` with your inputs.
   - Streams AG-UI events (text, reasoning, tool calls, step transitions) to the DataRobot frontend.
   - Propagates the platform-injected LLM and tools to every agent at runtime.
   - Tracks token usage from CrewAI output.

### `kickoff_inputs` callable

You provide a function that maps the raw user prompt to the dictionary of inputs your tasks expect:

```python
kickoff_inputs = lambda user_prompt: {
    "topic": user_prompt,
    "chat_history": "",   # include this key to opt into automatic history injection
}
```

The keys must match the `{placeholders}` used in your task descriptions and agent goals.

## Chat history

History injection is **opt-in**. Include a `"chat_history"` key with an empty string value in your kickoff inputs. The wrapper auto-populates it with prior conversation turns. Use `{chat_history}` in your task descriptions or agent backstories.

## Streaming

CrewAI crews are created with `stream=True` by default in the wrapper. The agent streams text chunks, reasoning steps, and tool calls as they happen.

## Example

See [agent_example.py](agent_example.py) for a complete, runnable planner + writer crew.

```bash
export DATAROBOT_API_TOKEN="<token>"
export DATAROBOT_ENDPOINT="https://app.datarobot.com/api/v2"

python docs/crewai/agent_example.py
```

## Developing from this example

See [AGENTS.md](AGENTS.md) for how to extend the crew, tasks, `kickoff_inputs`, and tools.
