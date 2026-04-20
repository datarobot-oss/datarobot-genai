# LlamaIndex + DataRobot

Build a multi-agent LlamaIndex workflow and run it as a DataRobot agent.

## Installation

```bash
pip install "datarobot-genai[llamaindex]"
```

## Key imports

```python
from datarobot_genai.llama_index.llm import get_llm
from datarobot_genai.llama_index.agent import datarobot_agent_class_from_llamaindex
from datarobot_genai.core.agents import make_system_prompt
```

## How it works

1. **`get_llm()`** returns a LlamaIndex `LiteLLM` instance backed by the DataRobot LLM Gateway. It reads `DATAROBOT_API_TOKEN` and `DATAROBOT_ENDPOINT` from the environment.

2. **Define your workflow** using the standard LlamaIndex `AgentWorkflow` API. Create `FunctionAgent` instances, wire them together, and build an `AgentWorkflow`.

3. **`datarobot_agent_class_from_llamaindex(workflow, agents, extract_response_text)`** produces a `LlamaIndexAgent` subclass that:
   - Runs the workflow and streams events from `handler.stream_events()`.
   - Emits AG-UI events (text messages, tool calls, agent-switch steps) to the DataRobot frontend.
   - Propagates the platform-injected LLM and tools to every agent at runtime.

### `extract_response_text` callback

You must provide a function that extracts the final text from the workflow result. This is framework-specific because LlamaIndex workflows can return state in different shapes:

```python
def extract_response_text(result_state, events):
    for event in reversed(events):
        resp = getattr(event, "response", None)
        if resp is not None:
            content = getattr(resp, "content", None)
            if content:
                return str(content)
    return ""
```

## Agent handoffs

LlamaIndex agents can hand off to each other via `can_handoff_to`. The wrapper automatically emits `StepStartedEvent` / `StepFinishedEvent` for each agent transition.

```python
planner = FunctionAgent(
    name="planner",
    ...,
    can_handoff_to=["writer"],
)
```

## Example

See [agent_example.py](agent_example.py) for a complete, runnable two-agent workflow.

```bash
export DATAROBOT_API_TOKEN="<token>"
export DATAROBOT_ENDPOINT="https://app.datarobot.com/api/v2"

python docs/llamaindex/agent_example.py
```

## Developing from this example

See [AGENTS.md](AGENTS.md) for how to extend the workflow, tools, prompts, and `extract_response_text`.
