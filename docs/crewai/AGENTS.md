<!--
  ~ Copyright 2026 DataRobot, Inc. and its affiliates.
  ~
  ~ Licensed under the Apache License, Version 2.0 (the "License");
  ~ you may not use this file except in compliance with the License.
  ~ You may obtain a copy of the License at
  ~
  ~     http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS,
  ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ~ See the License for the specific language governing permissions and
  ~ limitations under the License.
-->

# Developing from this CrewAI example

Use [agent_example.py](agent_example.py) as a starting point for a DataRobot-backed CrewAI crew.

## What this example does

- **`get_llm()`** — returns a CrewAI `LLM` using LiteLLM against the DataRobot LLM Gateway.
- **`Agent` / `Task` / `Crew`** — standard CrewAI objects. Task `description` and agent `goal` / `backstory` use placeholders such as `{topic}`; they must match keys returned by `kickoff_inputs`.
- **`kickoff_inputs(user_prompt)`** — maps the incoming user string to `crew.kickoff_async(inputs=...)`. Include `"chat_history": ""` to opt into automatic multi-turn history (the base class fills it from prior messages).
- **`datarobot_agent_class_from_crew(crew, agents, tasks, kickoff_inputs)`** — merges injected tools with each agent’s original tools on `set_tools`.

## Typical changes

1. **More tasks or agents** — add `Agent` and `Task` rows, order tasks in `tasks`, and keep `Crew(agents=..., tasks=..., stream=True)` aligned with CrewAI’s execution order.
2. **New placeholders** — add keys to `kickoff_inputs` and use `{key}` in task descriptions and agent goals consistently.
3. **Tools** — register CrewAI tools (for example `@tool`) on agents; injected tools are appended by the DataRobot wrapper.
4. **Chat history** — use `{chat_history}` in descriptions and return `"chat_history": ""` from `kickoff_inputs` so the base class can inject a transcript.
5. **Avoid lambda for kickoff** — use a named `def kickoff_inputs(...)` so you can add logic (parsing JSON user prompts, default values).

## Run and debug

```bash
export DATAROBOT_API_TOKEN="..."
export DATAROBOT_ENDPOINT="https://app.datarobot.com/api/v2"
python docs/crewai/agent_example.py
```

Use `verbose=True` on the agent if you need Crew/agent verbosity.

## Where to look in the library

- `datarobot_genai.crewai.agent` — `CrewAIAgent`, `datarobot_agent_class_from_crew`
- `datarobot_genai.crewai.llm` — `get_llm`
- `datarobot_genai.core.agents` — `make_system_prompt`, `BaseAgent`

## Reference implementation

See `e2e-tests/dragent/crewai/myagent.py` for a fuller crew (extra tools, history in tasks).
