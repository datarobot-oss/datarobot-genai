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

# Developing from this LangGraph example

Use [agent_example.py](agent_example.py) as a starting point for a DataRobot-backed LangGraph agent.

## What this example does

- **`get_llm()`** — wires the LangChain chat model to the DataRobot LLM Gateway (env: `DATAROBOT_API_TOKEN`, `DATAROBOT_ENDPOINT`).
- **`graph_factory(llm, tools, verbose)`** — builds an uncompiled `StateGraph`. The platform can inject extra `tools` (for example MCP); merge them into your agents like the e2e tests: `tools=[...] + tools` or pass `tools` into `create_agent`.
- **`prompt_template`** — shapes each user turn. Variables in the template (for example `{topic}`, `{chat_history}`) must match what you pass from the user message (JSON object or plain text mapping in `LangGraphAgent.convert_input_message`).
- **`datarobot_agent_class_from_langgraph(...)`** — produces a class whose `invoke()` streams AG-UI events.

## Typical changes

1. **Add nodes or edges** — extend `graph_factory`: new `graph.add_node`, `add_edge`, or conditional routing. Keep the factory signature `(llm, tools, verbose)` so the wrapper stays valid.
2. **Add tools** — define LangChain `@tool` functions or `StructuredTool` instances and include them in `create_agent(..., tools=[my_tool] + tools, ...)`.
3. **Prompts** — adjust `ChatPromptTemplate` and `make_system_prompt(...)` strings. If you add `{chat_history}` to the template, prior turns are injected automatically.
4. **Structured input** — if the last user message is JSON, keys can fill template variables (for example `"topic": "..."`).
5. **Subclass instead of the helper** — for full control, subclass `LangGraphAgent`, implement `workflow` and `prompt_template`, and keep the same `invoke` contract.

## Run and debug

```bash
export DATAROBOT_API_TOKEN="..."
export DATAROBOT_ENDPOINT="https://app.datarobot.com/api/v2"
python docs/langgraph/agent_example.py
```

Set `verbose=True` on `MyAgent(..., verbose=True)` to enable LangGraph debug output from the factory.

## Where to look in the library

- `datarobot_genai.langgraph.agent` — `LangGraphAgent`, `datarobot_agent_class_from_langgraph`
- `datarobot_genai.langgraph.llm` — `get_llm` and deployment/gateway helpers
- `datarobot_genai.core.agents` — `make_system_prompt`, `BaseAgent`

## Reference implementation

The e2e agent under `e2e-tests/dragent/langgraph/myagent.py` follows the same pattern with an extra tool and NAT-injected LLM; compare when wiring production deployments.
