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

# Developing from this LlamaIndex example

Use [agent_example.py](agent_example.py) as a starting point for a DataRobot-backed LlamaIndex `AgentWorkflow`.

## What this example does

- **`get_llm()`** — returns a LlamaIndex LiteLLM client pointed at the DataRobot LLM Gateway.
- **`FunctionAgent` instances** — each has a name, system prompt, `llm`, optional tools, and optional `can_handoff_to` for multi-agent routing.
- **`AgentWorkflow`** — `root_agent` is the entry agent; handoffs follow `can_handoff_to`.
- **`extract_response_text(state, events)`** — required by `datarobot_agent_class_from_llamaindex`. Adjust it if your workflow stores the final answer elsewhere (for example last `AgentOutput` event).
- **`datarobot_agent_class_from_llamaindex(workflow, agents, extract_response_text)`** — at runtime, `set_llm` / `set_tools` on the DataRobot class updates every `FunctionAgent` in `agents`.

## Typical changes

1. **More agents** — add another `FunctionAgent`, list it in `agents`, wire `can_handoff_to` / workflow edges as LlamaIndex expects, and set `AgentWorkflow(..., root_agent="...")` appropriately.
2. **Tools** — attach `FunctionTool` (or LlamaIndex tools) to each agent. Injected platform tools are appended in the wrapper; keep agent-specific tools on the agent when you build the workflow.
3. **Model** — pass `model_name="datarobot/your-model"` to `get_llm(...)` if you do not rely on `LLM_DEFAULT_MODEL` in the environment.
4. **History / memory** — in user messages you can include placeholders `{chat_history}` and `{memory}` in the string sent to `workflow.run`; see `LlamaIndexAgent.invoke` in the library for behavior.
5. **Stronger extraction** — if streaming shows the right text but evaluation needs a single string, refine `extract_response_text` to match your event stream.

## Run and debug

```bash
export DATAROBOT_API_TOKEN="..."
export DATAROBOT_ENDPOINT="https://app.datarobot.com/api/v2"
python docs/llamaindex/agent_example.py
```

## Where to look in the library

- `datarobot_genai.llama_index.agent` — `LlamaIndexAgent`, `datarobot_agent_class_from_llamaindex`
- `datarobot_genai.llama_index.llm` — `get_llm`
- `datarobot_genai.core.agents` — `make_system_prompt`, `BaseAgent`

## Reference implementation

See `e2e-tests/dragent/llamaindex/myagent.py` for the same planner/writer pattern with tools and gateway model naming.
