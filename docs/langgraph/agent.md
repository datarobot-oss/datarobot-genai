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

# LangGraph sample: what you configure

Aligned with [`e2e-tests/dragent/langgraph/`](../../e2e-tests/dragent/langgraph/).

## `workflow.yaml`

| Piece | What you see |
|---|---|
| **`general.front_end._type: dragent_fastapi`** | DRAgent’s HTTP/SSE front end. |
| **`llms.datarobot_llm._type: datarobot-llm-component`** | One named LLM; routing follows [LLM configuration](../llm.md) and env. |
| **`workflow._type: langgraph_agent`** | Tells DRAgent to use the LangGraph integration. |
| **`workflow.llm_name`** | Must match the key under **`llms:`**. |
| **`workflow.description` / `verbose`** | Shown in tooling and logs. |

You do **not** describe the graph in YAML—the Python module supplies it.

## `myagent.py`

This is where the **StateGraph**, **prompt template**, and **your own tools** live. The platform may also pass **additional tools** (for example MCP): your factory should **combine** those with yours so the model can call everything listed for the deployment.

Patterns visible in the file:

- **Chat history**&mdash;included only if the prompt template expects a `chat_history` variable (see the sample template).
- **Graph factory**&mdash;receives the LLM, injected tools, and verbosity from the runner so one codebase works locally and on DataRobot.

[`register.py`](../../e2e-tests/dragent/langgraph/register.py) connects this module to NAT/DRAgent; copy its shape when you add a new agent package.

## Human in the loop

The sample `myagent.py` can include a **review** node that calls LangGraph’s **`interrupt()`** between other nodes (planner → **human review** → writer). That pattern needs a **checkpointer** and a stable **`thread_id`** across the interrupt and resume requests. See [hitl.md](hitl.md) for behavior, `langgraph_resume`, DRAgent’s shared `HITL_E2E_CHECKPOINTER` in `register.py`, and links to interrupt/resume tests.
