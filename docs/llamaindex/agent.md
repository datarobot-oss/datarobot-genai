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

# LlamaIndex sample: what you configure

Aligned with [`e2e-tests/dragent/llamaindex/`](../../e2e-tests/dragent/llamaindex/).

## `workflow.yaml`

| Piece | What you see |
|---|---|
| **`llms:`** + **`datarobot-llm-component`** | Named LLM for the app; see [LLM configuration](../llm.md). |
| **`workflow._type: llamaindex_agent`** | DRAgent uses the LlamaIndex integration. |
| **`workflow.llm_name`** | Must match **`llms`**. |

## `myagent.py`

You define the **AgentWorkflow**, **agents** (e.g. planner / writer), and how the **final answer** is read from streamed events—the sample shows one way to extract text. Placeholders like **`{chat_history}`** or **`{memory}`** in the user message string are filled by the integration when you opt in (see the example’s string patterns).

[`register.py`](../../e2e-tests/dragent/llamaindex/register.py) wires the package for DRAgent.
