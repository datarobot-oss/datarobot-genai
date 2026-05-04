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

# CrewAI sample: what you configure

Aligned with [`e2e-tests/dragent/crewai/`](../../e2e-tests/dragent/crewai/).

## `workflow.yaml`

| Piece | What you see |
|---|---|
| **`workflow._type: crewai_agent`** | DRAgent runs the CrewAI integration. |
| **`llms`** + **`workflow.llm_name`** | Same naming pattern as other samples. |

## `myagent.py`

You define **agents**, **tasks**, the **crew**, and the **kickoff input mapping** (placeholders like `{topic}` / `{chat_history}` in task text must match the keys you return). The sample shows history opt-in and optional tools.

[`register.py`](../../e2e-tests/dragent/crewai/register.py) registers the package with DRAgent.
