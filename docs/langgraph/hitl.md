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

# Human in the loop (LangGraph + DRAgent)

This page describes how **interrupt / resume** works when you use LangGraph inside `datarobot_genai` and the e2e DRAgent sample.

## Why you need a checkpointer

LangGraph only **remembers** a paused run if the compiled graph was built with a [`Checkpointer`](https://langchain-ai.github.io/langgraph/reference/checkpoints/). The `LangGraphAgent` constructor accepts **`checkpointer=...`** and passes it to `StateGraph.compile(...)`.

If `checkpointer` is `None`, the agent cannot resume from an interrupt and will not turn a follow-up user message into `Command(resume=...)` (see `_command_for_pending_interrupt` in [`langgraph/agent.py`](../../src/datarobot_genai/langgraph/agent.py)).

## What clients see in the event stream

When the graph reports an `__interrupt__` update, streaming emits a **`CUSTOM`** event named **`on_interrupt`**, then a **`RUN_FINISHED`** with `result["langgraph"]["interrupted"]` so UIs can show approval UI before the next call.

## Compile-time breakpoints (optional)

You can also pass **`interrupt_before`** / **`interrupt_after`** when constructing the agent; they are forwarded to `StateGraph.compile(...)`, for example to pause *before* a node without custom `interrupt()` code in that node.

## DRAgent / NAT: passing the checkpointer (e2e register)

The minimal [`register.py`](../../e2e-tests/dragent/langgraph/register.py) builds `MyAgent(..., checkpointer=HITL_E2E_CHECKPOINTER)`.

A **new** `InMemorySaver()` on every request would **drop** checkpoint state, so the interrupt response and the resume request would not see the same graph state. The sample uses a **module-level shared** `InMemorySaver` for e2e only. Real deployments should use a **durable** checkpointer appropriate to your environment.

## Further reading and tests

| Location | What it shows |
|----------|----------------|
| [`e2e-tests/dragent_tests/test_interrupt_resume.py`](../../e2e-tests/dragent_tests/test_interrupt_resume.py) | HTTP/SSE: interrupt stream, then resume with a plain user message. |
| [`datarobot_genai.langgraph.agent`](../../src/datarobot_genai/langgraph/agent.py) | `LANGGRAPH_RESUME_STATE_KEY`, input resolution, and AG-UI events. |

Env reference for the LLM: [LLM configuration (shared)](../llm.md).
