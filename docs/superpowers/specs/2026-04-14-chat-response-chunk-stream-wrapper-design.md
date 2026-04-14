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

# ChatResponseChunk Stream Wrapper

## Problem

NAT 1.6's `tool_calling_agent` added a `_stream_fn` that yields `ChatResponseChunk` objects. The DRAGent frontend expects `DRAgentEventResponse` with valid AG-UI event sequences (START before CONTENT, END after last CONTENT). A stateless per-chunk type converter cannot track this lifecycle.

## Solution

Wrap the `stream_fn` at the `per_user_tool_calling_agent` registration level. The wrapper consumes `ChatResponseChunk` objects, tracks AG-UI lifecycle state per-request, and yields `DRAgentEventResponse` with proper event sequencing.

## Location

`src/datarobot_genai/dragent/plugins/per_user_tool_calling_agent.py`

## Design

1. NAT's `tool_calling_agent_workflow` yields a `FunctionInfo` with `single_fn` and `stream_fn`
2. We intercept the `FunctionInfo`, wrap its `stream_fn` in an async generator that converts `ChatResponseChunk` to `DRAgentEventResponse`
3. The wrapper creates fresh state per-request (no global/module state)
4. We create a new `FunctionInfo` with the original `single_fn` but our wrapped `stream_fn`
5. Output type already matches -- no type converter involved

## Wrapper State (per-request)

- `active_message_id: str | None` -- tracks whether TEXT_MESSAGE_START was sent
- `active_tool_calls: set[str]` -- tracks which TOOL_CALL_STARTs were sent

## Event Mapping

| ChatResponseChunk | AG-UI Events |
|---|---|
| First text chunk | TEXT_MESSAGE_START + TEXT_MESSAGE_CONTENT |
| Subsequent text chunk | TEXT_MESSAGE_CONTENT |
| Tool call with new id | TOOL_CALL_START + TOOL_CALL_ARGS |
| Tool call with known id | TOOL_CALL_ARGS |
| Stream end | TEXT_MESSAGE_END (if active), TOOL_CALL_END (for each active) |

## What Doesn't Change

- converters.py
- register.py
- step_adaptor.py

## Success Criteria

- NAT e2e tests pass: `test_generate_streaming`, `test_mcp_tool_is_called`, `test_generate_objectid_tool_is_called`
- AG-UI sequence validation passes
- No global/module state
