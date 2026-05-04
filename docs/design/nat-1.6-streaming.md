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

# NAT 1.6 streaming architecture in DRAgent

## Background

NAT 1.4.1 did not expose a streaming path for `tool_calling_agent`. All intermediate output was delivered through NAT's `StepAdaptor.process()` callback, which receives fully-formed `IntermediateStep` objects and converts them to AG-UI events. This is the "step adaptor" path.

In [NVIDIA/NeMo-Agent-Toolkit#1595](https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1595), NAT added token-by-token streaming to `tool_calling_agent` via a `stream_fn` that yields `ChatResponseChunk` objects (OpenAI-compatible streaming deltas). In [NVIDIA/NeMo-Agent-Toolkit#1717](https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1717), NAT added incremental tool call chunk streaming to that same path. These chunks contain partial text content and incremental tool call arguments as they arrive from the LLM. NAT's built-in frontend renders these directly, but DRAgent needs AG-UI events, so we have to convert them.

## Two delivery paths now coexist

After NAT 1.6, DRAgent has two parallel paths for delivering events to the AG-UI client:

1. **Step adaptor path** (`StepAdaptor.process()`): NAT calls this with complete `IntermediateStep` objects for reasoning steps, tool starts/ends, custom events, and run lifecycle events. This is still the primary path for most event types.

2. **Stream conversion path** (`stream_converter.py`): The `per_user_tool_calling_agent` wraps the NAT `stream_fn` and pipes `ChatResponseChunk` objects through `convert_chunks_to_agui_events()`, which converts them to `TextMessage*` and `ToolCall*` AG-UI events. This path handles real-time text streaming and incremental tool call argument delivery.

Both paths emit `DRAgentEventResponse` objects that the SSE transport sends to the client. The step adaptor path handles structured events (reasoning, run lifecycle), while the stream conversion path handles low-latency token-by-token delivery.

## How stream conversion works

`convert_chunks_to_agui_events()` (in `stream_converter.py`) is a standalone async generator that consumes `ChatResponseChunk` objects and yields `DRAgentEventResponse` batches. It is fully self-contained with no dependency on the step adaptor. It tracks state across chunks:

- **Text streaming**: The first chunk with text content triggers `TextMessageStartEvent`. Subsequent chunks produce `TextMessageContentEvent`. On stream completion, `TextMessageEndEvent` is emitted.

- **Tool call streaming**: OpenAI-style tool call deltas arrive incrementally. The first chunk for a tool call carries an `id` and the function `name`, producing `ToolCallStartEvent`. Follow-up chunks carry only an `index` (no `id`) and argument fragments, producing `ToolCallArgsEvent`. A `tool_index_map` maintains the index-to-id mapping so follow-up chunks can be correlated. On stream completion, `ToolCallEndEvent` is emitted for all active tool calls.

- **Parallel tool calls**: Multiple tool calls can arrive interleaved in a single chunk (different `index` values). Each is tracked independently.

## Error handling

Upstream exceptions (e.g., LLM provider errors, network failures) are caught by `convert_chunks_to_agui_events()` and surfaced to the AG-UI client as `RunErrorEvent(code="STREAM_ERROR")`. The exception is **not** propagated to the caller. This is intentional: NAT's streaming infrastructure does not expect exceptions from `stream_fn` consumers, and propagating would cause unhandled error responses or broken SSE connections.

If the client disconnects mid-stream (`GeneratorExit`), end events are skipped (the client is gone) and the generator exits cleanly.

**Note:** The step adaptor path (`StepAdaptor.process()`) handles errors differently. A single step failing to process (bad payload, serialization error) returns `None` and logs the exception. It does **not** emit `RunErrorEvent`, because a step-level error does not mean the entire run has failed&mdash;the agent continues and subsequent steps follow.

## Where the wrapping happens

`per_user_tool_calling_agent.py` is the glue. It:

1. Calls `tool_calling_agent_workflow.__wrapped__()` to get the original `FunctionInfo`
2. If `stream_fn` is present, wraps it: the wrapper pipes chunks through `convert_chunks_to_agui_events()` from `stream_converter.py`
3. Yields a new `FunctionInfo` with the wrapped `stream_fn` and the original `single_fn`
4. If `stream_fn` is `None`, yields the original `FunctionInfo` unchanged

## Other NAT 1.6 changes affecting the runtime

- **UserManager monkey-patch**: In [NVIDIA/NeMo-Agent-Toolkit#1775](https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1775), NAT added centralized user identity management via `UserManager.extract_user_from_connection()`, but it only supports standard auth (JWT, cookies, API key). DRAgent monkey-patches this method to also check `X-DataRobot-Authorization-Context`, falling back to the original implementation. Applied once at import time with an idempotency guard. This also made our previous `set_metadata_from_http_request` override ineffective (NAT overwrites `user_id` after the call), so that override was removed.

- **Health routes**: NAT 1.6 no longer calls `self.add_health_route(app)` during setup. DRAgent registers health endpoints (`/`, `/ping`, `/health`) explicitly in `build_app()`.

- **verify_ssl stripping**: In [NVIDIA/NeMo-Agent-Toolkit#1640](https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1640), NAT added `verify_ssl` to LLM config objects. CrewAI forwards all config keys to litellm, which rejects unknown keys. We strip `verify_ssl` at both entry points (`_crewai_model_factory` and `litellm_crewai_internal`).

- **Import path changes**: NAT 1.6 moved `nat.agent.tool_calling_agent` to `nat.plugins.langchain.agent.tool_calling_agent`. `TokenUsageBaseModel` moved from `nat.profiler.callbacks` to `nat.data_models.token_usage` (see [NVIDIA/NeMo-Agent-Toolkit#1748](https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1748)).

- **CrewAI callback patch removed**: In [NVIDIA/NeMo-Agent-Toolkit#1803](https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1803), NAT fixed the `CrewAIProfilerHandler._llm_call_monkey_patch` for crewai >= 1.1.0 (see [NVIDIA/NeMo-Agent-Toolkit#1802](https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues/1802)), so our compatibility patch was removed.

- **User identity via JWT**: In [NVIDIA/NeMo-Agent-Toolkit#1584](https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1584), NAT added JWT/cookie-based user ID resolution, which [#1775](https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1775) later centralized into `UserManager`. DRAgent's auth context header is not part of NAT's supported auth methods, hence the monkey-patch.
