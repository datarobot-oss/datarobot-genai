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

# A2A Tasks, Artifacts, Files and Images

By default a DRAgent A2A server replies with a plain text `Message`. To return
**files, images, or structured data**, configure an *artifact builder*: one class
with one method.

## Why this is needed

NAT's A2A adapter is *Phase 1 / message-only*. It reads text from the inbound
request and discards file and data parts, flattens the workflow result to a string,
and returns a bare `Message`.

In the A2A specification, `Artifact` objects only exist on a `Task` — a `Message`
has no `artifacts` field. So artifacts are not merely unsupported through that
path; they are **structurally impossible**. `datarobot_genai.dragent.a2a_artifacts`
supplies the missing half: full inbound part parsing, and a task lifecycle that can
carry artifacts.

## Quick start

Implement a builder:

```python
# myapp/finance.py
from datarobot_genai.dragent.a2a_artifacts import (
    data_artifact,
    file_artifact,
    text_artifact,
)


class FinanceArtifacts:
    async def build_artifacts(self, inputs, response_text):
        return [
            text_artifact("summary", response_text),
            data_artifact("analysis", {"symbol": "AAPL", "change_pct": 1.2}),
            file_artifact("chart.png", render_chart(), "image/png"),
        ]
```

Reference it from `workflow.yaml`:

```yaml
general:
  front_end:
    _type: dragent_fastapi
    a2a:
      server:
        name: Finance Agent
        description: Answers questions about market data.
      artifact_builder: "myapp.finance.FinanceArtifacts"
      task_mode: auto
```

That is the whole integration. No custom front-end plugin, no subclassing, no
imports from private modules.

## Concepts

An **artifact** is a *named container* of parts — it is not itself binary data.
Binary payloads travel inside a file part. One artifact may hold several part kinds
at once.

| Part kind | Carries | Helper |
|---|---|---|
| `TextPart` | human-readable text | `text_artifact` |
| `DataPart` | structured JSON | `data_artifact` |
| `FilePart` / `FileWithBytes` | a file, inlined as base64 | `file_artifact` |
| `FilePart` / `FileWithUri` | a file, by reference | `file_uri_artifact` |
| several at once | summary + data + file together | `mixed_artifact` |

## Response shape: `task_mode`

Both `Task` and `Message` are valid results for `message/send`
(`SendMessageSuccessResponse.result: Task | Message`), so the response shape is
selectable.

| Mode | Behaviour | Use when |
|---|---|---|
| `auto` *(default)* | Run the workflow, then decide: no artifacts → `Message`; one or more → `Task` carrying them. No progress events, because the outcome decides the shape. | Most agents. Non-breaking for callers that expect a message. |
| `always` | Open the `Task` up front and emit `submitted` → `working` → `completed`. | Callers need progress on long-running work, or the agent always produces artifacts. |
| `never` | Always reply with a `Message`, even if the builder returns artifacts. | Explicit opt-out. |

There is a genuine trade-off between `auto` and `always`. Emitting `working`
requires committing to a `Task` *before* the workflow has run, so it cannot depend
on whether artifacts exist. `auto` therefore has no progress events; `always` has
them but is always task-shaped.

## Reading inbound files

`build_artifacts` receives an `A2ARequestInputs` carrying **everything** the caller
sent, not just text:

```python
class DescribeUpload:
    async def build_artifacts(self, inputs, response_text):
        for file in inputs.files:
            print(file.describe())        # "chart.png (image/png, 4096 bytes inline)"
            if file.is_inline:
                process(file.content)     # decoded bytes
            else:
                fetch(file.uri)           # FileWithUri reference

        return [data_artifact("received", {"count": len(inputs.files)})]
```

| Attribute | Description |
|---|---|
| `inputs.text` | Concatenated text of every `TextPart` |
| `inputs.files` | Every `FilePart`, decoded into `InboundFile` |
| `inputs.data` | The `data` payload of every `DataPart` |
| `inputs.has_attachments` | `True` when any file or data part was sent |

An `InboundFile` has exactly one of `content` (inline bytes) or `uri` (reference),
mirroring the spec's `FileWithBytes` / `FileWithUri` union. Base64 decoding is
tolerant of newline-wrapped and base64url payloads; an attachment that cannot be
decoded is logged and skipped rather than failing the whole request.

A **file-only request** ("here is an image, describe it") is legitimate under the
A2A spec and is supported — the workflow simply receives an empty query.

## Choosing a file representation

```python
# Inline: self-contained, but base64 inflates the payload by ~33%.
file_artifact("chart.png", png_bytes, "image/png")

# By reference: scales to large files, but the caller must be able to reach it.
file_uri_artifact("report.csv", "https://example.com/report.csv", "text/csv")
```

A URI inherits no A2A credentials. Use a pre-signed URL or a separately authorised
endpoint — otherwise the caller will be unable to fetch it.

## Multi-part artifacts

Most real deliverables are a summary, the machine-readable equivalent, and the
generated file, delivered as one named unit:

```python
mixed_artifact(
    "q3-report",
    text="Revenue up 4.1% quarter over quarter.",
    data={"revenue_change_pct": 4.1},
    files=[("q3.csv", csv_bytes, "text/csv")],
)
```

## Advertise your modalities

If your agent can return files, declare that on the agent card. The card describes
*capability*, not what any single response happened to contain — a text-only card
tells callers not to expect files, and some will not look for them.

```yaml
a2a:
  server:
    name: Finance Agent
    default_input_modes: ["text/plain", "image/png"]
    default_output_modes: ["text/plain", "application/json", "image/png", "text/csv"]
```

## Multi-turn conversations

Reuse **`contextId`**, not `taskId`. A task that has reached a terminal state
cannot be continued — the A2A request handler rejects a follow-up that references
one. `contextId` is what threads related tasks together.

## Cancellation

`tasks/cancel` publishes a terminal `canceled` state so the caller stops waiting.

This is **best-effort bookkeeping, not preemption**: work already in flight is not
interrupted, because NAT exposes no cancellation hook into a running session. Any
compute already started runs to completion and its result is discarded. If you need
true cancellation, subclass `TaskArtifactAgentExecutor`, thread a cancellation
token through `run_workflow`, and check it between steps.

## Advanced: customising workflow invocation

The builder hook covers most cases. To change *how* the workflow is invoked — for
example to keep a structured result rather than a flattened string — subclass the
executor directly:

```python
from datarobot_genai.dragent.a2a_artifacts import TaskArtifactAgentExecutor


class StructuredExecutor(TaskArtifactAgentExecutor):
    async def run_workflow(self, inputs, context):
        async with self.session_manager.session() as session:
            async with session.run(inputs.text) as runner:
                self.last_result = await runner.result(to_type=MyModel)
                return self.last_result.summary
```

`TaskArtifactAgentExecutor` is concrete and public, with a single base class — there
is no base-ordering requirement to get wrong.

## The other side: consuming artifacts

Everything above makes an agent *return* artifacts. The mirror problem is that a
*calling* agent can neither send files nor read artifacts back, because NAT
flattens A2A in both directions:

| Direction | NAT flattens at | Consequence |
|---|---|---|
| Sending | `A2ABaseClient.send_message(message_text: str)` | builds a text-only `Message` |
| Reading | `A2ABaseClient.extract_text_from_events()` | artifacts silently discarded |

Neither is a protocol limit — the raw events already carry every artifact.
`datarobot_genai.dragent.a2a_artifact_client` supplies the missing helpers.

### From an LLM: one config line

```yaml
function_groups:
  finance_agent:
    _type: authenticated_a2a_client
    registry:
      external_id: agent-finance
    auth_provider: okta_auth
    artifact_client: true
```

That registers two functions alongside NAT's text-only `call`:

| Function | Purpose |
|---|---|
| `send_with_attachments(message, attach_data=None, attach_uris=None)` | Sends text, structured data and file URIs; returns a rendered report of the Task and every artifact |
| `get_task_artifacts(task_id)` | Re-reads a known task's artifacts |

Naming the group in `tool_names` exposes both automatically.

!!! note "Why this is opt-in"
    Registering functions changes which tools an agent's LLM can select, so
    `artifact_client` defaults to `false`. Unset, tool selection is unchanged.

Files go by **URI** here rather than by content, because a model cannot emit raw
bytes. To send bytes, call `send_parts()` from code.

### From code: bytes, and structured access

```python
from datarobot_genai.dragent.a2a_artifact_client import (
    OutboundFile, iter_artifacts, save_task_files, summarize_task,
)

group = await builder.get_function_group("finance_agent")

events = await group.send_parts(
    text="analyse this",
    files=[OutboundFile("in.png", "image/png", content=png_bytes)],
    data={"threshold": 0.5},
)

for artifact in iter_artifacts(events):        # every Artifact, de-duplicated
    ...
paths = save_task_files(events, "/tmp/out")    # inline files written to disk
report = summarize_task(events)                # text rendering, for a tool result
```

`send_parts()` goes through the function group's own authenticated client, so the
Okta cross-application-access exchange applies exactly as it does to a text-only
call. Applications never need to reach into the client to borrow it.

### Rendering vs access

`summarize_task()` returns **text**, because a NAT function's return value becomes
an LLM tool result: previews are truncated and files are reported by name, type and
size rather than inlined. When code consumes the result, prefer `iter_artifacts()`
and `save_task_files()`.

`save_task_files()` writes inline files only — a `FileWithUri` carries none of the
call's authentication, so fetching it is the caller's problem — and flattens names
to their basename, so an artifact cannot write outside the output directory.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Response is `"kind": "message"` with no artifacts | No `artifact_builder` configured, or the builder returned `[]` | Set `a2a.artifact_builder`; check the builder's return value |
| `send_with_attachments` not offered to the LLM | `artifact_client` not set | Set `artifact_client: true` on the function group; confirm the group is named in `tool_names` |
| `A2A client not initialized` from `send_parts()` | Function group not entered | Resolve it via `builder.get_function_group(...)`, which enters it |
| `A2A client unavailable: agent card registry lookup failed` | Degraded mode after a registry miss | Fix `registry.external_id` / `deployment_id`; check `DATAROBOT_API_TOKEN` |
| `a2a.artifact_builder ... is not a class` / `Could not import` | Bad dotted path | Use `package.module.ClassName`; confirm it is importable from the agent's working directory |
| `... does not implement build_artifacts` | Class lacks the method | Implement `async def build_artifacts(self, inputs, response_text)` |
| Artifacts vanish; only text arrives | Caller is reading with a text-only client | Read `result.artifacts`; text-only callers concatenate `TextPart`s and drop the rest |
| Caller cannot fetch a URI file | The URI inherits no A2A auth | Use a pre-signed URL, or inline the bytes |
| Large or slow responses | Big files inlined as base64 | Switch to `file_uri_artifact` |
| `Task failed: ...` in a text message | The builder or workflow raised | Check the deployment logs; the traceback is logged |

## API reference

Full signatures and docstrings for every public name are generated from the source:

- **API → dragent → a2a_artifacts** (producing) — `ArtifactBuilder`,
  `OutboundArtifact`, `TaskArtifactAgentExecutor`, `A2ARequestInputs`,
  `InboundFile`, and the `*_artifact` helpers.
- **API → dragent → a2a_artifact_client** (consuming) — `OutboundFile`,
  `build_client_message`, `build_send_message_payload`, `iter_artifacts`,
  `summarize_task`, `save_task_files`.
