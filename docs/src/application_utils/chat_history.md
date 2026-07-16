# AG-UI Chat History

The chat-history layer turns an AG-UI agent's event stream into durable,
typed chat history on top of the [Memory Service ORM](memory_orm.md).  It gives
you three things:

- **Typed chat models** — `Chat` (a `DRSession`) and `Message` (a `DREvent`),
  with nested `ToolCall` / `Reasoning` models living inside a single message
  event body.  One Memory Service event per logical message.
- **A transparent storage wrapper** — `AGUIStorageAgent` forwards an inner
  agent's events verbatim (in real time) while a background task folds the same
  stream into persisted history: inbound user messages are stored, prior history
  is replayed into the run, and every text / tool-call / reasoning delta is
  captured.
- **Disconnect survival and cancellation** — `StreamPersistenceManager` runs the
  storage agent behind an unbounded queue so persistence completes even if the
  client stops reading, and exposes cancellation that marks in-flight records
  `interrupted`.

Everything is imported from a single package:

```python
from datarobot_genai.application_utils.chat_history import (
    AGUIAgent,
    AGUIStorageAgent,
    ChatRepository,
    ChatSessionRegistry,
    MessageRepository,
    RunHandle,
    StreamPersistenceManager,
)
```

The chat-history package depends on `persistence` **and** `ag_ui`; the
`persistence` sub-package never imports `ag_ui`, so the ORM stays
transport-agnostic.

## Quick-start

Wire the repositories, wrap your inner agent, and run it through the manager.
`StreamPersistenceManager.run` returns a `RunHandle` you iterate for the
client-facing stream, cancel, or await to completion.

```python
import asyncio
from uuid import uuid4

from ag_ui.core import RunAgentInput, UserMessage

from datarobot_genai.application_utils.chat_history import (
    AGUIStorageAgent,
    ChatRepository,
    ChatSessionRegistry,
    MessageRepository,
    StreamPersistenceManager,
)
from datarobot_genai.application_utils.persistence import DRMemoryServiceClient, DRMemorySpace


async def main(inner_agent) -> None:
    user_id = uuid4()

    async with DRMemoryServiceClient() as client:
        space = await DRMemorySpace.post(client, deduplication_key="my-chat-app-v1")

        # A registry shares the chat_uuid -> session_id cache across repos.
        registry = ChatSessionRegistry(space)
        chat_repo = ChatRepository(space, registry)
        message_repo = MessageRepository(space, registry)

        # The factory builds a fresh storage agent per run; the manager owns the
        # in-flight-run registry keyed by (thread_id, run_id).
        manager = StreamPersistenceManager(
            lambda: AGUIStorageAgent(
                "assistant", user_id, chat_repo, message_repo, inner_agent
            )
        )

        input = RunAgentInput(
            thread_id="thread-1",
            run_id="run-1",
            messages=[UserMessage(id="m1", role="user", content="Hello!")],
            tools=[],
            context=[],
            state={},
            forwarded_props={},
        )

        handle = await manager.run(input)

        # Stream events to the client. Stopping early (a disconnect) does NOT
        # abort persistence — the producer keeps draining behind the scenes.
        async for event in handle.events():
            ...  # forward `event` to your UI / SSE response

        # Wait for the background persistence to finish, then read history back.
        await handle.wait()
        history = await message_repo.get_chat_messages(
            (await chat_repo.get_chat_by_thread_id(user_id, "thread-1")).chat_uuid
        )
        for message in history:
            print(message.role, message.content, message.tool_calls, message.reasonings)


# `inner_agent` is any AGUIAgent: `run(input) -> AsyncGenerator[BaseEvent]`.
asyncio.run(main(inner_agent=...))
```

Cancel a run through the handle (or `manager.cancel(thread_id, run_id)`):

```python
handle = await manager.run(input)
...
handle.cancel()      # -> True if a live run was found, False otherwise
await handle.wait()  # in-flight records are finalised as `interrupted`
```

## Data model

| Model | Base | Stored as |
|---|---|---|
| `Chat` | `DRSession` | One session per thread. `thread_id` is a range key (indexed `//thread/{thread_id}/` lookup); `dedup_key = sha256("chat"\0user_uuid\0thread_id)` makes create idempotent. Metadata: `name`, `chat_uuid`, `user_uuid`. |
| `Message` | `DREvent` (`event_type="message"`) | One event per logical message. Body carries `v`, `role`, `status`, `in_progress`, and the nested `tool_calls: list[ToolCall]` / `reasonings: list[Reasoning]`. |
| `ToolCall` | `BaseModel` | Nested in a message body — `tool_call_id`, `name`, `arguments`, `content`, `status`. |
| `Reasoning` | `BaseModel` | Nested in a message body — `name`, `content`, `status`. |

The nested models round-trip through the Memory Service ORM's typed
serialization (see [Memory Service ORM](memory_orm.md)); empty `content` /
`arguments` are transparently encoded to a zero-width placeholder to satisfy the
service's `min_length=1`.

`Role` values: `developer`, `system`, `assistant`, `user`, `tool`, `reasoning`.
`MessageStatus` values: `active`, `complete`, `interrupted`, `errored`.

`ChatRepository` and `MessageRepository` are the CRUD facades (create/adopt a
chat idempotently, append messages, mutate nested tool-calls / reasonings,
read history sorted by sequence id).  `ChatSessionRegistry` resolves
`chat_uuid → session_id` via a bounded in-process cache, then an indexed
`chat:<uuid>` [`EntityLocator`](#secondary-index-and-consistency) point lookup —
never a full-space scan.

## Secondary index and consistency

The Memory Service has no cross-session event query, so resolving an entity by
its application UUID — `chat_uuid → session`, `message_uuid → chat`, or a
tool-call / reasoning `uuid → parent message` — could otherwise only be done by
scanning every session in the space.  Instead the layer keeps a dedup-keyed
secondary index: one `EntityLocator` (itself a `DRSession`) per locatable entity,
looked up by the exact key `"<kind>:<uuid>"` (`kind` is one of `chat`, `message`,
`tool_call`, `reasoning`).  Every cold lookup is a single indexed point `get` —
there are **no full-space scans**.

Locators live in the **same** Memory Space as the chats; their `//loc/`
description prefix keeps them out of every `Chat.list` result (and vice-versa),
so no second space is needed.

### Best-effort writes and the consistency trade-off

Each create writes the entity first (the event or session — the **source of
truth**), then writes its locator best-effort.  A locator write that fails is
logged and swallowed; it never fails the operation.  The deliberate consequences:

- The worker that created the entity is unaffected — its in-process cache is warm.
- If a locator write is **lost**, a *different* replica (or a lookup after a
  restart) that addresses the entity **by uuid** — `get_message`,
  `update_message`, a tool-call / reasoning update — may report "not found" for
  an entity that actually exists.
- The data is **not** lost: `get_chat_messages` lists the chat's events directly
  and still returns the message.

This backend therefore offers **no strong cross-replica read-after-write
consistency for uuid-addressed lookups**.  A caller that cannot tolerate that
needs a transactional backend, not this one.

### Orphans

`delete_chat` soft-deletes the session (and its events) in one shot; there is no
per-message delete, and it deliberately does **not** walk and delete the chat's
locators.  Those locators are left as **orphans** — cheap, bounded by the same
soft-delete TTL as the sessions they point at, and harmless: a stale locator
resolves to a dead session and is treated as not-found, and because UUIDs never
collide it can never resurrect or mis-route a new entity.

## Run-outcome status semantics

Every message / tool-call / reasoning record carries a `status`.  How a run ends
determines the terminal status of the records that were still `in_progress`:

| How the run ends | Terminal status | Mechanism |
|---|---|---|
| **Normal completion** | `complete` | The inner agent emits `RunFinishedEvent`; `handle_run_lifecycle` flips the active message (and its tool-calls / reasonings) to `in_progress=False`, `status=complete`. |
| **Explicit cancel** | `interrupted` | `RunHandle.cancel()` / `StreamPersistenceManager.cancel()` cancels the producer task; the `CancelledError` propagates into `AGUIStorageAgent.run`, whose `finally` calls `_finalize_interrupted` to flip every still-`in_progress` record to `status=interrupted`. |
| **Inner-agent raw crash** | `errored` | The inner agent raises (rather than emitting a terminal `RunErrorEvent`).  `AGUIStorageAgent.run` records the exception, its `finally` calls `_finalize_errored` to flip still-active records to `status=errored`, then re-raises — the manager's producer catches it and synthesizes a client-facing `RunErrorEvent(code=INTERNAL_ERROR)` so the client never hangs. |
| **Client disconnect** | `complete` | The client stops reading `RunHandle.events()`, but the producer drains the inner agent into its **unbounded** queue regardless, so the run finishes normally and persists as `complete`.  `await handle.wait()` blocks until that drain finishes. |

> An in-band `RunErrorEvent` emitted *by the inner agent* is a normal terminal
> event: `handle_run_lifecycle` records those records as `errored` directly.
> `_finalize_errored` is specifically the safety net for a **raw exception** that
> escaped the inner agent without a terminal event.

## Extensibility

`AGUIStorageAgent` is an *open* state machine: every seam a consumer might want
to customise is an overridable method, and it depends only on repository
Protocols.  There are four extension points.

### 1. Event-dispatch registry

`event_handlers()` maps each AG-UI event type to a handler-method name.  Extend
it (walking the MRO, so subclasses of known events resolve automatically) to
persist brand-new or custom event types; unrecognised events fall through to
`handle_unknown_event` (a no-op by default).

```python
class MyStorageAgent(AGUIStorageAgent):
    @classmethod
    def event_handlers(cls):
        return {**super().event_handlers(), MyCustomEvent: "handle_my_custom_event"}

    async def handle_my_custom_event(self, state, chat, event):
        ...  # persist the custom event
```

### 2. Category handlers

`handle_text_message`, `handle_tool_call`, `handle_reasoning`,
`handle_run_lifecycle` and `handle_step` own each event family.  Override one to
change how a whole category is stored — e.g. to extract structured content
instead of appending raw text deltas.

```python
class StructuredTextAgent(AGUIStorageAgent):
    async def handle_text_message(self, state, chat, event):
        # e.g. parse `event` into structured fields before buffering / flushing
        await super().handle_text_message(state, chat, event)
```

### 3. Message-build hooks

`build_message_create`, `build_tool_call_create`, `build_reasoning_create` and
`message_update_fields` construct the DTOs the repository persists.  Override
them to populate extra fields declared on your `Message` / `ToolCall` /
`Reasoning` subclasses.

```python
class TaggedStorageAgent(AGUIStorageAgent):
    def build_message_create(self, state, chat, agui_id, role):
        base = super().build_message_create(state, chat, agui_id, role)
        return MyMessageCreate(**base.model_dump(), team="research")

    def message_update_fields(self, state, event):
        return {"finished_at": state.current_event_timestamp}
```

### 4. Repository coupling (Protocols)

The agent references only `ChatRepositoryLike` / `MessageRepositoryLike`
(`runtime_checkable` `typing.Protocol`s), so any conforming backend — a SQL
store, an in-memory fake for tests — is a drop-in replacement with no imports of
the concrete classes.

```python
class InMemoryChatRepo:  # structurally satisfies ChatRepositoryLike
    async def get_chat_by_thread_id(self, user_uuid, thread_id): ...
    async def create_chat(self, chat_data): ...
    # ...remaining protocol methods...

agent = AGUIStorageAgent("assistant", user_id, InMemoryChatRepo(), my_message_repo, inner)
```

## Running acceptance tests

The chat-history acceptance suite drives a scripted inner agent against a live
Memory Service.  It is skipped by default and requires credentials:

```bash
export DATAROBOT_ENDPOINT="https://app.datarobot.com/api/v2"
export DATAROBOT_API_TOKEN="<your-token>"

uv run pytest tests/application_utils/chat_history/acceptance -m integration -vv
```
