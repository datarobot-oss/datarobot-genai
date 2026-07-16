# application-utils — Memory Service Light ORM

`datarobot-genai[application-utils]` ships a lightweight async ORM over the
DataRobot Agentic Memory Service.  Think of it as a typed, Pydantic v2-style
document store that persists sessions (documents) and events (log entries)
through a bearer-auth REST API, using your existing `DATAROBOT_ENDPOINT` and
`DATAROBOT_API_TOKEN` credentials.

## Install

```bash
pip install "datarobot-genai[application-utils]"
```

## Quick-start

```python
import asyncio
from typing import Annotated

from datarobot_genai.application_utils.persistence import (
    DRConcurrencyField,
    DRDeduplicationKey,
    DREvent,
    DRMemorySpace,
    DRRangeKey,
    DRSession,
    DRMemoryServiceClient,
    SYSTEM_PARTICIPANT,
)


# ── 1. Define your domain models ─────────────────────────────────────────────

class ChatSession(DRSession):
    """A chat session stored in the Memory Service."""

    __description_prefix__ = "chat"          # stable prefix; part of every query

    tenant:  Annotated[str, DRRangeKey]       # description segment 1 (range queries)
    topic:   Annotated[str, DRRangeKey]       # description segment 2 (range queries)
    chat_id: Annotated[str, DRDeduplicationKey]  # point-lookup / idempotent create
    rev:     Annotated[int, DRConcurrencyField]  # mirrors server version integer
    title:   str = ""                         # plain metadata (payload only)


class ChatMessage(DREvent, session=ChatSession):
    """A single message in a chat session."""

    __event_type__ = "message"

    score: float = 0.0   # extra body field; round-trips through the wire


# ── 2. Use the ORM ────────────────────────────────────────────────────────────

async def demo() -> None:
    async with DRMemoryServiceClient() as client:
        # Create or adopt an existing space (idempotent via deduplication_key)
        space = await DRMemorySpace.post(
            client,
            description="My agent memory",
            deduplication_key="my-agent-space-v1",
        )

        # Create (or adopt) a session
        session = await ChatSession.post(
            space,
            tenant="acme",
            topic="billing",
            chat_id="billing-chat-001",
            title="Billing enquiry",
            rev=1,
        )

        # Append events
        msg = await ChatMessage.post(
            session,
            content="Hello, I need help with my invoice.",
            emitter_type="user",
            emitter_id="aabbccddeeff001122334455",  # 24-hex ObjectId
            score=0.9,
        )
        print(msg.sequence_id, msg.created_at)

        # List recent messages
        recent = await ChatMessage.last(session, n=10)

        # Fetch by range-key prefix (all billing sessions for "acme")
        billing_sessions = await ChatSession.list(space, tenant="acme", topic="billing")

        # Update session metadata (optimistic-concurrency guard via If-Match)
        await session.patch(title="Resolved billing enquiry")

        # Patch an event (guarded by its createdAt token)
        await msg.patch(score=0.5)

asyncio.run(demo())
```

## Environment variables

| Variable | Description |
|---|---|
| `DATAROBOT_ENDPOINT` | Full DataRobot API base URL, e.g. `https://app.datarobot.com/api/v2`. |
| `DATAROBOT_API_TOKEN` | DataRobot API bearer token. |

Both are resolved automatically; pass them as constructor arguments to override.

## Core concepts

### Memory space (`DRMemorySpace`)

A namespace that owns sessions and events.  Create one per agent deployment or
application context.  Idempotent via `deduplication_key`.

```python
space = await DRMemorySpace.post(client, deduplication_key="my-app-v1")
space2 = await DRMemorySpace.get(client, space.id)
spaces = await DRMemorySpace.list(client, deduplication_key="my-app-v1")
await space.patch(description="Updated description")
await space.delete()
```

### Sessions (`DRSession`)

Documents stored in a memory space.  Subclass `DRSession` and annotate fields
with ORM markers:

| Marker | Wire field | Purpose |
|---|---|---|
| `Annotated[T, DRDeduplicationKey]` | `deduplicationKey` | Unique key; idempotent create. |
| `Annotated[T, DRRangeKey]` | `description` segment | Range / prefix queries. |
| `Annotated[T, DRConcurrencyField]` | `version` mirror | User-visible version counter. |
| *(plain field)* | `metadata` | Arbitrary payload; not queryable. |

Declare range-key fields **in query order** — a list query must specify a
*contiguous leading prefix* (see §Range-key encoding below).

#### Lifecycle strategies (TTL)

By default, every `DRSession` subclass sends a single `soft_delete` lifecycle
strategy on creation, triggered by a **2-year TTL** (`DEFAULT_SESSION_TTL_SECONDS`,
`63072000` seconds) — the Memory Service's own maximum for a TTL trigger. Sessions
therefore auto-clean unless you override this.

Override `__lifecycle_strategies__` to use a shorter TTL (or a different strategy):

```python
class ChatSession(DRSession):
    __description_prefix__ = "chat"
    __lifecycle_strategies__ = [
        {"type": "soft_delete", "trigger": {"ttl": 30 * 86400}},  # 30 days
    ]
    ...
```

Set it to an empty list to send no lifecycle strategies at all:

```python
class ChatSession(DRSession):
    __lifecycle_strategies__ = []
```

Lifecycle strategies are sent **on create only**; `session.patch(...)` never
touches them. Up to 5 strategy objects are allowed per session.

### Events (`DREvent`)

An append-only log under a session.  Bind to a session type with
`class MyEvent(DREvent, session=MySession)`.  All plain declared fields map to
the `body` dict.

```python
class ChatMessage(DREvent, session=ChatSession):
    __event_type__ = "message"   # "message" | "tool_output" | "status"
    score: float = 0.0
```

## Range-key encoding

Range keys are encoded in the session `description` field using a hierarchical
path scheme:

```
description = "//" + esc(prefix) + "/" + esc(k1) + "/" + esc(k2) + "/"
```

`esc(v)` percent-encodes `%` (→ `%25`) then `/` (→ `%2F`), so values can
contain arbitrary text including slashes.  The leading `//` and trailing `/`
after every segment create an **anchored prefix** — a substring-match on the
service side is equivalent to a hierarchy prefix query.

**Example** — two sessions stored under `chat/acme`:

```
//chat/acme/billing/    ← tenant=acme, topic=billing
//chat/acme/support/    ← tenant=acme, topic=support
```

Query `list(tenant="acme")` sends `description=//chat/acme/` which matches
**both**.  Query `list(tenant="acme", topic="billing")` sends
`description=//chat/acme/billing/` which matches **only the first**.

> ⚠️ **Case-insensitive caveat** — the service performs a case-insensitive
> substring match, so `Acme` and `acme` are treated as the same tenant.
> Values that differ only in case will collide.

## Optimistic concurrency

### Sessions

Session PATCH sends an `If-Match: <version>` header.  If the server's version
has advanced since you last fetched the session, the service returns HTTP 409
and the ORM raises `DRMemoryVersionConflictError`.  Resolve by re-fetching
(`DRSession.get`) and retrying.

```python
try:
    await session.patch(title="New title")
except DRMemoryVersionConflictError:
    session = await ChatSession.get(space, id=session.id)
    await session.patch(title="New title")
```

### Events

Event PATCH uses `createdAt` as a query-string concurrency token instead of
a header.  A stale token yields HTTP 422 and `DRMemoryVersionConflictError`.

```python
try:
    await msg.patch(content="Corrected text")
except DRMemoryVersionConflictError:
    # Re-list to obtain fresh tokens
    events = await ChatMessage.list(session)
    msg = next(e for e in events if e.sequence_id == msg.sequence_id)
    await msg.patch(content="Corrected text")
```

## Batch operations

```python
# Atomic batch create (up to 200 events)
msgs = await ChatMessage.post_batch(
    session,
    events=[
        {"content": "First", "emitter_type": "agent"},
        {"content": "Second", "emitter_type": "agent", "score": 0.5},
    ],
)

# Atomic batch patch (up to 200 events)
await ChatMessage.patch_batch(
    session,
    updates=[
        (msgs[0], {"score": 0.9}),
        (msgs[1], {"content": "Updated second"}),
    ],
)
```

## Emitter participant check

When `emitter_type="user"`, the emitter's ObjectId must be in the session's
`participants` list.  The ORM raises `DRMemoryBadRequestError` early (before the
HTTP call) if the emitter is not a participant.

The system sentinel `SYSTEM_PARTICIPANT = "000000000000000000000000"` is a
client-side convention for agent-owned sessions.  It is a valid ObjectId
accepted by the service.

## Error types

| Exception | HTTP status | Cause |
|---|---|---|
| `DRMemoryBadRequestError` | 400 | Invalid request (emitter not a participant, etc.) |
| `DRMemoryNotFoundError` | 404 | Resource does not exist |
| `DRMemoryConflictError` | 409 | Deduplication conflict on create (ORM auto-adopts) |
| `DRMemoryVersionConflictError` | 409 / 422 | Stale If-Match or createdAt token |
| `DRMemoryValidationError` | 422 | Schema validation error |
| `DRMemoryServiceError` | other 4xx/5xx | Unexpected error |

## Public API

```python
from datarobot_genai.application_utils.persistence import (
    DRMemorySpace,
    DRSession,
    DREvent,
    DRDeduplicationKey,
    DRRangeKey,
    DRConcurrencyField,
    DRMemoryServiceClient,
    SYSTEM_PARTICIPANT,
    DEFAULT_SESSION_TTL_SECONDS,
    DRMemoryServiceError,
    DRMemoryNotFoundError,
    DRMemoryBadRequestError,
    DRMemoryValidationError,
    DRMemoryConflictError,
    DRMemoryVersionConflictError,
)
```

## Running integration tests

Integration tests are skipped by default.  To run them against a live endpoint:

```bash
export DATAROBOT_ENDPOINT="https://app.datarobot.com/api/v2"
export DATAROBOT_API_TOKEN="<your-token>"
export DR_MEMORY_LIVE_INTEGRATION="1"

uv run pytest tests/application_utils/persistence/integration -vv
```
