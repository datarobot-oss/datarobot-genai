# Context

We now have a Session/Event ORM in src/datarobot_genai/application_utils/persistence. The session/event API is already used by recipe-datarobot-agent-application repo,
locally ~/workspace/recipe-datarobot-agent-application for storing per-user
chat history, but it does not use this package's ORM.

The following subsection is a report on how that agent-application uses session and event for historh.

## Agent Application Chat History

There are actually two independent uses of DataRobot's Memory Service in this codebase — don't conflate them (only the first is relevant):

┌──────────────┬─────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────┐
│              │            fastapi_server/app/memory/ (this map)            │ agent/ — dr_mem0_memory (docs/agent/agent-memory.md) │
├──────────────┼─────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ Purpose      │ Persist raw chat history (threads/messages) shown in the UI │ Long-term fact recall across conversations           │
├──────────────┼─────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ Config       │ APPLICATION_MEMORY_SPACE_ID / USE_APPLICATION_MEMORY_SPACE  │ AGENT_MEMORY_SPACE_ID                                │
├──────────────┼─────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ Memory Space │ Separate space, owned by the FastAPI server                 │ Separate space, owned by the agent's NAT workflow    │
└──────────────┴─────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────┘

Everything below is the first one — the chat-storage layer.

1. SDK primitives (datarobot.models.memory)

MemorySpace                              # container: memory/{space_id}/
 └── Session                             # memory/{space_id}/sessions/{id}/
      ├── participants: list[str]        # 1 participant in this deployment (max_length=1)
      ├── description: str | None        # indexed, used as a lookup key
      ├── metadata: dict | None          # app-defined JSON
      ├── deduplication_key: str | None  # idempotent create
      ├── version: int                   # optimistic concurrency (If-Match)
      └── Event[]                        # memory/{space_id}/sessions/{id}/events/{seq}/
           ├── body: dict                # app-defined JSON payload
           ├── event_type: str
           ├── emitter: {type, id}       # "user" | "agent"
           ├── sequence_id: int          # ordinal, used to patch in place
           └── created_at

Session.post_event / .events() / .update_event(sequence_id, body=...) are the three calls the whole chat layer is built on.

2. Pluggable repository layer

fastapi_server/app/deps.py picks the backend at startup — the same Chat/Message Pydantic/SQLModel types are served by either implementation:

                     USE_APPLICATION_MEMORY_SPACE?
                            │
              ┌─────────────┴─────────────┐
              no                          yes
              │                            │
      ChatRepository/               MemoryChatRepository/
      MessageRepository             MemoryMessageRepository
      (SQLModel: chat/message         (datarobot.models.memory:
       tables, SQLite/Postgres)        Session/Event, one MemorySpace)

repo_types.py defines ChatRepositoryLike = ChatRepository | MemoryChatRepository (same for messages) so callers (app/ag_ui/storage.py, app/api/v1/chat.py) are backend-agnostic.

3. Entity mapping — the core of it

┌────────────────────────────────────┬───────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│            App concept             │        Memory-service concept         │                                                   Notes                                                    │
├────────────────────────────────────┼───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Chat (a "chat thread", AG-UI       │ Session                               │ 1 session per chat. session.participants = [memory_space_participant_id(user_uuid)] (exactly one — the     │
│ thread_id)                         │                                       │ user; the agent is not a session participant).                                                             │
├────────────────────────────────────┼───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Message (+ nested tool_calls[],    │ Event                                 │ 1 event per message — tool calls and reasoning steps are not separate events; they're nested JSON inside   │
│ reasonings[])                      │                                       │ the same event body, keyed by msg.uuid.                                                                    │
├────────────────────────────────────┼───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Chat fields (name, thread_id,      │ Session.metadata (dict)               │ Read/written via session_metadata() (camelCase→snake_case normalization since the REST API can echo        │
│ chat_uuid, user_uuid)              │                                       │ camelCase).                                                                                                │
├────────────────────────────────────┼───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Chat lookup key                    │ Session.description                   │ f"/thread/{thread_id}" at create time, f"/chat/{chat_uuid}" used elsewhere — an indexed field so           │
│                                    │                                       │ Session.list(..., description=...) is a fast point lookup instead of a full scan.                          │
├────────────────────────────────────┼───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Chat idempotency                   │ Session.deduplication_key             │ SHA-256 of ("chat", user_uuid, thread_id) — a retried "create chat" call adopts the existing session       │
│                                    │                                       │ (MemorySessionDeduplicationError.existing_session_id) instead of duplicating.                              │
├────────────────────────────────────┼───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Message content/role/step/etc.     │ Event.body (JSON, versioned {"v": 1,  │ message_to_payload() / payload_to_message() in repos.py are the (de)serializers.                           │
│                                    │ ...})                                 │                                                                                                            │
├────────────────────────────────────┼───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Message emitter                    │ Event.emitter                         │ {"type": "user", "id": <participant_id>} for user messages; {"type": "agent"} for assistant/tool/reasoning │
│                                    │                                       │  (since the agent isn't a real session participant).                                                       │
├────────────────────────────────────┼───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Message ordering / "last message"  │ Event.sequence_id                     │ Events are listed and sorted by sequence_id; "get last message" = highest sequence_id whose body parses as │
│                                    │                                       │  a message.                                                                                                │
├────────────────────────────────────┼───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Streaming update-in-place          │ Session.update_event(sequence_id,     │ A message event is created once (post_event, in_progress=True) then patched repeatedly as content streams  │
│                                    │ body=...)                             │ in — not appended as new events.                                                                           │
└────────────────────────────────────┴───────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

4. Streaming → storage state machine (app/ag_ui/storage.py)

AGUIAgentWithStorage consumes the AG-UI event stream (TextMessageStart/Content/End, ToolCallStart/Args/End, Thinking*) and folds it onto one Event per logical message:

AG-UI event stream                  Memory-service call
───────────────────                 ────────────────────
TextMessageStartEvent        ──►    session.post_event(body=message_to_payload(msg), emitter, "message")
TextMessageContentEvent(Δ)   ──►    (buffered in memory, flushed periodically)
  …buffer full / message end ──►    session.update_event(seq, body=message_to_payload(msg))
ToolCallStart/Args/End        ──►    same event body updated: msg.tool_calls[i] patched, re-serialized
ThinkingStart/Content/End     ──►    same event body updated: msg.reasonings[i] patched, re-serialized
RunFinished / RunError        ──►    final update_event(...) with in_progress=False (+ error)

So a single chat turn with reasoning + a tool call still produces one Event, whose body JSON grows/mutates over several update_event calls keyed by the event's fixed sequence_id.

5. Read paths

- get_all_chats(user) → Session.list(space_id, participants=[pid]), paginated.
- get_chat_by_thread_id → Session.list(..., description="/thread/{id}") first (indexed), falls back to a full paginated scan matching metadata.thread_id/metadata.user_uuid (covers legacy/unindexed sessions).
- get_chat_messages(chat_id) → session.events() (paginated) → filter to bodies that parse as a message (payload_to_message) → sort by sequence_id.
- get_last_messages(chat_ids) → session.events(last_n=100), take max sequence_id.
- Reverse lookups with no cached mapping (e.g. "which chat owns this tool_call uuid") fall back to scanning all sessions' events — _discover_chat_for_message / _discover_tool_call / _discover_reasoning. MemoryChatRepository/MemoryMessageRepository keep small in-process dict caches (_msg_chat, _tc_chat, …) plus ChatSessionRegistry (chat_uuid → session_id) to avoid re-scanning on the hot path.

6. Identifiers worth knowing

- MEMORY_APP_AGENT_PARTICIPANT_ID — a stable pseudo-ObjectId (sha256 of a fixed string) used as the emitter id for the agent side, since sessions can't have a second real participant.
- _MEMORY_MIN_LENGTH_PLACEHOLDER ("\u200b") — the memory-service body schema rejects empty strings (min_length=1) on some fields, so empty content is wire-encoded as a zero-width space and stripped back to "" on read (_wire_non_empty_str/_app_str).
- MEMORY_CHAT_MESSAGE_EVENT_TYPE = "message" — all chat-message events are tagged with this event_type; the memory-service's list API only accepts message | tool_output | status as filter values.

7. Sibling usage (same primitives, no events)

MemoryIdentityRepository and MemoryUserRepository (identity_repos.py, user_repos.py) also model rows as Sessions with metadata only (IDENTITY_METADATA_VERSION/USER_METADATA_VERSION discriminate document type in the shared space) — no Event log, since identities/user profiles aren't a sequence of messages. Same Session.create(deduplication_key=...) idempotency pattern, same indexed description trick (/user/{id}/identity/{provider}, /user/email/{email}).

---
Where to look in code: fastapi_server/app/memory/{repos,registry,sessions,constants,metadata_keys}.py for chat/message; fastapi_server/app/chats/__init__.py + app/messages/__init__.py for the domain models shared by both backends; app/ag_ui/storage.py for the AG-UI→storage state machine; app/deps.py:create_deps for backend selection.

# Goals

In the application utils package, I would like us to implement

1. ORM classes for Chat and Message that capture what exists in Agent Application.
2. Similarly, implementations for ChatRepository and MessageRepository that match Memory*Repository in Agent Application (dropping the Memory prefix as we won't implement the SQL alternative here).
3. 1-2 should be implemented in an extensible way. I.e. a consumer could add fields to Chat/Message by subclassing and the Repositories would persist and retrieve those extra fields.
4. An implementation of ag_ui storage for wrapping a stream of AG-UI events in a persistence layer. We don't have to implement the connecting piece in agent application that connect that stream to an agent. 
5. We should however, take a look at ../efm-agent-nvidia-v2 for the addition to this machinery that makes this storage stream (a) survive disconnection by consumer (so threads are completely consumed and persistence even with client disconnects) and (b) have an explicit cancelation mechanism that stops streaming/consumption and persists message as interrupted.
6. The ag_ui storage implementation should be extensible in four ways: (1) the user can override the AG-UI event -> message handling so for example to add custom fields, (2) the user could add handling of a new AG-UI event type (e.g. custom events), and (3) the user can override the default text event handling so that instead of saving everything to content it, e.g., could extract some structured content and store it in a different format in message, (4) it is loosely coupled to the Chat/Message Repository---by default it injects the classes defined in (2), but a user could implement the same interface (e.g. with a sql backend) and it would still work.
7. Thorough unit test coverage, including coverage of these extensibility points. Acceptance tests that test the e2e AG-UI consumption storage flow against real memory api.
8. Documentation of usage in docs/. For this, we also need to do some cleanup of previous application_utils documenation: (a) move docs/application_utils to docs/src/application_utils (mistake of prior PR) and (b) split out memory ORM from README.md, making README.md an index for memory ORM and this new chat history documentation.
9. Package conventions: minor version bumped (changelog, pyproject, task install to regenerate uv.lock), lint passes, tests (including new acceptance tests) pass.

# Package Structure Sketch

```
src/application_utils/
  chat_history/
    models.py (chat / message models)
    repositories.py (chat / message repo)
    ag_ui_storage.py (ag / ui adapter)
```