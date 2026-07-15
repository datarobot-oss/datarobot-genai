# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AG-UI event stream → chat-history persistence.

:class:`AGUIStorageAgent` wraps an inner AG-UI agent.  It transparently forwards
the inner agent's event stream while, on an internal background task, folding
that same stream into persisted chat history: one Memory Service event per
logical message, with tool calls and reasoning steps nested in the message body.

Design
------
The agent is an *open* state machine — every seam a consumer might want to
customise is an overridable method:

* **Event dispatch registry** — :meth:`AGUIStorageAgent.event_handlers` maps each
  AG-UI event type to a handler method name.  Subclasses extend it to handle
  brand-new (e.g. custom) event types; unrecognised events fall through to
  :meth:`AGUIStorageAgent.handle_unknown_event` (a no-op by default).
* **Category handlers** — :meth:`~AGUIStorageAgent.handle_text_message`,
  :meth:`~AGUIStorageAgent.handle_tool_call`,
  :meth:`~AGUIStorageAgent.handle_reasoning`,
  :meth:`~AGUIStorageAgent.handle_run_lifecycle` and
  :meth:`~AGUIStorageAgent.handle_step`.  Override, for example,
  :meth:`~AGUIStorageAgent.handle_text_message` to extract structured content
  instead of appending raw deltas to ``content``.
* **Message-build hooks** — :meth:`~AGUIStorageAgent.build_message_create`,
  :meth:`~AGUIStorageAgent.build_tool_call_create`,
  :meth:`~AGUIStorageAgent.build_reasoning_create` and
  :meth:`~AGUIStorageAgent.message_update_fields`.  Override these to populate
  the extra fields a :class:`~.models.Message` / :class:`~.models.ToolCall`
  subclass declares.
* **Repository coupling** — the agent depends only on the
  :class:`~.repositories.ChatRepositoryLike` /
  :class:`~.repositories.MessageRepositoryLike` protocols, so any conforming
  backend (a SQL store, an in-memory fake) is a drop-in replacement.

Reasoning events use the (deprecated) AG-UI ``Thinking*`` events; the dispatch
registry makes adopting the drafted ``Reasoning*`` events a registration rather
than a rewrite.
"""

from __future__ import annotations

import abc
import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from ag_ui.core import BaseEvent
from ag_ui.core import RunAgentInput
from ag_ui.core import RunErrorEvent
from ag_ui.core import RunFinishedEvent
from ag_ui.core import RunStartedEvent
from ag_ui.core import StepFinishedEvent
from ag_ui.core import StepStartedEvent
from ag_ui.core import TextMessageChunkEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from ag_ui.core import ThinkingEndEvent
from ag_ui.core import ThinkingStartEvent
from ag_ui.core import ThinkingTextMessageContentEvent
from ag_ui.core import ThinkingTextMessageEndEvent
from ag_ui.core import ThinkingTextMessageStartEvent
from ag_ui.core import ToolCallArgsEvent
from ag_ui.core import ToolCallChunkEvent
from ag_ui.core import ToolCallEndEvent
from ag_ui.core import ToolCallResultEvent
from ag_ui.core import ToolCallStartEvent

from datarobot_genai.application_utils.chat_history.models import Chat
from datarobot_genai.application_utils.chat_history.models import ChatCreate
from datarobot_genai.application_utils.chat_history.models import Message
from datarobot_genai.application_utils.chat_history.models import MessageCreate
from datarobot_genai.application_utils.chat_history.models import MessageReasoningCreate
from datarobot_genai.application_utils.chat_history.models import MessageReasoningUpdate
from datarobot_genai.application_utils.chat_history.models import MessageStatus
from datarobot_genai.application_utils.chat_history.models import MessageToolCallCreate
from datarobot_genai.application_utils.chat_history.models import MessageToolCallUpdate
from datarobot_genai.application_utils.chat_history.models import MessageUpdate
from datarobot_genai.application_utils.chat_history.models import Reasoning
from datarobot_genai.application_utils.chat_history.models import Role
from datarobot_genai.application_utils.chat_history.models import ToolCall
from datarobot_genai.application_utils.chat_history.repositories import ChatRepositoryLike
from datarobot_genai.application_utils.chat_history.repositories import MessageRepositoryLike
from datarobot_genai.application_utils.chat_history.translate import ExtendedBaseMessage
from datarobot_genai.application_utils.chat_history.translate import translate_messages

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_CHAT_NAME_LIMIT = 100
_CHAT_NAME_DROP_LEEWAY = 30
#: Timestamps below this are almost certainly epoch *seconds*, not millis.
_EPOCH_MILLIS_FLOOR = 100_000_000_000


class ErrorCodes(StrEnum):
    """AG-UI ``RunErrorEvent`` codes emitted by the storage layer."""

    INVALID_INPUT = "INVALID_INPUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


def _epoch(seconds: int = 0) -> datetime:
    """Return a timezone-aware UTC datetime for *seconds* since the epoch."""
    return datetime.fromtimestamp(seconds, tz=UTC)


def _truncate_chat_name(text: str) -> str:
    """Truncate a chat name to a readable length on a word boundary.

    Parameters
    ----------
    text : str
        The candidate name (typically the first user message).

    Returns
    -------
    str
        *text* trimmed to at most :data:`_CHAT_NAME_LIMIT` characters, cut on a
        word boundary when one is close enough to the limit.
    """
    text = text.strip()
    if len(text) <= _CHAT_NAME_LIMIT:
        return text
    word_end = None
    for token in reversed(list(_TOKEN_RE.finditer(text[: _CHAT_NAME_LIMIT + 1]))):
        if token.end() <= _CHAT_NAME_LIMIT:
            word_end = token.end()
            break
    if word_end is None or (_CHAT_NAME_LIMIT - word_end) > _CHAT_NAME_DROP_LEEWAY:
        return text[:_CHAT_NAME_LIMIT].strip()
    return text[:word_end]


@dataclass
class StorageState:
    """Mutable per-run state threaded through the storage state machine.

    A fresh instance is created for each consumer run and reset on every
    :class:`ag_ui.core.RunStartedEvent`.
    """

    active_step: str | None = None
    current_event_timestamp: datetime = field(default_factory=_epoch)
    active_reasoning_title: str | None = None
    active_reasoning: Reasoning | None = None
    active_tool_call: ToolCall | None = None
    active_message: Message | None = None
    # Content buffers batch small streamed deltas to cut down on writes.
    buffered_message_content: str = ""
    buffered_tool_call_arguments: str = ""
    buffered_reasoning_content: str = ""

    def reset(self) -> None:
        """Clear all fields back to their defaults (on a new run)."""
        self.active_step = None
        self.current_event_timestamp = _epoch()
        self.active_reasoning_title = None
        self.active_reasoning = None
        self.active_tool_call = None
        self.active_message = None
        self.buffered_message_content = ""
        self.buffered_tool_call_arguments = ""
        self.buffered_reasoning_content = ""


class AGUIAgent(abc.ABC):
    """Minimal AG-UI agent contract: a named object exposing an event stream."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def run(self, input: RunAgentInput) -> AsyncGenerator[BaseEvent, None]:
        """Yield the agent's AG-UI :class:`~ag_ui.core.BaseEvent` stream."""
        ...


class AGUIStorageAgent(AGUIAgent):
    """Wrap an inner AG-UI agent, persisting its event stream as chat history.

    The wrapper is transparent: :meth:`run` yields the inner agent's events
    unchanged and in real time.  Persistence happens on a separate background
    task fed from an internal queue, so a slow or failing store never stalls the
    outgoing stream, and consumer disconnection does not abort persistence.
    """

    #: One-time warning latch for suspected epoch-seconds timestamps.
    _epoch_seconds_warning_emitted = False

    def __init__(
        self,
        name: str,
        user_id: UUID,
        chat_repo: ChatRepositoryLike,
        message_repo: MessageRepositoryLike,
        inner: AGUIAgent,
        minimal_chunk_to_persist: int = 5000,
        max_queue_size: int = 10_000,
        put_timeout: float = 0.1,
        translate: Any = None,
    ) -> None:
        """Initialise the storage agent.

        Parameters
        ----------
        name : str
            The wrapper agent's name (used as the ``name`` on persisted agent
            messages).
        user_id : UUID
            The owning user; scopes chat lookup / creation.
        chat_repo : ChatRepositoryLike
            Chat repository (any conforming backend).
        message_repo : MessageRepositoryLike
            Message repository (any conforming backend).
        inner : AGUIAgent
            The wrapped agent.  Its message ids are expected to be stable within
            a run.
        minimal_chunk_to_persist : int
            Flush a content buffer once it reaches this many characters.
        max_queue_size : int
            Maximum events buffered for persistence before back-pressure applies.
        put_timeout : float
            Maximum seconds to wait when enqueuing an event for persistence.
        translate : Any
            Optional override for the history-translation callable; defaults to
            :func:`~.translate.translate_messages`.

        Raises
        ------
        ValueError
            If *inner* is itself an :class:`AGUIStorageAgent`.
        """
        super().__init__(name)
        if isinstance(inner, AGUIStorageAgent):
            raise ValueError("Cannot wrap an AGUIStorageAgent with a second storage layer.")
        self._user_id = user_id
        self._chat_repo = chat_repo
        self._message_repo = message_repo
        self._inner = inner
        self._minimal_chunk_to_persist = minimal_chunk_to_persist
        self._max_queue_size = max_queue_size
        self._put_timeout = put_timeout
        self._translate = translate if translate is not None else translate_messages
        self._handlers: dict[type[BaseEvent], str] = self.event_handlers()

    # ── Extension point: event-type dispatch registry ─────────────────────────

    @classmethod
    def event_handlers(cls) -> dict[type[BaseEvent], str]:
        """Return the AG-UI-event-type → handler-method-name dispatch table.

        Override (typically ``{**super().event_handlers(), CustomEvent: "..."}``)
        to register a handler for a new event type.

        Returns
        -------
        dict[type[ag_ui.core.BaseEvent], str]
            Maps each handled event class to the name of the instance method that
            processes it.
        """
        return {
            RunStartedEvent: "handle_run_lifecycle",
            RunFinishedEvent: "handle_run_lifecycle",
            RunErrorEvent: "handle_run_lifecycle",
            StepStartedEvent: "handle_step",
            StepFinishedEvent: "handle_step",
            TextMessageStartEvent: "handle_text_message",
            TextMessageContentEvent: "handle_text_message",
            TextMessageEndEvent: "handle_text_message",
            TextMessageChunkEvent: "handle_text_message",
            ToolCallStartEvent: "handle_tool_call",
            ToolCallArgsEvent: "handle_tool_call",
            ToolCallResultEvent: "handle_tool_call",
            ToolCallEndEvent: "handle_tool_call",
            ToolCallChunkEvent: "handle_tool_call",
            ThinkingStartEvent: "handle_reasoning",
            ThinkingEndEvent: "handle_reasoning",
            ThinkingTextMessageStartEvent: "handle_reasoning",
            ThinkingTextMessageContentEvent: "handle_reasoning",
            ThinkingTextMessageEndEvent: "handle_reasoning",
        }

    def _resolve_handler(self, event_type: type[BaseEvent]) -> str | None:
        """Return the handler-method name for *event_type*, walking its MRO."""
        for klass in event_type.__mro__:
            name = self._handlers.get(klass)  # type: ignore[arg-type]
            if name is not None:
                return name
        return None

    async def handle_unknown_event(self, state: StorageState, chat: Chat, event: BaseEvent) -> None:
        """Handle an event with no registered handler.

        The default implementation ignores the event.  Override to persist
        custom event types.
        """
        logger.debug("No storage handler for event type %s; ignoring.", type(event).__name__)

    # ── Extension point: history translation ──────────────────────────────────

    def translate(self, messages: list[Message]) -> list[ExtendedBaseMessage]:
        """Translate stored messages into AG-UI history messages.

        Delegates to the injected translate callable (default
        :func:`~.translate.translate_messages`).  Override to customise the
        replayed history shape.
        """
        return list(self._translate(messages))

    # ── Extension point: message-build hooks ───────────────────────────────────

    def build_message_create(
        self,
        state: StorageState,
        chat: Chat,
        agui_id: str | None,
        role: str | None,
    ) -> MessageCreate:
        """Build the DTO used to create a new (agent) message.

        Override to populate extra fields declared on a :class:`~.models.Message`
        subclass (returning a matching ``MessageCreate`` subclass).

        Parameters
        ----------
        state : StorageState
            Current machine state (for ``active_step`` and the event timestamp).
        chat : Chat
            The chat the message belongs to.
        agui_id : str | None
            The AG-UI message id, when known.
        role : str | None
            The message role, defaulting to ``assistant``.

        Returns
        -------
        MessageCreate
            The DTO passed to :meth:`MessageRepositoryLike.create_message`.
        """
        return MessageCreate(
            step=state.active_step,
            chat_id=chat.chat_uuid,
            agui_id=agui_id,
            role=role or Role.ASSISTANT.value,
            name=self.name,
            content="",
            error=None,
            in_progress=True,
            timestamp=state.current_event_timestamp,
        )

    def build_tool_call_create(
        self,
        state: StorageState,
        tool_call_id: str,
        tool_call_name: str | None,
    ) -> MessageToolCallCreate:
        """Build the DTO used to append a tool call to the active message.

        Override to populate extra fields declared on a
        :class:`~.models.ToolCall` subclass.
        """
        assert state.active_message is not None
        return MessageToolCallCreate(
            tool_call_id=tool_call_id,
            agui_id=tool_call_id,
            message_uuid=state.active_message.message_uuid,
            role=Role.TOOL.value,
            name=tool_call_name or "UNKNOWN",
            created_at=state.current_event_timestamp,
        )

    def build_reasoning_create(
        self, state: StorageState, name: str | None
    ) -> MessageReasoningCreate:
        """Build the DTO used to append a reasoning step to the active message.

        Override to populate extra fields declared on a
        :class:`~.models.Reasoning` subclass.
        """
        assert state.active_message is not None
        return MessageReasoningCreate(
            role=Role.REASONING.value,
            message_uuid=state.active_message.message_uuid,
            name=name or "",
            created_at=state.current_event_timestamp,
        )

    def message_update_fields(self, state: StorageState, event: BaseEvent) -> dict[str, Any]:
        """Return extra fields to merge into a terminal message update.

        The default is empty.  Override to persist extra fields (declared on a
        :class:`~.models.Message` subclass) when a message completes.  Unknown
        keys are ignored by the base :class:`~.models.MessageUpdate`.
        """
        return {}

    # ── AGUIAgent.run — transparent forward + background persistence ───────────

    async def run(self, input: RunAgentInput) -> AsyncGenerator[BaseEvent, None]:
        """Persist inbound user messages, replay history, then stream the inner agent.

        Inbound *new* messages must be user messages belonging to this chat; a
        violation yields a terminal :class:`~ag_ui.core.RunErrorEvent` with the
        :attr:`ErrorCodes.INVALID_INPUT` code and stops the run.  The inner
        agent's stream is yielded verbatim while a background task persists it;
        on cancellation, still-active records are flipped to ``interrupted``, and
        when the inner agent crashes with a raw exception they are flipped to
        ``errored`` and the exception is re-raised.

        Parameters
        ----------
        input : RunAgentInput
            The AG-UI run input.  ``input.messages`` is replaced in place with
            the full translated chat history before the inner agent runs.

        Yields
        ------
        ag_ui.core.BaseEvent
            The inner agent's events (or a terminal error event).
        """
        chat = await self._resolve_or_create_chat(input)

        for message in input.messages:
            rejection = await self._persist_inbound_message(chat, message)
            if rejection is not None:
                yield rejection
                return

        # Replace the inbound messages with the full stored history so the inner
        # agent receives the conversation context.
        existing_messages = list(await self._message_repo.get_chat_messages(chat.chat_uuid))
        input.messages = self.translate(existing_messages)  # type: ignore[assignment]

        queue: asyncio.Queue[BaseEvent | None] = asyncio.Queue(maxsize=self._max_queue_size)
        consumer = asyncio.create_task(self._storage_consumer(queue, chat))

        cancelled = False
        error: Exception | None = None
        try:
            async for event in self._inner.run(input):
                self._backfill_timestamp(event)
                try:
                    await asyncio.wait_for(queue.put(event), timeout=self._put_timeout)
                except TimeoutError:
                    logger.error(
                        "Storage queue (max %d) full; dropping event from persistence after %.3fs.",
                        self._max_queue_size,
                        self._put_timeout,
                    )
                yield event
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception as exc:
            # The inner agent crashed mid-run with a raw exception (rather than a
            # terminal RunErrorEvent).  Record it so the finally clause finalises
            # still-active records as ``errored``, then re-raise: the producer
            # above relies on the raised exception to synthesize a client-facing
            # RunErrorEvent.
            error = exc
            raise
        finally:
            try:
                await asyncio.wait_for(queue.put(None), timeout=self._put_timeout)
            except TimeoutError:
                logger.error("Storage queue full; could not enqueue the finalization sentinel.")
            try:
                await consumer
            except Exception:
                logger.warning("Storage consumer task failed", exc_info=True)
            if cancelled:
                await self._finalize_interrupted(chat)
            elif error is not None:
                await self._finalize_errored(chat, str(error) or repr(error))

    # ── Chat resolution / inbound persistence ─────────────────────────────────

    async def _resolve_or_create_chat(self, input: RunAgentInput) -> Chat:
        """Return the chat for this thread, creating it (named from the first message)."""
        existing = await self._chat_repo.get_chat_by_thread_id(self._user_id, input.thread_id)
        if existing is not None:
            return existing

        first = input.messages[0] if input.messages else None
        if first is not None and isinstance(first.content, str) and first.content.strip():
            chat_name = _truncate_chat_name(first.content)
        else:
            chat_name = "New Chat"
        return await self._chat_repo.create_chat(
            ChatCreate(user_uuid=self._user_id, name=chat_name, thread_id=input.thread_id)
        )

    async def _persist_inbound_message(self, chat: Chat, message: Any) -> RunErrorEvent | None:
        """Persist one inbound message, or return a terminal rejection event.

        Parameters
        ----------
        chat : Chat
            The resolved chat.
        message : Any
            An AG-UI input message.

        Returns
        -------
        RunErrorEvent | None
            An :attr:`ErrorCodes.INVALID_INPUT` error event when the message is
            invalid (already lives in another chat, or is a *new* non-user
            message); otherwise ``None`` after persisting a new user message.
        """
        existing = await self._message_repo.get_message_by_agui_id(chat.chat_uuid, message.id)
        if existing is not None:
            if chat.chat_uuid != existing.chat_id:
                return RunErrorEvent(
                    message="Messages do not all belong to the same chat",
                    code=ErrorCodes.INVALID_INPUT.value,
                )
            return None

        if message.role != Role.USER.value:
            return RunErrorEvent(
                message="The user cannot create new non-user messages.",
                code=ErrorCodes.INVALID_INPUT.value,
            )

        await self._message_repo.create_message(
            MessageCreate(
                chat_id=chat.chat_uuid,
                role=Role.USER.value,
                agui_id=message.id,
                name=message.name or "",
                content=message.content or "",
                error=None,
                in_progress=False,
                status=MessageStatus.COMPLETE.value,
            )
        )
        return None

    # ── Timestamp handling ─────────────────────────────────────────────────────

    def _backfill_timestamp(self, event: BaseEvent) -> None:
        """Ensure *event* has a timestamp; warn once on a likely epoch-seconds value.

        A missing timestamp is backfilled with the current epoch-millis so the
        persisted timeline stays monotonic even when an upstream framework omits
        it (e.g. a new user message arriving mid-response).
        """
        if not event.timestamp:
            event.timestamp = int(datetime.now(UTC).timestamp() * 1_000)
        if not self._epoch_seconds_warning_emitted and event.timestamp < _EPOCH_MILLIS_FLOOR:
            logger.warning(
                "Received a timestamp that is probably epoch seconds when expecting epoch millis!"
            )
            self._epoch_seconds_warning_emitted = True

    @staticmethod
    def _epoch_milli_or_now(timestamp: int | None) -> datetime:
        """Convert an epoch-millis timestamp to a UTC datetime, falling back to now."""
        if timestamp is None:
            return datetime.now(UTC)
        try:
            return datetime.fromtimestamp(timestamp / 1_000, tz=UTC)
        except (ValueError, OSError, OverflowError):
            return datetime.now(UTC)

    # ── Background consumer ────────────────────────────────────────────────────

    async def _storage_consumer(self, queue: asyncio.Queue[BaseEvent | None], chat: Chat) -> None:
        """Drain the persistence queue, dispatching each event, until the sentinel.

        Per-event errors are logged and swallowed so one bad event never stalls
        persistence of the rest; a guaranteed final flush writes any buffered
        content on completion.
        """
        state = StorageState()
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                try:
                    state.current_event_timestamp = self._epoch_milli_or_now(event.timestamp)
                    await self._dispatch(state, chat, event)
                except Exception:
                    logger.error("Error processing event in storage consumer", exc_info=True)
                    continue
        except Exception:
            logger.error("Storage consumer task failed", exc_info=True)
        finally:
            try:
                await self.flush_message_buffer(state)
                await self.flush_reasoning_buffer(state)
                await self.flush_tool_call_buffer(state)
            except Exception:
                logger.error("Error during final flush in storage consumer", exc_info=True)

    async def _dispatch(self, state: StorageState, chat: Chat, event: BaseEvent) -> None:
        """Route *event* to its registered handler, or to ``handle_unknown_event``."""
        handler_name = self._resolve_handler(type(event))
        if handler_name is None:
            await self.handle_unknown_event(state, chat, event)
            return
        await getattr(self, handler_name)(state, chat, event)

    # ── Category handlers ──────────────────────────────────────────────────────

    async def handle_run_lifecycle(self, state: StorageState, chat: Chat, event: BaseEvent) -> None:
        """Reset state on run start; flush and finalise records on finish / error."""
        if isinstance(event, RunStartedEvent):
            state.reset()
            return

        if isinstance(event, (RunFinishedEvent, RunErrorEvent)):
            await self.flush_message_buffer(state)
            await self.flush_reasoning_buffer(state)
            await self.flush_tool_call_buffer(state)

        if isinstance(event, RunFinishedEvent):
            if state.active_message is not None:
                await self._message_repo.update_message(
                    state.active_message.message_uuid,
                    MessageUpdate(
                        in_progress=False,
                        status=MessageStatus.COMPLETE.value,
                        **self.message_update_fields(state, event),
                    ),
                )
            if state.active_reasoning is not None:
                await self._message_repo.update_message_reasoning(
                    state.active_reasoning.uuid,
                    MessageReasoningUpdate(in_progress=False, status=MessageStatus.COMPLETE.value),
                )
            if state.active_tool_call is not None:
                await self._message_repo.update_message_tool_call(
                    state.active_tool_call.uuid,
                    MessageToolCallUpdate(in_progress=False, status=MessageStatus.COMPLETE.value),
                )
        elif isinstance(event, RunErrorEvent):
            error = f"[{event.code}] {event.message}" if event.code else event.message
            if state.active_message is not None:
                await self._message_repo.update_message(
                    state.active_message.message_uuid,
                    MessageUpdate(
                        in_progress=False, error=error, status=MessageStatus.ERRORED.value
                    ),
                )
            if state.active_reasoning is not None:
                await self._message_repo.update_message_reasoning(
                    state.active_reasoning.uuid,
                    MessageReasoningUpdate(
                        in_progress=False, error=error, status=MessageStatus.ERRORED.value
                    ),
                )
            if state.active_tool_call is not None:
                await self._message_repo.update_message_tool_call(
                    state.active_tool_call.uuid,
                    MessageToolCallUpdate(
                        in_progress=False, error=error, status=MessageStatus.ERRORED.value
                    ),
                )

    async def handle_step(self, state: StorageState, chat: Chat, event: BaseEvent) -> None:
        """Track the active step name across ``StepStarted`` / ``StepFinished``."""
        if isinstance(event, StepStartedEvent):
            state.active_step = event.step_name
        elif isinstance(event, StepFinishedEvent):
            state.active_step = None

    async def handle_text_message(self, state: StorageState, chat: Chat, event: BaseEvent) -> None:
        """Fold text-message events onto the active message's ``content``.

        Override to store structured content instead of raw appended deltas.
        """
        if isinstance(event, TextMessageStartEvent):
            await self._ensure_message_exists(state, chat, event.message_id, event.role)
        elif isinstance(event, TextMessageContentEvent):
            await self._ensure_message_exists(state, chat, event.message_id, None)
            assert state.active_message is not None
            state.buffered_message_content += event.delta
            if len(state.buffered_message_content) >= self._minimal_chunk_to_persist:
                await self.flush_message_buffer(state)
        elif isinstance(event, TextMessageEndEvent):
            await self._ensure_message_exists(state, chat, event.message_id, None)
            assert state.active_message is not None
            await self.flush_message_buffer(state)
            await self._message_repo.update_message(
                state.active_message.message_uuid, MessageUpdate(in_progress=False)
            )
        elif isinstance(event, TextMessageChunkEvent):
            await self._ensure_message_exists(state, chat, event.message_id, None)
            assert state.active_message is not None
            state.active_message.content += event.delta or ""
            await self._message_repo.update_message(
                state.active_message.message_uuid,
                MessageUpdate(content=state.active_message.content, in_progress=False),
            )

    async def handle_tool_call(self, state: StorageState, chat: Chat, event: BaseEvent) -> None:
        """Fold tool-call events onto a tool call nested in the active message."""
        if isinstance(event, ToolCallStartEvent):
            await self._ensure_message_exists(state, chat, event.parent_message_id, None)
            await self._ensure_tool_call_exists(state, event.tool_call_id, event.tool_call_name)
        elif isinstance(event, ToolCallArgsEvent):
            await self._ensure_message_exists(state, chat, None, None)
            await self._ensure_tool_call_exists(state, event.tool_call_id, None)
            assert state.active_tool_call is not None
            state.buffered_tool_call_arguments += event.delta
            if len(state.buffered_tool_call_arguments) >= self._minimal_chunk_to_persist:
                await self.flush_tool_call_buffer(state)
        elif isinstance(event, ToolCallResultEvent):
            await self._ensure_message_exists(state, chat, None, None)
            await self._ensure_tool_call_exists(state, event.tool_call_id, None)
            assert state.active_tool_call is not None
            await self.flush_tool_call_buffer(state)
            await self._message_repo.update_message_tool_call(
                state.active_tool_call.uuid, MessageToolCallUpdate(content=event.content)
            )
        elif isinstance(event, ToolCallEndEvent):
            await self._ensure_message_exists(state, chat, None, None)
            await self._ensure_tool_call_exists(state, event.tool_call_id, None)
            assert state.active_tool_call is not None
            await self.flush_tool_call_buffer(state)
            await self._message_repo.update_message_tool_call(
                state.active_tool_call.uuid, MessageToolCallUpdate(in_progress=False)
            )
        elif isinstance(event, ToolCallChunkEvent):
            await self._ensure_message_exists(state, chat, event.parent_message_id, None)
            fallback = state.active_tool_call.tool_call_id if state.active_tool_call else None
            await self._ensure_tool_call_exists(
                state, event.tool_call_id or fallback or "", event.tool_call_name
            )
            assert state.active_tool_call is not None
            state.buffered_tool_call_arguments += event.delta or ""
            if len(state.buffered_tool_call_arguments) >= self._minimal_chunk_to_persist:
                await self.flush_tool_call_buffer(state)

    async def handle_reasoning(self, state: StorageState, chat: Chat, event: BaseEvent) -> None:
        """Fold ``Thinking*`` events onto a reasoning step nested in the message.

        Note
        ----
        AG-UI reasoning is still drafted; the (deprecated) ``Thinking*`` events
        are handled here.  Adopting the drafted ``Reasoning*`` events is a matter
        of registering them in :meth:`event_handlers`.
        """
        if isinstance(event, ThinkingStartEvent):
            state.active_reasoning_title = event.title
        elif isinstance(event, ThinkingEndEvent):
            await self.flush_reasoning_buffer(state)
            state.active_reasoning_title = None
            if state.active_reasoning is not None:
                await self._message_repo.update_message_reasoning(
                    state.active_reasoning.uuid, MessageReasoningUpdate(in_progress=False)
                )
                state.active_reasoning = None
        elif isinstance(event, ThinkingTextMessageStartEvent):
            await self._ensure_message_exists(state, chat, None, None)
            assert state.active_message is not None
            state.active_reasoning = await self._message_repo.create_message_reasoning(
                self.build_reasoning_create(state, state.active_reasoning_title)
            )
        elif isinstance(event, ThinkingTextMessageContentEvent):
            await self._ensure_message_exists(state, chat, None, None)
            assert state.active_message is not None
            await self._ensure_active_reasoning(state)
            assert state.active_reasoning is not None
            if isinstance(event.delta, str):
                state.buffered_reasoning_content += event.delta
            elif isinstance(event.delta, list):
                state.buffered_reasoning_content += "\n" + json.dumps(event.delta)
            else:
                logger.warning("Received reasoning '%s' of unanticipated type.", event.delta)
            if len(state.buffered_reasoning_content) >= self._minimal_chunk_to_persist:
                await self.flush_reasoning_buffer(state)
        elif isinstance(event, ThinkingTextMessageEndEvent):
            await self._ensure_message_exists(state, chat, None, None)
            assert state.active_message is not None
            await self._ensure_active_reasoning(state)
            assert state.active_reasoning is not None
            await self.flush_reasoning_buffer(state)
            await self._message_repo.update_message_reasoning(
                state.active_reasoning.uuid, MessageReasoningUpdate(in_progress=False)
            )
            state.active_reasoning = None

    async def _ensure_active_reasoning(self, state: StorageState) -> None:
        """Ensure ``state.active_reasoning`` points at a live in-progress reasoning.

        Reloads the active message so its nested reasonings are fresh, reuses the
        latest still-in-progress one, or creates a new reasoning step.
        """
        if state.active_reasoning is not None:
            return
        assert state.active_message is not None
        message = state.active_message
        if message.chat_id is not None and message.agui_id is not None:
            reloaded = await self._message_repo.get_message_by_agui_id(
                message.chat_id, message.agui_id
            )
            if reloaded is not None:
                state.active_message = reloaded
                message = reloaded
        latest = next(
            iter(
                sorted(
                    (r for r in message.reasonings if r.in_progress),
                    key=lambda r: r.created_at,
                    reverse=True,
                )
            ),
            None,
        )
        if latest is not None:
            state.active_reasoning = latest
        else:
            state.active_reasoning = await self._message_repo.create_message_reasoning(
                self.build_reasoning_create(state, state.active_reasoning_title)
            )

    # ── Buffer flushers ────────────────────────────────────────────────────────

    async def flush_message_buffer(self, state: StorageState) -> None:
        """Persist and clear buffered message content, if any."""
        if state.active_message is not None and state.buffered_message_content:
            state.active_message.content += state.buffered_message_content
            await self._message_repo.update_message(
                state.active_message.message_uuid,
                MessageUpdate(content=state.active_message.content),
            )
            state.buffered_message_content = ""

    async def flush_tool_call_buffer(self, state: StorageState) -> None:
        """Persist and clear buffered tool-call arguments, if any."""
        if state.active_tool_call is not None and state.buffered_tool_call_arguments:
            state.active_tool_call.arguments += state.buffered_tool_call_arguments
            await self._message_repo.update_message_tool_call(
                state.active_tool_call.uuid,
                MessageToolCallUpdate(arguments=state.active_tool_call.arguments),
            )
            state.buffered_tool_call_arguments = ""

    async def flush_reasoning_buffer(self, state: StorageState) -> None:
        """Persist and clear buffered reasoning content, if any."""
        if state.active_reasoning is not None and state.buffered_reasoning_content:
            state.active_reasoning.content += state.buffered_reasoning_content
            await self._message_repo.update_message_reasoning(
                state.active_reasoning.uuid,
                MessageReasoningUpdate(content=state.active_reasoning.content),
            )
            state.buffered_reasoning_content = ""

    # ── One-event-per-message folding ─────────────────────────────────────────

    async def _ensure_message_exists(
        self,
        state: StorageState,
        chat: Chat,
        agui_id: str | None,
        role: str | None,
    ) -> None:
        """Point ``state.active_message`` at the message this event belongs to.

        Starting a message with a *new* ``agui_id`` closes out (flushes +
        finalises) the previous active message.  An existing message is reused by
        ``agui_id`` when given, otherwise by matching the chat's last message's
        role; failing both, a new message is created via
        :meth:`build_message_create`.
        """
        active_message = state.active_message

        # Starting a different message: close out the prior one.
        if agui_id and active_message is not None and active_message.agui_id != agui_id:
            update = MessageUpdate(in_progress=False)
            if state.buffered_message_content:
                active_message.content += state.buffered_message_content
                update.content = active_message.content
            await self._message_repo.update_message(active_message.message_uuid, update)
            active_message = None

        if active_message is None:
            state.buffered_message_content = ""
            if agui_id:
                retrieved = await self._message_repo.get_message_by_agui_id(chat.chat_uuid, agui_id)
                active_message = retrieved or await self._message_repo.create_message(
                    self.build_message_create(state, chat, agui_id, role)
                )
            else:
                last = (await self._message_repo.get_last_messages([chat.chat_uuid])).get(
                    chat.chat_uuid
                )
                if last is not None and last.role == (role or Role.ASSISTANT.value):
                    active_message = last
                else:
                    active_message = await self._message_repo.create_message(
                        self.build_message_create(state, chat, agui_id, role)
                    )

        state.active_message = active_message

    async def _ensure_tool_call_exists(
        self,
        state: StorageState,
        tool_call_id: str,
        tool_call_name: str | None,
    ) -> None:
        """Point ``state.active_tool_call`` at the tool call for *tool_call_id*.

        Reuses the active tool call, an existing one found by AG-UI id, or
        creates a new one via :meth:`build_tool_call_create`.

        Raises
        ------
        RuntimeError
            When there is no active message to attach the tool call to.
        """
        if state.active_message is None:
            raise RuntimeError(f"Creating {tool_call_id} with no corresponding active message")
        if (
            state.active_tool_call is not None
            and state.active_tool_call.tool_call_id == tool_call_id
        ):
            return
        existing = await self._message_repo.get_tool_call_by_agui_id(
            state.active_message.message_uuid, tool_call_id
        )
        state.active_tool_call = existing or await self._message_repo.create_message_tool_call(
            self.build_tool_call_create(state, tool_call_id, tool_call_name)
        )

    # ── Interruption finalisation ──────────────────────────────────────────────

    async def _finalize_interrupted(self, chat: Chat) -> None:
        """Flip every still-in-progress record in the chat to ``interrupted``."""
        try:
            for msg in await self._message_repo.get_chat_messages(chat.chat_uuid):
                if msg.in_progress:
                    await self._message_repo.update_message(
                        msg.message_uuid,
                        MessageUpdate(in_progress=False, status=MessageStatus.INTERRUPTED.value),
                    )
                for tc in msg.tool_calls:
                    if tc.in_progress:
                        await self._message_repo.update_message_tool_call(
                            tc.uuid,
                            MessageToolCallUpdate(
                                in_progress=False, status=MessageStatus.INTERRUPTED.value
                            ),
                        )
                for r in msg.reasonings:
                    if r.in_progress:
                        await self._message_repo.update_message_reasoning(
                            r.uuid,
                            MessageReasoningUpdate(
                                in_progress=False, status=MessageStatus.INTERRUPTED.value
                            ),
                        )
        except Exception:
            logger.warning("Failed to finalize interrupted messages", exc_info=True)

    async def _finalize_errored(self, chat: Chat, error_message: str) -> None:
        """Flip every still-in-progress record in the chat to ``errored``.

        Used when the inner agent crashes mid-run with a raw Python exception
        (rather than emitting a terminal :class:`~ag_ui.core.RunErrorEvent`), so
        that records left mid-stream are not persisted as dangling ``active``.

        Parameters
        ----------
        chat : Chat
            The chat whose still-in-progress records are finalised.
        error_message : str
            The error detail recorded on each finalised record.
        """
        try:
            for msg in await self._message_repo.get_chat_messages(chat.chat_uuid):
                if msg.in_progress:
                    await self._message_repo.update_message(
                        msg.message_uuid,
                        MessageUpdate(
                            in_progress=False,
                            error=error_message,
                            status=MessageStatus.ERRORED.value,
                        ),
                    )
                for tc in msg.tool_calls:
                    if tc.in_progress:
                        await self._message_repo.update_message_tool_call(
                            tc.uuid,
                            MessageToolCallUpdate(
                                in_progress=False,
                                error=error_message,
                                status=MessageStatus.ERRORED.value,
                            ),
                        )
                for r in msg.reasonings:
                    if r.in_progress:
                        await self._message_repo.update_message_reasoning(
                            r.uuid,
                            MessageReasoningUpdate(
                                in_progress=False,
                                error=error_message,
                                status=MessageStatus.ERRORED.value,
                            ),
                        )
        except Exception:
            logger.warning("Failed to finalize errored messages", exc_info=True)
