# Copyright 2025 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import json
import logging
import queue
import uuid
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from typing import Any

from crewai.events.event_bus import CrewAIEventsBus
from crewai.events.event_types import AgentExecutionStartedEvent
from crewai.events.event_types import AgentReasoningCompletedEvent
from crewai.events.event_types import AgentReasoningFailedEvent
from crewai.events.event_types import AgentReasoningStartedEvent
from crewai.events.event_types import ToolUsageErrorEvent
from crewai.events.event_types import ToolUsageFinishedEvent
from crewai.events.event_types import ToolUsageStartedEvent

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    """Neutral CrewAI tool event; the dragent converts it to AG-UI ``ToolCall*``.

    ``kind="call"`` = invocation (name + args); ``kind="result"`` = its output or error.
    """

    kind: str
    tool_call_id: str
    name: str = ""
    args: str = ""
    content: str = ""
    # Owning agent, captured when the call fires; the invoke loop opens its step before
    # emitting the call (native tool calls send no content chunk to drive the transition).
    agent_role: str = ""


class CrewAIStreamingEventListener:
    """Collects CrewAI events to distinguish the types of events currently happening."""

    def __init__(self) -> None:
        self.reasoning_event = False
        # Active agent role from the bus: the gateway LLM streams bare chunks with no agent_role,
        # so chunk.agent_role is empty and the dragent falls back to this for per-agent steps.
        self.active_agent_role: str = ""
        # Tool-call records from CrewAI's tool-usage bus events, drained by the invoke loop.
        # Handlers fire on a thread pool, so this is read cross-thread (Queue is thread-safe).
        self.tool_call_events: queue.Queue[ToolCallRecord] = queue.Queue()
        # Pair each Finished/Error to its Started by (tool_name, tool_args), not arrival order:
        # thread-pool handlers arrive out of order and carry no per-call id. Buffer whichever
        # side comes first until its partner arrives. Relies on CrewAI echoing tool_args on the
        # Finished/Error event (it does today); if that stops, a result would buffer unpaired.
        # Two calls with identical (name, args) pair FIFO -- indistinguishable without a per-call
        # id, so a non-deterministic tool's two results may swap.
        self._calls_awaiting_result: dict[tuple[str, str], deque[str]] = defaultdict(deque)
        self._results_awaiting_call: dict[tuple[str, str], deque[str]] = defaultdict(deque)

    @staticmethod
    def _pair_key(event: Any) -> tuple[str, str]:
        """Stable key shared by a tool's Started and its Finished/Error events."""
        name = getattr(event, "tool_name", "") or "tool"
        args = getattr(event, "tool_args", "") or ""
        if not isinstance(args, str):
            args = json.dumps(args, default=str, sort_keys=True)
        return name, args

    def setup_listeners(self, crewai_event_bus: CrewAIEventsBus) -> None:

        @crewai_event_bus.on(AgentReasoningStartedEvent)
        def on_agent_reasoning_started(_: Any, event: Any) -> None:
            logger.debug("Reasoning started")
            self.reasoning_event = True

        @crewai_event_bus.on(AgentReasoningCompletedEvent)
        def on_agent_reasoning_completed(_: Any, event: Any) -> None:
            logger.debug("Reasoning completed")
            self.reasoning_event = False

        @crewai_event_bus.on(AgentReasoningFailedEvent)
        def on_agent_reasoning_failed(_: Any, event: Any) -> None:
            logger.debug("Reasoning failed")
            self.reasoning_event = False

        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def on_agent_execution_started(_: Any, event: Any) -> None:
            # Use event.agent.role; the flat event.agent_role is None in current CrewAI.
            agent = getattr(event, "agent", None)
            self.active_agent_role = (getattr(agent, "role", "") or "") if agent else ""
            logger.debug("Agent execution started: %s", self.active_agent_role)

        def put_result(event: Any, content: str) -> None:
            key = self._pair_key(event)
            calls = self._calls_awaiting_result.get(key)
            if calls:  # its Started already arrived -> pair to that call
                tool_call_id = calls.popleft()
                if not calls:
                    del self._calls_awaiting_result[key]
                self.tool_call_events.put(
                    ToolCallRecord(kind="result", tool_call_id=tool_call_id, content=content)
                )
            else:  # Finished/Error beat its Started (handler reorder) -> buffer it
                self._results_awaiting_call[key].append(content)

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_tool_usage_started(_: Any, event: Any) -> None:
            key = self._pair_key(event)
            tool_call_id = str(uuid.uuid4())
            # Attribute by the event's own agent; active_agent_role may already point at the next
            # agent (handler raced ahead), so fall back to it only when the event carries no role.
            agent_role = (getattr(event, "agent_role", None) or "") or self.active_agent_role
            self.tool_call_events.put(
                ToolCallRecord(
                    kind="call",
                    tool_call_id=tool_call_id,
                    name=key[0],
                    args=key[1],
                    agent_role=agent_role,
                )
            )
            buffered = self._results_awaiting_call.get(key)
            if buffered:  # its result already arrived (handler reorder) -> emit it now, in order
                self.tool_call_events.put(
                    ToolCallRecord(
                        kind="result", tool_call_id=tool_call_id, content=buffered.popleft()
                    )
                )
                if not buffered:
                    del self._results_awaiting_call[key]
            else:
                self._calls_awaiting_result[key].append(tool_call_id)

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_tool_usage_finished(_: Any, event: Any) -> None:
            put_result(event, str(getattr(event, "output", "")))

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def on_tool_usage_error(_: Any, event: Any) -> None:
            put_result(event, f"Error: {getattr(event, 'error', '')}")
