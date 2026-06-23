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
"""Real-time logging of CrewAI agent/task/tool lifecycle events.

CrewAI's per-step detail (which agent, which tool, with what args and result)
never reaches the Python logger on its own, so a dragent run shows little more
than a stream of LiteLLM calls. This listener subscribes to the CrewAI event
bus and logs that lifecycle as it happens.
"""

from __future__ import annotations

import logging
from typing import Any

from crewai.events.event_bus import CrewAIEventsBus
from crewai.events.event_types import AgentExecutionStartedEvent
from crewai.events.event_types import AgentReasoningCompletedEvent
from crewai.events.event_types import AgentReasoningFailedEvent
from crewai.events.event_types import AgentReasoningStartedEvent
from crewai.events.event_types import CrewKickoffStartedEvent
from crewai.events.event_types import TaskCompletedEvent
from crewai.events.event_types import TaskFailedEvent
from crewai.events.event_types import TaskStartedEvent
from crewai.events.event_types import ToolUsageErrorEvent
from crewai.events.event_types import ToolUsageFinishedEvent
from crewai.events.event_types import ToolUsageStartedEvent

logger = logging.getLogger(__name__)

_MAX_LEN = 200


def _truncate(value: Any, limit: int = _MAX_LEN) -> str:
    text = str(value)
    return text if len(text) <= limit else f"{text[:limit]}…"


def _role(event: Any) -> str:
    # CrewAI populates `agent_role` on tool events but leaves it blank on
    # agent/task lifecycle events, so fall back to the attached agent object.
    role = getattr(event, "agent_role", "") or ""
    if role:
        return role
    agent = getattr(event, "agent", None) or getattr(getattr(event, "task", None), "agent", None)
    return getattr(agent, "role", "") or ""


class CrewAILoggingEventListener:
    """Logs the CrewAI agent/task/tool lifecycle for dragent observability."""

    def setup_listeners(self, crewai_event_bus: CrewAIEventsBus) -> None:
        def _register(event_type: Any, render: Any) -> None:
            @crewai_event_bus.on(event_type)
            def _handler(_: Any, event: Any) -> None:
                # A logging handler must never break the crew run, so swallow
                # anything an unexpected payload throws.
                try:
                    render(event)
                except Exception:
                    logger.debug("CrewAI logging handler error", exc_info=True)

        _register(
            CrewKickoffStartedEvent,
            lambda e: logger.debug("crew kickoff: inputs=%s", _truncate(getattr(e, "inputs", ""))),
        )
        _register(
            AgentExecutionStartedEvent,
            lambda e: logger.info("[%s] agent started", _role(e)),
        )
        _register(
            TaskStartedEvent,
            lambda e: logger.info("[%s] task started: %s", _role(e), getattr(e, "task_name", "")),
        )
        _register(
            TaskCompletedEvent,
            lambda e: logger.info(
                "[%s] task completed: %s", _role(e), _truncate(getattr(e, "output", ""))
            ),
        )
        _register(
            TaskFailedEvent,
            lambda e: logger.warning(
                "[%s] task failed: %s", _role(e), _truncate(getattr(e, "error", ""))
            ),
        )
        _register(
            ToolUsageStartedEvent,
            lambda e: logger.info(
                "[%s] tool call: %s(%s) attempt=%s",
                _role(e),
                getattr(e, "tool_name", ""),
                _truncate(getattr(e, "tool_args", "")),
                getattr(e, "run_attempts", ""),
            ),
        )
        _register(
            ToolUsageFinishedEvent,
            lambda e: logger.info(
                "[%s] tool %s -> %s",
                _role(e),
                getattr(e, "tool_name", ""),
                _truncate(getattr(e, "output", "")),
            ),
        )
        _register(
            ToolUsageErrorEvent,
            lambda e: logger.warning(
                "[%s] tool %s error: %s",
                _role(e),
                getattr(e, "tool_name", ""),
                _truncate(getattr(e, "error", "")),
            ),
        )
        _register(
            AgentReasoningStartedEvent,
            lambda e: logger.debug("[%s] reasoning started", _role(e)),
        )
        _register(
            AgentReasoningCompletedEvent,
            lambda e: logger.debug("[%s] reasoning completed", _role(e)),
        )
        _register(
            AgentReasoningFailedEvent,
            lambda e: logger.debug("[%s] reasoning failed", _role(e)),
        )
