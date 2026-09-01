# Copyright 2025 DataRobot, Inc. and its affiliates.
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

"""Idempotent LlamaIndex auto-instrumentation for agent telemetry.

``opentelemetry-instrumentation-llamaindex`` spans every workflow step
(``AgentWorkflow`` is itself a ``Workflow``) but never attaches an agent
identity to them - :class:`_AgentNameSpanHandler` fills that gap by reacting
to the same span-enter notifications LlamaIndex's own instrumentation uses,
without needing to re-implement span creation.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

from datarobot_opentelemetry.semconv import SpanAttributes as DataRobotSpanAttributes
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.span.base import BaseSpan
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.context import Token
from opentelemetry.instrumentation.llamaindex import LlamaIndexInstrumentor
from pydantic import PrivateAttr

from datarobot_genai.core.telemetry.agent_identity import attach_agent_name_baggage
from datarobot_genai.core.telemetry.agent_identity import detach_agent_name_baggage

logger = logging.getLogger(__name__)

_INSTRUMENTED = {"llamaindex": False}


class _AgentNameSpanHandler(BaseSpanHandler[BaseSpan]):
    """Sets ``gen_ai.agent.name`` on the current span for any workflow step
    whose event carries ``current_agent_name`` - every step in this repo's
    ``AgentWorkflow`` (single- or multi-agent) does, via
    ``llama_index.core.agent.workflow.workflow_events``.

    Registered after :class:`LlamaIndexInstrumentor`'s own span handler, so
    by the time this fires the OTel span it just opened is already the
    current span in context - this only tags it, it never creates one.
    """

    # Keyed by span id so each span's own baggage detach is independent of any
    # other span's - nested agent spans must unwind in the same order they
    # attached, like a stack.
    _baggage_tokens: dict[str, Token[Context]] = PrivateAttr(default_factory=dict)

    @classmethod
    def class_name(cls) -> str:
        return "DataRobotAgentNameSpanHandler"

    def span_enter(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Any | None = None,  # noqa: ARG002
        parent_id: str | None = None,  # noqa: ARG002
        tags: dict[str, Any] | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        ev = bound_args.arguments.get("ev")
        agent_name = getattr(ev, "current_agent_name", None)
        if agent_name:
            trace.get_current_span().set_attribute(
                DataRobotSpanAttributes.GEN_AI_AGENT_NAME, agent_name
            )
            token = attach_agent_name_baggage(agent_name)
            if token is not None:
                self._baggage_tokens[id_] = token

    def new_span(self, *args: Any, **kwargs: Any) -> None:
        return None

    def span_exit(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,  # noqa: ARG002
        instance: Any | None = None,  # noqa: ARG002
        result: Any | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        detach_agent_name_baggage(self._baggage_tokens.pop(id_, None))

    def span_drop(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,  # noqa: ARG002
        instance: Any | None = None,  # noqa: ARG002
        err: BaseException | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        detach_agent_name_baggage(self._baggage_tokens.pop(id_, None))

    def prepare_to_exit_span(self, *args: Any, **kwargs: Any) -> None:
        return None

    def prepare_to_drop_span(self, *args: Any, **kwargs: Any) -> None:
        return None


def instrument() -> None:
    """Idempotently enable the LlamaIndex OpenTelemetry instrumentor."""
    if _INSTRUMENTED["llamaindex"]:
        logger.info("LlamaIndex instrumentation already enabled")
        return
    try:
        LlamaIndexInstrumentor().instrument()
        get_dispatcher().add_span_handler(_AgentNameSpanHandler())
        _INSTRUMENTED["llamaindex"] = True
    except Exception as e:
        logger.info(f"LlamaIndex instrumentation failed: {e}")
