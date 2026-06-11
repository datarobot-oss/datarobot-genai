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
"""``create_agent`` middleware for reasoning-model robustness.

A reasoning model (e.g. ``gpt-oss``) whose prompt urges tool use can spend its
whole turn planning a tool call that the bound tool set cannot satisfy, and
return an ``AIMessage`` with **no tool calls and no answer text** — only
reasoning. ``create_agent``'s loop exits on zero tool calls, so that answerless
message becomes the node's final output and propagates (e.g. an inter-agent
relay forwards empty content and the run produces no real answer).

``FinalAnswerMiddleware`` detects that dead-end after the model call and
re-invokes the model once with an ephemeral nudge to produce the final answer.
It is cause-agnostic: a final ``AIMessage`` with neither text nor tool calls is
never a useful outcome, so recovery cannot misfire on healthy turns.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware import ModelCallResult
from langchain.agents.middleware import ModelRequest
from langchain.agents.middleware import ModelResponse
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from datarobot_genai.core.agents.reasoning import flatten_to_text

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_DEFAULT_NUDGE = (
    "Provide your final answer to the user now, based on your analysis above. "
    "Do not call any tools."
)


def _last_ai_message(response: ModelCallResult | ModelResponse[Any]) -> AIMessage | None:
    """Return the last ``AIMessage`` from any ``ModelCallResult`` shape."""
    if isinstance(response, AIMessage):
        return response
    result = getattr(response, "result", None)
    if result is None:
        # ExtendedModelResponse wraps the ModelResponse.
        inner = getattr(response, "model_response", None)
        result = getattr(inner, "result", None) if inner is not None else None
    for message in reversed(result or []):
        if isinstance(message, AIMessage):
            return message
    return None


def _is_dead_end(response: ModelCallResult | ModelResponse[Any]) -> bool:
    """Return True when the model produced neither tool calls nor answer text.

    ``flatten_to_text`` keeps only text blocks, so both an empty ``content``
    and a thinking-only content list (the shapes ``ChatLiteLLM`` produces for a
    reasoning turn with no answer) read as empty.
    """
    message = _last_ai_message(response)
    if message is None:
        return False
    return not message.tool_calls and not flatten_to_text(message.content).strip()


class FinalAnswerMiddleware(AgentMiddleware):
    """Recover a final answer when a reasoning model returns none.

    On a dead-end model turn (no tool calls, no answer text), re-invoke the
    model once with the conversation plus an ephemeral nudge message; when
    tools are bound, the retry also sets ``tool_choice="none"`` so the model
    cannot stall on another tool attempt. The dead-end message is discarded —
    only the returned response is committed to graph state. If the retry is
    also a dead-end it is returned as-is (with a warning logged); reasoning is
    never promoted to answer content.

    Usage::

        agent = create_agent(llm, tools=tools, middleware=[FinalAnswerMiddleware()])
    """

    def __init__(self, *, nudge: str = _DEFAULT_NUDGE) -> None:
        super().__init__()
        self.tools: list[BaseTool] = []
        self.nudge = nudge

    def _finalize_request(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        return request.override(
            messages=[*request.messages, HumanMessage(content=self.nudge)],
            tool_choice="none" if request.tools else request.tool_choice,
        )

    def _log_recovery(self, recovered: ModelCallResult | ModelResponse[Any]) -> None:
        if _is_dead_end(recovered):
            logger.warning(
                "Model returned no tool calls and no answer text even after a "
                "finalize nudge; returning the answerless response unchanged."
            )
        else:
            logger.info("Recovered a final answer from a dead-end model turn.")

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelCallResult:
        """Run the model; on a dead-end, retry once with a finalize nudge."""
        response: ModelCallResult = handler(request)
        if _is_dead_end(response):
            response = handler(self._finalize_request(request))
            self._log_recovery(response)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelCallResult:
        """Async variant of ``wrap_model_call``."""
        response: ModelCallResult = await handler(request)
        if _is_dead_end(response):
            response = await handler(self._finalize_request(request))
            self._log_recovery(response)
        return response
