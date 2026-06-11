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

import logging
from typing import Any
from unittest.mock import Mock

import pytest
from langchain.agents.middleware import ExtendedModelResponse
from langchain.agents.middleware import ModelRequest
from langchain.agents.middleware import ModelResponse
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

from datarobot_genai.langgraph.middleware import FinalAnswerMiddleware
from datarobot_genai.langgraph.middleware import _is_dead_end

DEAD_END_EMPTY = AIMessage(
    content="", additional_kwargs={"reasoning_content": "We must use the browse tool..."}
)
DEAD_END_THINKING = AIMessage(
    content=[{"type": "thinking", "thinking": "Let's search for sources."}]
)
REAL_ANSWER = AIMessage(content="THE FINAL ANSWER")
TOOL_CALL = AIMessage(
    content="", tool_calls=[{"name": "word_counter", "args": {"text": "a b"}, "id": "tc1"}]
)


def result_of(response: Any) -> list[Any]:
    """Narrow a ModelCallResult to the ModelResponse result list."""
    assert isinstance(response, ModelResponse)
    return response.result


def make_request(*, tools: list[Any] | None = None, tool_choice: Any = None) -> ModelRequest[Any]:
    return ModelRequest(
        model=Mock(),
        messages=[HumanMessage(content="think about ai")],
        tools=tools or [],
        tool_choice=tool_choice,
    )


def make_handler(script: list[AIMessage]) -> tuple[Any, list[ModelRequest[Any]]]:
    """Scripted sync handler; records every request it receives."""
    seen: list[ModelRequest[Any]] = []

    def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.append(request)
        return ModelResponse(result=[script[min(len(seen) - 1, len(script) - 1)]])

    return handler, seen


def make_async_handler(script: list[AIMessage]) -> tuple[Any, list[ModelRequest[Any]]]:
    seen: list[ModelRequest[Any]] = []

    async def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.append(request)
        return ModelResponse(result=[script[min(len(seen) - 1, len(script) - 1)]])

    return handler, seen


@pytest.mark.parametrize(
    "response, expected",
    [
        # Empty content with reasoning hoisted to additional_kwargs: dead end.
        (ModelResponse(result=[DEAD_END_EMPTY]), True),
        # Thinking-only content list (ChatLiteLLM injection shape): dead end.
        (ModelResponse(result=[DEAD_END_THINKING]), True),
        # Real answer text: not a dead end.
        (ModelResponse(result=[REAL_ANSWER]), False),
        # Tool-calling turn: not a dead end (the loop continues to tools).
        (ModelResponse(result=[TOOL_CALL]), False),
        # Bare AIMessage return shape is handled.
        (DEAD_END_EMPTY, True),
        (REAL_ANSWER, False),
        # ExtendedModelResponse wrapping shape is handled.
        (ExtendedModelResponse(model_response=ModelResponse(result=[DEAD_END_EMPTY])), True),
        # No AIMessage at all: nothing to recover.
        (ModelResponse(result=[]), False),
    ],
)
def test_is_dead_end(response: Any, expected: bool) -> None:
    assert _is_dead_end(response) is expected


def test_dead_end_retries_once_with_nudge_and_returns_answer() -> None:
    handler, seen = make_handler([DEAD_END_EMPTY, REAL_ANSWER])
    request = make_request()

    response = FinalAnswerMiddleware().wrap_model_call(request, handler)

    assert len(seen) == 2
    assert result_of(response) == [REAL_ANSWER]
    retry = seen[1]
    assert isinstance(retry.messages[-1], HumanMessage)
    assert "final answer" in retry.messages[-1].content
    # The original request is not mutated (override returns a new request).
    assert len(request.messages) == 1


def test_text_answer_does_not_retry() -> None:
    handler, seen = make_handler([REAL_ANSWER])

    response = FinalAnswerMiddleware().wrap_model_call(make_request(), handler)

    assert len(seen) == 1
    assert result_of(response) == [REAL_ANSWER]


def test_tool_call_turn_does_not_retry() -> None:
    handler, seen = make_handler([TOOL_CALL])

    response = FinalAnswerMiddleware().wrap_model_call(make_request(), handler)

    assert len(seen) == 1
    assert result_of(response) == [TOOL_CALL]


def test_retry_with_tools_bound_sets_tool_choice_none() -> None:
    handler, seen = make_handler([DEAD_END_EMPTY, REAL_ANSWER])
    request = make_request(tools=[Mock()], tool_choice="auto")

    FinalAnswerMiddleware().wrap_model_call(request, handler)

    assert seen[1].tool_choice == "none"


def test_retry_without_tools_preserves_tool_choice() -> None:
    handler, seen = make_handler([DEAD_END_EMPTY, REAL_ANSWER])
    request = make_request(tools=[], tool_choice=None)

    FinalAnswerMiddleware().wrap_model_call(request, handler)

    assert seen[1].tool_choice is None


def test_double_dead_end_returns_last_response_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler, seen = make_handler([DEAD_END_EMPTY, DEAD_END_THINKING])

    with caplog.at_level(logging.WARNING, logger="datarobot_genai.langgraph.middleware"):
        response = FinalAnswerMiddleware().wrap_model_call(make_request(), handler)

    assert len(seen) == 2  # exactly one recovery attempt
    assert result_of(response) == [DEAD_END_THINKING]
    assert any("finalize nudge" in record.message for record in caplog.records)


def test_custom_nudge_is_used() -> None:
    handler, seen = make_handler([DEAD_END_EMPTY, REAL_ANSWER])

    FinalAnswerMiddleware(nudge="Answer now.").wrap_model_call(make_request(), handler)

    assert seen[1].messages[-1].content == "Answer now."


async def test_async_dead_end_recovers() -> None:
    handler, seen = make_async_handler([DEAD_END_EMPTY, REAL_ANSWER])

    response = await FinalAnswerMiddleware().awrap_model_call(make_request(), handler)

    assert len(seen) == 2
    assert result_of(response) == [REAL_ANSWER]
    assert "final answer" in seen[1].messages[-1].content


async def test_async_text_answer_does_not_retry() -> None:
    handler, seen = make_async_handler([REAL_ANSWER])

    response = await FinalAnswerMiddleware().awrap_model_call(make_request(), handler)

    assert len(seen) == 1
    assert result_of(response) == [REAL_ANSWER]
