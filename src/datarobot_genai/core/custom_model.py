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
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers to implement DataRobot Custom Model chat entrypoints."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from typing import Literal

from openai.types.chat import CompletionCreateParams
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming
from openai.types.chat.completion_create_params import CompletionCreateParamsStreaming

from datarobot_genai.core.chat import CustomModelChatResponse
from datarobot_genai.core.chat import CustomModelStreamingResponse
from datarobot_genai.core.chat import to_custom_model_chat_response
from datarobot_genai.core.chat import to_custom_model_streaming_response
from datarobot_genai.core.chat.auth import resolve_authorization_context
from datarobot_genai.core.telemetry_agent import instrument

logger = logging.getLogger(__name__)


def load_model() -> tuple[ThreadPoolExecutor, asyncio.AbstractEventLoop]:
    """Initialize a dedicated event loop within a worker thread.

    Returns
    -------
    (ThreadPoolExecutor, asyncio.AbstractEventLoop)
        A single-worker executor and the associated event loop.
    """
    thread_pool_executor = ThreadPoolExecutor(1)
    event_loop = asyncio.new_event_loop()
    thread_pool_executor.submit(asyncio.set_event_loop, event_loop).result()
    return (thread_pool_executor, event_loop)


def chat_entrypoint(
    agent_cls: type[Any],
    completion_create_params: CompletionCreateParams
    | CompletionCreateParamsNonStreaming
    | CompletionCreateParamsStreaming,
    load_model_result: tuple[ThreadPoolExecutor, asyncio.AbstractEventLoop],
    *,
    work_dir: str | None = None,
    framework: Literal["crewai", "langgraph", "llamaindex", "nat"] | None = None,
    **kwargs: Any,
) -> CustomModelChatResponse | Iterator[CustomModelStreamingResponse]:
    """Run a generic Custom Model chat entrypoint for agent-based implementations.

    Parameters
    ----------
    agent_cls : Type[Any]
        The agent class to instantiate. Must define an ``async invoke(...)`` method
        returning either:
          - a tuple (response_text, pipeline_interactions, usage_metrics)
          - or an async generator yielding (delta_text, pipeline_interactions, usage_metrics)
    completion_create_params : CompletionCreateParams | ...
        Parameters supplied by OpenAI-compatible Chat API.
    load_model_result : tuple[ThreadPoolExecutor, asyncio.AbstractEventLoop]
        Values returned by :func:`load_model`.
    work_dir : Optional[str]
        Working directory to ``chdir`` into before invoking the agent. This is useful
        when relative paths are used in agent templates.
    framework : Optional[Literal["crewai", "langgraph", "llamaindex", "nat"]]
        When provided, idempotently instruments HTTP clients, OpenAI SDK, and the
        given framework. If omitted, general instrumentation is still applied.
    **kwargs : Any
        Extra values forwarded for header-based auth context extraction.
    """
    thread_pool_executor, event_loop = load_model_result

    # Set up telemetry (idempotent). When framework is provided, instrument it as well.
    try:
        instrument(framework)
    except Exception:
        # Instrumentation is best-effort; proceed regardless
        pass

    # Optionally change working directory for frameworks which rely on relative paths
    if work_dir:
        try:
            os.chdir(work_dir)
        except Exception as e:
            logger.warning(f"Failed to change working directory to {work_dir}: {e}")

    # Retrieve authorization context using all supported methods for downstream agents/tools
    completion_create_params["authorization_context"] = resolve_authorization_context(
        completion_create_params, **kwargs
    )
    # Keep only allowed headers from the forwarded_headers.
    incoming_headers = kwargs.get("headers", {}) or {}
    allowed_headers = {"x-datarobot-api-token", "x-datarobot-api-key"}
    forwarded_headers = {k: v for k, v in incoming_headers.items() if k.lower() in allowed_headers}
    completion_create_params["forwarded_headers"] = forwarded_headers

    # Instantiate user agent with all supplied completion params including auth context
    agent = agent_cls(**completion_create_params)

    # Invoke the agent and check if it returns a generator or a tuple
    result = thread_pool_executor.submit(
        event_loop.run_until_complete,
        agent.invoke(completion_create_params=completion_create_params),
    ).result()

    # Streaming response (async generator)
    if isinstance(result, AsyncGenerator):
        return to_custom_model_streaming_response(
            thread_pool_executor,
            event_loop,
            result,
            model=completion_create_params.get("model"),
        )

    # Non-streaming response
    response_text, pipeline_interactions, usage_metrics = result
    return to_custom_model_chat_response(
        response_text,
        pipeline_interactions,
        usage_metrics,
        model=completion_create_params.get("model"),
    )
