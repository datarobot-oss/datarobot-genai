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

from datarobot.models.genai.agent.auth import get_authorization_context
from datarobot.models.genai.agent.auth import set_authorization_context
from datarobot_drum import RuntimeParameters
from openai.types.chat import CompletionCreateParams
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming
from openai.types.chat.completion_create_params import CompletionCreateParamsStreaming

from datarobot_genai.core.chat import CustomModelChatResponse
from datarobot_genai.core.chat import CustomModelStreamingResponse
from datarobot_genai.core.chat import initialize_authorization_context
from datarobot_genai.core.chat import to_custom_model_chat_response
from datarobot_genai.core.chat import to_custom_model_streaming_response
from datarobot_genai.core.telemetry_agent import instrument

logger = logging.getLogger(__name__)


def maybe_set_env_from_runtime_parameters(key: str) -> None:
    """Set an environment variable from a DRUM Runtime Parameter if it exists.

    This is safe to call outside of the DataRobot runtime. If the parameter is not available,
    the function does nothing.
    """
    runtime_parameter_placeholder_value = "SET_VIA_PULUMI_OR_MANUALLY"
    try:
        runtime_parameter_value = RuntimeParameters.get(key)
        if (
            runtime_parameter_value
            and len(runtime_parameter_value) > 0
            and runtime_parameter_value != runtime_parameter_placeholder_value
        ):
            os.environ[key] = runtime_parameter_value
    except ValueError:
        # Local dev: runtime parameters may be unavailable
        pass


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
    runtime_parameter_keys: list[str] | None = None,
    framework: Literal["crewai", "langgraph", "llamaindex"] | None = None,
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
    runtime_parameter_keys : Optional[List[str]]
        Runtime parameter keys (DataRobot custom model) to propagate into env. When
        ``None``, defaults to
        ``['EXTERNAL_MCP_URL', 'MCP_DEPLOYMENT_ID', 'SESSION_SECRET_KEY']``.
    framework : Optional[Literal["crewai", "langgraph", "llamaindex"]]
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

    # Load MCP runtime parameters and session secret if configured
    if runtime_parameter_keys is None:
        runtime_parameter_keys = ["EXTERNAL_MCP_URL", "MCP_DEPLOYMENT_ID", "SESSION_SECRET_KEY"]
    for key in runtime_parameter_keys:
        maybe_set_env_from_runtime_parameters(key)

    # Initialize the authorization context for downstream agents/tools
    initialize_authorization_context(completion_create_params, **kwargs)

    # Capture auth context out of the main thread to propagate into worker
    try:
        auth_context = get_authorization_context()
    except LookupError:
        auth_context = {}

    # Instantiate user agent with all supplied completion params
    agent = agent_cls(**completion_create_params)

    # Run agent within the worker loop, attaching auth context in that thread
    def invoke_with_auth_context() -> Any:
        try:
            set_authorization_context(auth_context)
        except AttributeError:
            pass
        return event_loop.run_until_complete(
            agent.invoke(completion_create_params=completion_create_params)
        )

    result = thread_pool_executor.submit(invoke_with_auth_context).result()

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
