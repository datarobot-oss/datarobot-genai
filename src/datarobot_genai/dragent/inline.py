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

"""In-process dragent execution for use as a DRUM alternative in run_agent.py.

Mirrors the entry shape of ``datarobot_drum``'s ``execute_drum_inline`` so the
host script can route between DRUM and dragent with a single env-var-gated
branch. The function always returns a single aggregated OpenAI
``ChatCompletion``; the request's ``stream`` flag is ignored because the
agentic playground only renders the final assistant message.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from pathlib import Path

from openai.types.chat import ChatCompletion
from openai.types.chat.completion_create_params import CompletionCreateParamsBase

logger = logging.getLogger(__name__)

# Sentinel user_id passed to ``SessionManager.session(...)`` so that per-user
# dragent workflows (which raise when ``user_id`` is ``None``) succeed in the
# single-shot inline path. Shared workflows ignore it.
INLINE_USER_ID = "dragent-inline"
WORKFLOW_FILENAME = "workflow.yaml"


def _resolve_config_path(custom_model_dir: Path, config_file: Path | None) -> Path:
    """Resolve the dragent workflow YAML.

    The explicit ``config_file`` argument wins when supplied; otherwise the
    function expects ``<custom_model_dir>/workflow.yaml`` to exist.
    """
    candidate = (
        Path(config_file)
        if config_file is not None
        else (Path(custom_model_dir) / WORKFLOW_FILENAME)
    )
    if not candidate.exists():
        raise FileNotFoundError(
            f"DRAgent workflow config not found at {candidate}. "
            f"Pass config_file=... or place {WORKFLOW_FILENAME} in {custom_model_dir}."
        )
    return candidate


async def execute_dragent_inline_async(
    chat_completion: CompletionCreateParamsBase,
    custom_model_dir: Path,
    *,
    config_file: Path | None = None,
    default_headers: dict[str, str] | None = None,
) -> ChatCompletion:
    """Execute a dragent workflow in-process and return the final OpenAI ``ChatCompletion``.

    The function delegates aggregation to NAT: ``runner.result(to_type=ChatResponse)``
    is the same call the dragent FastAPI route uses for non-streaming
    ``/v1/chat/completions`` requests. NAT workflows already declare how to
    produce a single response — ``per_user_tool_calling_agent`` registers
    ``single_output_type=ChatResponse`` natively, and dragent-native agents
    (``base``, ``langgraph``, ``crewai``, ``llamaindex``) declare
    ``Streaming(convert=aggregate_dragent_event_responses)`` so NAT collapses
    their stream and then chains the registered global converter
    ``DRAgentEventResponse → ChatResponseChunk`` through to ``ChatResponse``.
    The OpenAI ``ChatCompletion`` is then a structural re-validation of NAT's
    OpenAI-compatible ``ChatResponse``.

    Parameters
    ----------
    chat_completion
        OpenAI Chat Completions create-params dict (``stream`` is ignored).
    custom_model_dir
        Directory containing the agent code. ``workflow.yaml`` is loaded from
        here when ``config_file`` is not supplied.
    config_file
        Optional explicit override of the workflow YAML path.
    default_headers
        Optional HTTP headers to inject into the workflow's auth/LLM components
        (forwarded to ``load_workflow``).
    """
    # Local imports keep the optional NAT dependency out of any path that just
    # imports the symbol but never calls it (e.g. when DRUM is selected).
    # The dragent front-end's global type converters are registered by NAT
    # plugin discovery via the ``nat.front_ends`` entry point ``dragent`` when
    # ``load_workflow`` runs.
    from nat.data_models.api_server import ChatResponse

    from datarobot_genai.core.chat.completions import (
        convert_chat_completion_params_to_run_agent_input,
    )
    from datarobot_genai.core.telemetry_bootstrap import setup_dragent_tracing
    from datarobot_genai.dragent.frontends.request import DRAgentRunAgentInput
    from datarobot_genai.nat.helpers import load_workflow

    # Wire tracing before NAT loads the workflow so spans from workflow build
    # and execution reach the OTLP exporter. Idempotent and safely shares
    # state with the FastAPI server path.
    setup_dragent_tracing(service_name="dragent-inline")

    workflow_path = _resolve_config_path(Path(custom_model_dir), config_file)
    logger.info("Running dragent workflow from %s", workflow_path)

    # Build the workflow input as ``DRAgentRunAgentInput`` (a ``RunAgentInput``
    # subclass) so NAT's registered converters (e.g.
    # ``DRAgentRunAgentInput -> ChatRequest`` /
    # ``DRAgentRunAgentInput -> ChatRequestOrMessage``) match by exact type
    # for workflows whose input type isn't ``RunAgentInput``.
    base_input = convert_chat_completion_params_to_run_agent_input(chat_completion)
    run_agent_input = DRAgentRunAgentInput.model_validate(base_input.model_dump())

    async with load_workflow(workflow_path, headers=default_headers) as session_manager:
        async with session_manager.session(user_id=INLINE_USER_ID) as session:
            async with session.run(run_agent_input) as runner:
                response: ChatResponse = await runner.result(to_type=ChatResponse)

    # NAT's ``ChatResponse`` is documented as OpenAI Chat Completions API
    # compatible (same field layout); ``mode="json"`` serialises ``created`` as
    # an int via the field serializer so the resulting dict satisfies OpenAI's
    # stricter typing. Preserve the caller's ``model`` value when the workflow
    # didn't propagate one (NAT defaults to ``"unknown-model"``).
    payload = response.model_dump(mode="json")
    if (requested_model := chat_completion.get("model")) and payload.get("model") in (
        None,
        "unknown-model",
    ):
        payload["model"] = requested_model
    return ChatCompletion.model_validate(payload)


def execute_dragent_inline(
    chat_completion: CompletionCreateParamsBase,
    custom_model_dir: Path,
    *,
    config_file: Path | None = None,
    default_headers: dict[str, str] | None = None,
) -> ChatCompletion:
    """Run :func:`execute_dragent_inline_async` synchronously for ``run_agent.py``.

    Sync entry point used by ``datarobot-user-models``'s ``run_agent.py`` when the
    dragent inline path is enabled. The host process may already have an asyncio
    loop running on the current thread (for example when ``run_agent_procedure``
    is invoked from an ``ipykernel`` driven agentic notebook) without exposing
    that loop to this call site; in that case ``asyncio.run`` cannot be used here
    and the coroutine is executed in a worker thread with its own event loop.
    """
    coro = execute_dragent_inline_async(
        chat_completion=chat_completion,
        custom_model_dir=custom_model_dir,
        config_file=config_file,
        default_headers=default_headers,
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()
