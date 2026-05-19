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

Mirrors the contract of ``datarobot_drum``'s ``execute_drum_inline`` so callers
can route between DRUM and dragent with a single env-var-gated branch.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import uuid
from pathlib import Path
from typing import Any

from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.completion_create_params import CompletionCreateParamsBase

logger = logging.getLogger(__name__)

# Sentinel user_id passed to ``SessionManager.session(...)`` so that per-user
# dragent workflows (which raise when ``user_id`` is ``None``) succeed in the
# single-shot inline path. Shared workflows ignore it. Surfacing a real
# per-user identity from ``chat_completion["authorization_context"]`` is a
# follow-up; see open follow-ups in the implementation plan.
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
) -> ChatCompletion | list[ChatCompletionChunk]:
    """Execute a dragent workflow in-process for one chat completion request.

    Parameters
    ----------
    chat_completion
        OpenAI Chat Completions create-params dict. ``stream`` controls the return
        shape but the workflow is always invoked in streaming mode internally and
        the result reshaped accordingly.
    custom_model_dir
        Directory containing the agent code. ``workflow.yaml`` is loaded from here
        when ``config_file`` is not supplied.
    config_file
        Optional explicit override of the workflow YAML path.
    default_headers
        Optional HTTP headers to inject into the workflow's auth/LLM components
        (forwarded to ``load_workflow``).

    Returns
    -------
    ChatCompletion
        When ``chat_completion["stream"]`` is falsy.
    list[ChatCompletionChunk]
        When ``chat_completion["stream"]`` is truthy.
    """
    # Local imports keep the optional NAT dependency out of any path that just
    # imports the symbol but never calls it (e.g. when DRUM is selected).
    from nat.data_models.api_server import ChatResponseChunk

    from datarobot_genai.core.chat.completions import (
        convert_chat_completion_params_to_run_agent_input,
    )

    # Side-effect import: registers global type converters
    # (DRAgentEventResponse -> ChatResponseChunk and related). Required for
    # ``runner.result_stream(to_type=ChatResponseChunk)`` to find a converter.
    from datarobot_genai.dragent.frontends import register as _register  # noqa: F401
    from datarobot_genai.nat.helpers import load_workflow

    workflow_path = _resolve_config_path(Path(custom_model_dir), config_file)
    logger.info("Running dragent workflow from %s", workflow_path)

    # Convert OpenAI Chat Completion params directly to AG-UI RunAgentInput.
    # Going through NAT's ChatRequest would reject role="tool" messages; the
    # AG-UI converter handles those correctly.
    run_agent_input = convert_chat_completion_params_to_run_agent_input(chat_completion)

    chunks: list[ChatResponseChunk] = []
    async with load_workflow(workflow_path, headers=default_headers) as session_manager:
        async with session_manager.session(user_id=INLINE_USER_ID) as session:
            async with session.run(run_agent_input) as runner:
                async for chunk in runner.result_stream(to_type=ChatResponseChunk):
                    chunks.append(chunk)

    # NAT's ChatResponseChunk is OpenAI-compatible; ``mode="json"`` serialises
    # ``created`` as an int (via the field serializer) so the resulting dict
    # passes OpenAI's stricter typing.
    openai_chunks = [ChatCompletionChunk.model_validate(c.model_dump(mode="json")) for c in chunks]

    if chat_completion.get("stream"):
        return openai_chunks

    return _aggregate_chunks_into_completion(
        openai_chunks,
        requested_model=chat_completion.get("model"),
    )


def execute_dragent_inline(
    chat_completion: CompletionCreateParamsBase,
    custom_model_dir: Path,
    *,
    config_file: Path | None = None,
    default_headers: dict[str, str] | None = None,
) -> ChatCompletion | list[ChatCompletionChunk]:
    """Run :func:`execute_dragent_inline_async` synchronously via ``asyncio.run``.

    Convenient for use from sync entry points such as
    ``datarobot-user-models``'s ``run_agent.py``.
    """
    return asyncio.run(
        execute_dragent_inline_async(
            chat_completion=chat_completion,
            custom_model_dir=custom_model_dir,
            config_file=config_file,
            default_headers=default_headers,
        )
    )


def _aggregate_chunks_into_completion(
    chunks: list[ChatCompletionChunk],
    *,
    requested_model: str | None,
) -> ChatCompletion:
    """Collapse a streaming chunk list into a single ``ChatCompletion``.

    Mirrors OpenAI's documented chunk-concatenation behavior: per-index
    ``delta.content`` strings are concatenated, per-index ``delta.tool_calls``
    are merged by ``index`` (id/name from the first non-null occurrence,
    ``function.arguments`` concatenated across chunks), and the last non-null
    ``finish_reason`` per index wins (defaulting to ``"stop"`` or
    ``"tool_calls"`` when only tool calls were emitted). ``usage`` and
    ``system_fingerprint`` come from the last chunk that supplies them.
    """
    if not chunks:
        return ChatCompletion(
            id=uuid.uuid4().hex,
            object="chat.completion",
            created=int(datetime.datetime.now(datetime.UTC).timestamp()),
            model=requested_model or "unknown-model",
            choices=[],
        )

    completion_id = chunks[0].id
    created = chunks[0].created
    model = requested_model or chunks[0].model
    system_fingerprint: str | None = None
    usage_payload: dict[str, Any] | None = None

    # Per-choice-index aggregator: collects text parts, tool-call shards and
    # the latest finish_reason.
    accumulators: dict[int, dict[str, Any]] = {}

    for chunk in chunks:
        if chunk.system_fingerprint:
            system_fingerprint = chunk.system_fingerprint
        if chunk.usage:
            usage_payload = chunk.usage.model_dump()

        for choice in chunk.choices:
            idx = choice.index
            acc = accumulators.setdefault(
                idx,
                {
                    "content_parts": [],
                    "tool_calls": {},
                    "finish_reason": None,
                },
            )
            delta = choice.delta
            if delta.content:
                acc["content_parts"].append(delta.content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    tc_acc = acc["tool_calls"].setdefault(
                        tc.index,
                        {
                            "id": None,
                            "type": "function",
                            "name": None,
                            "args_parts": [],
                        },
                    )
                    if tc.id:
                        tc_acc["id"] = tc.id
                    if tc.type:
                        tc_acc["type"] = tc.type
                    if tc.function is not None:
                        if tc.function.name:
                            tc_acc["name"] = tc.function.name
                        if tc.function.arguments:
                            tc_acc["args_parts"].append(tc.function.arguments)
            if choice.finish_reason:
                acc["finish_reason"] = choice.finish_reason

    openai_choices: list[dict[str, Any]] = []
    for idx in sorted(accumulators.keys()):
        acc = accumulators[idx]
        tool_calls_payload: list[dict[str, Any]] = []
        for tc_idx in sorted(acc["tool_calls"].keys()):
            tc = acc["tool_calls"][tc_idx]
            if tc["name"] is None and not tc["args_parts"]:
                # Drop empty placeholders so the payload validates as OpenAI.
                continue
            tool_calls_payload.append(
                {
                    "id": tc["id"] or f"call_{uuid.uuid4().hex[:24]}",
                    "type": tc["type"] or "function",
                    "function": {
                        "name": tc["name"] or "",
                        "arguments": "".join(tc["args_parts"]),
                    },
                }
            )

        joined_content = "".join(acc["content_parts"])
        message: dict[str, Any] = {
            "role": "assistant",
            "content": joined_content if joined_content else None,
        }
        if tool_calls_payload:
            message["tool_calls"] = tool_calls_payload

        finish_reason = acc["finish_reason"] or ("tool_calls" if tool_calls_payload else "stop")
        openai_choices.append(
            {
                "index": idx,
                "finish_reason": finish_reason,
                "message": message,
            }
        )

    payload: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": openai_choices,
    }
    if system_fingerprint:
        payload["system_fingerprint"] = system_fingerprint
    if usage_payload:
        payload["usage"] = usage_payload

    return ChatCompletion.model_validate(payload)
