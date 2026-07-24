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

"""Frame bare NAT stream errors as route-specific terminal SSE events."""

import json
from collections.abc import AsyncIterator
from typing import Any

from ag_ui.core import RunErrorEvent
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import Error
from pydantic import ValidationError

from datarobot_genai.core.agents import default_usage_metrics

from .response import DRAgentEventResponse

_FRAMING_APPLIED_ATTR = "_dragent_stream_error_framing"


def _nat_error(chunk: str) -> Error | None:
    """Return NAT's bare unframed ``Error`` payload, if present."""
    if not chunk.startswith("{"):
        return None
    try:
        return Error.model_validate_json(chunk)
    except ValidationError:
        return None


def _agui_error_frame(message: str) -> str:
    """AG-UI ``RUN_ERROR`` SSE frame (``/generate/stream``)."""
    response = DRAgentEventResponse(
        events=[RunErrorEvent(message=message, code="STREAM_ERROR")],
        usage_metrics=default_usage_metrics(),
    )
    return response.get_stream_data()


def _openai_error_frame(message: str) -> str:
    """OpenAI-shaped error SSE frame (``/chat/completions``)."""
    error = {"error": {"message": message, "type": "workflow_error", "code": None}}
    return f"data: {json.dumps(error)}\n\n"


def _framed(original: Any) -> Any:
    """Wrap a NAT streaming helper with route-specific error framing."""

    async def generate_streaming_response_as_str(*args: Any, **kwargs: Any) -> AsyncIterator[str]:
        openai = kwargs.get("output_type") is ChatResponseChunk
        error_frame = _openai_error_frame if openai else _agui_error_frame
        async for chunk in original(*args, **kwargs):
            error = _nat_error(chunk)
            yield error_frame(error.message) if error else chunk

    setattr(generate_streaming_response_as_str, _FRAMING_APPLIED_ATTR, True)
    return generate_streaming_response_as_str


def patch_stream_error_framing() -> None:
    """Patch NAT stream helpers used by ``/generate/stream`` and ``/chat/completions``."""
    from nat.front_ends.fastapi.routes import common_utils
    from nat.front_ends.fastapi.routes import v1_chat_completions

    for module in (common_utils, v1_chat_completions):
        helper = module.generate_streaming_response_as_str
        if not getattr(helper, _FRAMING_APPLIED_ATTR, False):
            module.generate_streaming_response_as_str = _framed(helper)
