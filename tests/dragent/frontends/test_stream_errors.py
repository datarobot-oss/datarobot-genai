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

import json
from contextlib import asynccontextmanager
from typing import Any

import pytest
from ag_ui.core import EventType
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nat.data_models.api_server import ChatResponseChunk
from nat.front_ends.fastapi import response_helpers
from nat.front_ends.fastapi.routes import common_utils
from nat.front_ends.fastapi.routes import v1_chat_completions

from datarobot_genai.dragent.frontends.response import DRAgentEventResponse
from datarobot_genai.dragent.frontends.stream_errors import patch_stream_error_framing

_ERROR_MESSAGE = "Token was created in a different Context"


@pytest.fixture(autouse=True)
def _restore_stream_symbols(monkeypatch):
    """patch_stream_error_framing rebinds module globals; snapshot them for auto-restore."""
    monkeypatch.setattr(
        common_utils,
        "generate_streaming_response_as_str",
        common_utils.generate_streaming_response_as_str,
    )
    monkeypatch.setattr(
        v1_chat_completions,
        "generate_streaming_response_as_str",
        v1_chat_completions.generate_streaming_response_as_str,
    )


@pytest.mark.asyncio
async def test_workflow_error_during_generate_stream_is_framed_as_run_error(monkeypatch):
    """A /generate/stream workflow error surfaces as a framed AG-UI RUN_ERROR, not
    NAT's bare workflow_error JSON that ``data:``-only clients drop.
    """

    async def failing_stream(payload: Any, **kwargs: Any):
        yield DRAgentEventResponse(events=[])  # one good chunk, then failure
        raise ValueError(_ERROR_MESSAGE)

    monkeypatch.setattr(response_helpers, "generate_streaming_response", failing_stream)
    patch_stream_error_framing()

    chunks = [
        chunk
        async for chunk in common_utils.generate_streaming_response_as_str(
            {"messages": []},
            session=None,
            streaming=True,
            result_type=None,
            output_type=DRAgentEventResponse,
        )
    ]

    # Every emitted line is SSE-framed; no bare JSON the data:-only parser would drop.
    assert chunks, "expected at least one streamed chunk"
    assert all(chunk.startswith("data:") for chunk in chunks), chunks

    # The terminal chunk is a framed AG-UI RUN_ERROR carrying the exception message.
    terminal = DRAgentEventResponse.model_validate_json(chunks[-1][len("data: ") :])
    assert len(terminal.events) == 1
    error_event = terminal.events[0]
    assert error_event.type == EventType.RUN_ERROR
    assert error_event.message == _ERROR_MESSAGE
    assert error_event.code == "STREAM_ERROR"


def test_patch_stream_error_framing_is_idempotent():
    """A second install must not re-wrap the helper (e.g. a repeated add_routes call)."""
    patch_stream_error_framing()
    after_first = common_utils.generate_streaming_response_as_str
    patch_stream_error_framing()
    after_second = common_utils.generate_streaming_response_as_str

    assert after_first is after_second


@pytest.mark.asyncio
async def test_non_data_success_frames_pass_through_untouched(monkeypatch):
    """Only NAT's bare Error JSON is reframed; other SSE frames (``intermediate_data:``,
    ``observability_trace:``) must pass through unchanged, not be mistaken for errors.
    """
    frames = [
        'intermediate_data: {"foo": 1}\n\n',
        'observability_trace: {"bar": 2}\n\n',
        'data: {"events": []}\n\n',
    ]

    async def fake_as_str(*args: Any, **kwargs: Any):
        for frame in frames:
            yield frame

    monkeypatch.setattr(common_utils, "generate_streaming_response_as_str", fake_as_str)
    patch_stream_error_framing()

    chunks = [
        chunk
        async for chunk in common_utils.generate_streaming_response_as_str(
            {"messages": []}, session=None, streaming=True, result_type=None, output_type=None
        )
    ]

    assert chunks == frames


@pytest.mark.asyncio
async def test_raw_json_non_error_chunk_is_not_reframed(monkeypatch):
    """Detection is a positive parse of NAT's ``Error`` model, not a framing heuristic: a raw-JSON
    chunk that is not an ``Error`` (extra fields, forbidden by the model) passes through untouched.
    """
    passthrough = '{"id": "x", "object": "chat.completion.chunk"}'

    async def fake_as_str(*args: Any, **kwargs: Any):
        yield passthrough

    monkeypatch.setattr(common_utils, "generate_streaming_response_as_str", fake_as_str)
    patch_stream_error_framing()

    chunks = [
        chunk
        async for chunk in common_utils.generate_streaming_response_as_str(
            {"messages": []}, session=None, streaming=True, result_type=None, output_type=None
        )
    ]

    assert chunks == [passthrough]


@pytest.mark.asyncio
async def test_chat_completions_stream_error_is_openai_shaped(monkeypatch):
    """On /chat/completions the error must be an OpenAI-shaped ``data: {"error":...}`` frame,
    not an AG-UI RUN_ERROR (which is foreign to the OpenAI chunk contract).
    """

    async def failing_stream(payload: Any, **kwargs: Any):
        yield DRAgentEventResponse(events=[])
        raise ValueError(_ERROR_MESSAGE)

    monkeypatch.setattr(response_helpers, "generate_streaming_response", failing_stream)
    patch_stream_error_framing()

    chunks = [
        chunk
        async for chunk in v1_chat_completions.generate_streaming_response_as_str(
            {"messages": []},
            session=None,
            streaming=True,
            result_type=ChatResponseChunk,
            output_type=ChatResponseChunk,
        )
    ]

    assert chunks, "expected at least one streamed chunk"
    assert all(chunk.startswith("data:") for chunk in chunks), chunks
    error = json.loads(chunks[-1][len("data: ") :])
    assert error["error"]["message"] == _ERROR_MESSAGE
    assert error["error"]["type"] == "workflow_error"
    assert "RUN_ERROR" not in chunks[-1]


@pytest.mark.asyncio
async def test_error_frame_follows_output_type_not_module(monkeypatch):
    """Framing keys on output_type, not the NAT module: the common_utils helper (which also backs
    the OpenAI-shaped /chat/stream) must emit an OpenAI error frame when output_type is
    ChatResponseChunk, not an AG-UI RUN_ERROR.
    """

    async def failing_stream(payload: Any, **kwargs: Any):
        yield DRAgentEventResponse(events=[])
        raise ValueError(_ERROR_MESSAGE)

    monkeypatch.setattr(response_helpers, "generate_streaming_response", failing_stream)
    patch_stream_error_framing()

    chunks = [
        chunk
        async for chunk in common_utils.generate_streaming_response_as_str(
            {"messages": []},
            session=None,
            streaming=True,
            result_type=ChatResponseChunk,
            output_type=ChatResponseChunk,
        )
    ]

    error = json.loads(chunks[-1][len("data: ") :])
    assert error["error"]["type"] == "workflow_error"
    assert "RUN_ERROR" not in chunks[-1]


def test_generate_stream_http_endpoint_emits_framed_run_error(monkeypatch):
    """End-to-end /generate/stream over a real TestClient (only the workflow faked):
    a workflow error streams as a framed AG-UI RUN_ERROR the e2e parser accepts.
    """

    async def failing_stream(payload: Any, **kwargs: Any):
        yield DRAgentEventResponse(events=[])
        raise ValueError(_ERROR_MESSAGE)

    monkeypatch.setattr(response_helpers, "generate_streaming_response", failing_stream)
    patch_stream_error_framing()

    @asynccontextmanager
    async def fake_session(**kwargs):
        yield object()

    class FakeWorker:
        _http_flow_handler = None

        def get_step_adaptor(self):
            return None

    class FakeSessionManager:
        def session(self, **kwargs):
            return fake_session()

    handler = common_utils.post_streaming_endpoint(
        worker=FakeWorker(),
        session_manager=FakeSessionManager(),
        request_type=dict,
        enable_interactive=False,
        streaming=True,
        result_type=None,
        output_type=DRAgentEventResponse,
    )
    app = FastAPI()
    app.add_api_route("/generate/stream", handler, methods=["POST"])

    with TestClient(app) as client:
        response = client.post("/generate/stream", json={"messages": []})

    body = response.text
    assert response.status_code == 200
    assert "workflow_error" not in body  # NAT's bare error must not leak
    assert '"type":"RUN_ERROR"' in body
    assert _ERROR_MESSAGE in body
    # Every SSE payload line is framed AG-UI (parseable into DRAgentEventResponse).
    for line in body.splitlines():
        if line.startswith("data: "):
            DRAgentEventResponse.model_validate_json(line[len("data: ") :])
