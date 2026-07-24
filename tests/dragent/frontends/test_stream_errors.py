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
    """Restore patched NAT stream helpers after each test."""
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
    """/generate/stream errors become framed AG-UI ``RUN_ERROR`` chunks."""

    async def failing_stream(payload: Any, **kwargs: Any):
        yield DRAgentEventResponse(events=[])  # good chunk, then failure
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

    assert chunks, "expected at least one streamed chunk"
    assert all(chunk.startswith("data:") for chunk in chunks), chunks

    terminal = DRAgentEventResponse.model_validate_json(chunks[-1][len("data: ") :])
    assert len(terminal.events) == 1
    error_event = terminal.events[0]
    assert error_event.type == EventType.RUN_ERROR
    assert error_event.message == _ERROR_MESSAGE
    assert error_event.code == "STREAM_ERROR"


def test_patch_stream_error_framing_is_idempotent():
    """Repeated patching does not wrap twice."""
    patch_stream_error_framing()
    after_first = common_utils.generate_streaming_response_as_str
    patch_stream_error_framing()
    after_second = common_utils.generate_streaming_response_as_str

    assert after_first is after_second


@pytest.mark.asyncio
async def test_non_data_success_frames_pass_through_untouched(monkeypatch):
    """Framed non-data SSE chunks pass through unchanged."""
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
    """Raw JSON that is not NAT ``Error`` passes through unchanged."""
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
    """/chat/completions errors use the OpenAI error shape."""

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
    """Error shape follows ``output_type``, not the NAT module."""

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
    """/generate/stream emits parseable AG-UI ``RUN_ERROR`` SSE."""

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
    assert "workflow_error" not in body  # no bare NAT error
    assert '"type":"RUN_ERROR"' in body
    assert _ERROR_MESSAGE in body
    # Data frames remain parseable AG-UI.
    for line in body.splitlines():
        if line.startswith("data: "):
            DRAgentEventResponse.model_validate_json(line[len("data: ") :])
