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

from __future__ import annotations

import asyncio
import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from nat.builder.context import Context
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import ChatResponseChoice
from nat.data_models.api_server import ChoiceMessage
from nat.data_models.api_server import Usage
from openai.types.chat import ChatCompletion
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan
from opentelemetry.trace import SpanContext
from opentelemetry.trace import TraceFlags

from datarobot_genai.dragent import execute_dragent_inline_async
from datarobot_genai.dragent.inline import _resolve_config_path
from datarobot_genai.dragent.inline import execute_dragent_inline

# --- Test helpers --------------------------------------------------------


def _make_chat_response(
    *,
    content: str = "Hello world.",
    finish_reason: str = "stop",
    model: str = "datarobot-e2e",
    id_: str = "cmpl-1",
) -> ChatResponse:
    return ChatResponse(
        id=id_,
        model=model,
        created=datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC),
        choices=[
            ChatResponseChoice(
                index=0,
                message=ChoiceMessage(role="assistant", content=content),
                finish_reason=finish_reason,  # type: ignore[arg-type]
            )
        ],
        usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )


def _build_fake_load_workflow(response: ChatResponse):
    """Build an async context manager that mimics load_workflow -> session -> runner."""

    class _FakeRunner:
        async def __aenter__(self):  # noqa: D401 - test stub
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: D401 - test stub
            return False

        async def result(self, to_type=None):  # noqa: D401 - test stub
            return response

    class _FakeSession:
        def run(self, _input):
            return _FakeRunner()

    class _FakeSessionManager:
        @asynccontextmanager
        async def session(self, user_id=None):  # noqa: D401 - test stub
            yield _FakeSession()

    @asynccontextmanager
    async def _fake_load_workflow(_path):
        yield _FakeSessionManager()

    return _fake_load_workflow


def _install_fake_workflow(monkeypatch: pytest.MonkeyPatch, response: ChatResponse) -> None:
    """Patch ``load_workflow`` and bypass disk lookup of workflow.yaml."""
    monkeypatch.setattr(
        "datarobot_genai.dragent.workflow.load_workflow",
        _build_fake_load_workflow(response),
    )
    monkeypatch.setattr(
        "datarobot_genai.dragent.inline._resolve_config_path",
        lambda custom_model_dir, config_file: Path("/tmp/fake-workflow.yaml"),
    )


def _build_trace_id_recording_load_workflow(response: ChatResponse, observed: dict):
    """Fake load_workflow whose runner records the NAT ``workflow_trace_id``.

    Mirrors what NAT's real runner reads: the ``workflow_trace_id`` that is
    active on the singleton context state at ``runner.result()`` time.
    """

    class _FakeRunner:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def result(self, to_type=None):
            observed["workflow_trace_id"] = Context.get().workflow_trace_id
            return response

    class _FakeSession:
        def run(self, _input):
            return _FakeRunner()

    class _FakeSessionManager:
        @asynccontextmanager
        async def session(self, user_id=None):
            yield _FakeSession()

    @asynccontextmanager
    async def _fake_load_workflow(_path):
        yield _FakeSessionManager()

    return _fake_load_workflow


def _install_trace_id_recording_workflow(
    monkeypatch: pytest.MonkeyPatch, response: ChatResponse, observed: dict
) -> None:
    monkeypatch.setattr(
        "datarobot_genai.dragent.workflow.load_workflow",
        _build_trace_id_recording_load_workflow(response, observed),
    )
    monkeypatch.setattr(
        "datarobot_genai.dragent.inline._resolve_config_path",
        lambda custom_model_dir, config_file: Path("/tmp/fake-workflow.yaml"),
    )


# --- Public API ----------------------------------------------------------


async def test_execute_dragent_inline_returns_chat_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GIVEN: a dragent workflow whose aggregated single output is a ChatResponse
    # (this is the contract NAT's runner.result(to_type=ChatResponse) provides
    # for both nat-native and dragent-native workflows).
    _install_fake_workflow(monkeypatch, _make_chat_response(content="Hello world."))

    # WHEN: execute_dragent_inline_async is called
    result = await execute_dragent_inline_async(
        chat_completion={
            "model": "datarobot-e2e",
            "messages": [{"role": "user", "content": "hi"}],
        },  # type: ignore[arg-type]
        custom_model_dir=Path("/tmp"),
    )

    # THEN: NAT's ChatResponse is re-validated as an OpenAI ChatCompletion
    assert isinstance(result, ChatCompletion)
    assert result.model == "datarobot-e2e"
    assert result.object == "chat.completion"
    assert len(result.choices) == 1
    assert result.choices[0].message.content == "Hello world."
    assert result.choices[0].finish_reason == "stop"


async def test_execute_dragent_inline_ignores_stream_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GIVEN: a workflow that emits a ChatResponse, requested with stream=True
    _install_fake_workflow(monkeypatch, _make_chat_response(content="Hi"))

    # WHEN: execute_dragent_inline_async is called with stream=True
    result = await execute_dragent_inline_async(
        chat_completion={
            "model": "datarobot-e2e",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },  # type: ignore[arg-type]
        custom_model_dir=Path("/tmp"),
    )

    # THEN: stream=True is ignored — a single aggregated ChatCompletion is returned
    # (the agentic playground does not render per-token streaming).
    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content == "Hi"


async def test_execute_dragent_inline_reports_configured_model_when_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GIVEN: a workflow that returns "unknown-model" (NAT's default when the
    # underlying workflow doesn't set one). The agent ignores the request's model
    # and runs its configured LLM, so the response must report that configured model.
    _install_fake_workflow(monkeypatch, _make_chat_response(content="ok", model="unknown-model"))

    # WHEN: execute_dragent_inline_async is called (the request model is irrelevant)
    with patch(
        "datarobot_genai.core.config.default_response_model",
        return_value="datarobot/anthropic/claude-sonnet-4-20250514",
    ):
        result = await execute_dragent_inline_async(
            chat_completion={
                "model": "ignored-by-agent",
                "messages": [{"role": "user", "content": "hi"}],
            },  # type: ignore[arg-type]
            custom_model_dir=Path("/tmp"),
        )

    # THEN: the configured LLM model is reported (not the request, not "unknown-model")
    assert result.model == "datarobot/anthropic/claude-sonnet-4-20250514"


async def test_execute_dragent_inline_keeps_workflow_provided_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GIVEN: a workflow that returns a real model name
    _install_fake_workflow(monkeypatch, _make_chat_response(content="ok", model="claude-sonnet"))

    # WHEN: execute_dragent_inline_async is called without a model in chat_completion
    result = await execute_dragent_inline_async(
        chat_completion={
            "messages": [{"role": "user", "content": "hi"}],
        },  # type: ignore[arg-type]
        custom_model_dir=Path("/tmp"),
    )

    # THEN: the workflow's model name is preserved (not overridden)
    assert result.model == "claude-sonnet"


# --- Config resolution --------------------------------------------------


def test_resolve_config_path_prefers_explicit_argument(tmp_path: Path) -> None:
    # GIVEN: an explicit override AND a workflow.yaml in custom_model_dir
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text("explicit")
    custom_dir = tmp_path / "model_dir"
    custom_dir.mkdir()
    (custom_dir / "workflow.yaml").write_text("default")

    # WHEN: resolving with explicit override
    resolved = _resolve_config_path(custom_dir, config_file=explicit)

    # THEN: the explicit argument wins
    assert resolved == explicit


def test_resolve_config_path_falls_back_to_custom_model_dir(tmp_path: Path) -> None:
    # GIVEN: no explicit override, workflow.yaml exists in custom dir
    custom_dir = tmp_path / "model_dir"
    custom_dir.mkdir()
    default_path = custom_dir / "workflow.yaml"
    default_path.write_text("default")

    # WHEN: resolving the config file
    resolved = _resolve_config_path(custom_dir, config_file=None)

    # THEN: the convention path is used
    assert resolved == default_path


def test_resolve_config_path_raises_file_not_found(tmp_path: Path) -> None:
    # GIVEN: nothing on disk
    empty_dir = tmp_path / "model_dir"
    empty_dir.mkdir()

    # WHEN/THEN: resolving raises FileNotFoundError naming the conventional path
    with pytest.raises(FileNotFoundError, match="workflow.yaml"):
        _resolve_config_path(empty_dir, config_file=None)


# --- Sync wrapper -------------------------------------------------------


def test_execute_dragent_inline_sync_wrapper_under_ipykernel_like_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GIVEN: a workflow that returns a ChatResponse
    _install_fake_workflow(monkeypatch, _make_chat_response(content="Sync"))
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _execute_sync_cell() -> ChatCompletion:
        return execute_dragent_inline(
            chat_completion={
                "model": "datarobot-e2e",
                "messages": [{"role": "user", "content": "hi"}],
            },  # type: ignore[arg-type]
            custom_model_dir=Path("/tmp"),
        )

    try:
        # WHEN: calling the sync entry point like run_agent inside ipykernel
        result = loop.run_until_complete(_execute_sync_cell())
    finally:
        asyncio.set_event_loop(None)
        loop.close()

    # THEN: the synchronous wrapper returns the same shape as the async function
    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content == "Sync"


# --- Trace id propagation -----------------------------------------------


async def test_execute_dragent_inline_propagates_active_trace_id_to_nat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GIVEN: an active OTel trace (as run_agent opens around the inline call)
    observed: dict = {}
    _install_trace_id_recording_workflow(monkeypatch, _make_chat_response(content="ok"), observed)

    trace_id = 0x0AF7651916CD43DD8448EB211C80319C
    span_context = SpanContext(
        trace_id=trace_id,
        span_id=0xB7AD6B7169203331,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    token = otel_context.attach(trace.set_span_in_context(NonRecordingSpan(span_context)))
    try:
        # WHEN: running the workflow inline
        await execute_dragent_inline_async(
            chat_completion={
                "model": "datarobot-e2e",
                "messages": [{"role": "user", "content": "hi"}],
            },  # type: ignore[arg-type]
            custom_model_dir=Path("/tmp"),
        )
    finally:
        otel_context.detach(token)

    # THEN: NAT saw the caller's trace id as its workflow_trace_id, so its spans
    # (and the middleware's) join the trace run_agent started.
    assert observed["workflow_trace_id"] == trace_id


async def test_execute_dragent_inline_without_active_trace_leaves_nat_trace_id_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GIVEN: no active OTel span (invalid span context)
    observed: dict = {}
    _install_trace_id_recording_workflow(monkeypatch, _make_chat_response(content="ok"), observed)

    # WHEN: running the workflow inline outside any span
    await execute_dragent_inline_async(
        chat_completion={
            "model": "datarobot-e2e",
            "messages": [{"role": "user", "content": "hi"}],
        },  # type: ignore[arg-type]
        custom_model_dir=Path("/tmp"),
    )

    # THEN: nothing is seeded — NAT keeps generating its own trace id downstream.
    assert observed["workflow_trace_id"] is None
