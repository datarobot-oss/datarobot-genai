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
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import ChatResponseChoice
from nat.data_models.api_server import ChoiceMessage
from nat.data_models.api_server import Usage
from openai.types.chat import ChatCompletion

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
    async def _fake_load_workflow(_path, headers=None):
        yield _FakeSessionManager()

    return _fake_load_workflow


def _install_fake_workflow(monkeypatch: pytest.MonkeyPatch, response: ChatResponse) -> None:
    """Patch ``load_workflow`` and bypass disk lookup of workflow.yaml."""
    monkeypatch.setattr(
        "datarobot_genai.nat.helpers.load_workflow",
        _build_fake_load_workflow(response),
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


# --- Header forwarding --------------------------------------------------


async def test_execute_dragent_inline_forwards_default_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GIVEN: a fake load_workflow that records the headers it was invoked with
    captured: dict[str, object] = {}
    fake = _build_fake_load_workflow(_make_chat_response(content="ok"))

    @asynccontextmanager
    async def _sniffing_load_workflow(path, headers=None):
        captured["path"] = path
        captured["headers"] = headers
        async with fake(path, headers=headers) as sm:
            yield sm

    monkeypatch.setattr(
        "datarobot_genai.nat.helpers.load_workflow",
        _sniffing_load_workflow,
    )
    monkeypatch.setattr(
        "datarobot_genai.dragent.inline._resolve_config_path",
        lambda custom_model_dir, config_file: Path("/tmp/fake.yaml"),
    )

    # WHEN: calling the inline function with default_headers
    await execute_dragent_inline_async(
        chat_completion={
            "model": "datarobot-e2e",
            "messages": [{"role": "user", "content": "hi"}],
        },  # type: ignore[arg-type]
        custom_model_dir=Path("/tmp"),
        default_headers={"x-datarobot-api-token": "secret"},
    )

    # THEN: load_workflow received the supplied headers
    assert captured["headers"] == {"x-datarobot-api-token": "secret"}
