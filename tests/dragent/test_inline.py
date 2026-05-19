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

import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import ChatResponseChunkChoice
from nat.data_models.api_server import ChoiceDelta
from nat.data_models.api_server import ChoiceDeltaToolCall
from nat.data_models.api_server import ChoiceDeltaToolCallFunction
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionChunk

from datarobot_genai.dragent import execute_dragent_inline_async
from datarobot_genai.dragent.inline import _resolve_config_path
from datarobot_genai.dragent.inline import execute_dragent_inline

# --- Test helpers ---------------------------------------------------------


def _make_chunk(
    *,
    choices: list[ChatResponseChunkChoice],
    id_: str = "cmpl-1",
    model: str = "datarobot-e2e",
) -> ChatResponseChunk:
    return ChatResponseChunk(
        id=id_,
        choices=choices,
        created=datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC),
        model=model,
    )


def _text_chunk(text: str, *, finish_reason: str | None = None) -> ChatResponseChunk:
    return _make_chunk(
        choices=[
            ChatResponseChunkChoice(
                index=0,
                delta=ChoiceDelta(content=text),
                finish_reason=finish_reason,  # type: ignore[arg-type]
            )
        ]
    )


def _build_fake_load_workflow(chunks: list[ChatResponseChunk]):
    """Build an async context manager that mimics ``load_workflow`` -> session -> runner."""

    class _FakeRunner:
        async def __aenter__(self):  # noqa: D401 - test stub
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: D401 - test stub
            return False

        async def result_stream(self, to_type):  # noqa: D401 - test stub
            for chunk in chunks:
                yield chunk

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


def _install_fake_workflow(
    monkeypatch: pytest.MonkeyPatch, chunks: list[ChatResponseChunk]
) -> None:
    """Patch ``load_workflow`` (and the deep session/run/stream chain) to yield ``chunks``."""
    # Patch at source: inline.py uses a local import inside the function body
    # so we must replace the symbol where it lives.
    monkeypatch.setattr(
        "datarobot_genai.nat.helpers.load_workflow",
        _build_fake_load_workflow(chunks),
    )

    # We don't depend on the file existing during the unit tests.
    monkeypatch.setattr(
        "datarobot_genai.dragent.inline._resolve_config_path",
        lambda custom_model_dir, config_file: Path("/tmp/fake-workflow.yaml"),
    )


# --- Aggregation: streaming vs non-streaming ------------------------------


async def test_execute_dragent_inline_non_streaming_aggregates_into_chat_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GIVEN: a dragent workflow that emits two text deltas and a final stop chunk
    _install_fake_workflow(
        monkeypatch,
        chunks=[
            _text_chunk("Hello"),
            _text_chunk(" world"),
            _text_chunk("", finish_reason="stop"),
        ],
    )

    # WHEN: execute_dragent_inline_async is called with stream=False
    result = await execute_dragent_inline_async(
        chat_completion={
            "model": "datarobot-e2e",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },  # type: ignore[arg-type]
        custom_model_dir=Path("/tmp"),
    )

    # THEN: the result is a single ChatCompletion with concatenated content
    assert isinstance(result, ChatCompletion)
    assert len(result.choices) == 1
    assert result.choices[0].message.content == "Hello world"
    assert result.choices[0].finish_reason == "stop"
    assert result.model == "datarobot-e2e"
    assert result.object == "chat.completion"


async def test_execute_dragent_inline_streaming_returns_chunk_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GIVEN: a dragent workflow that emits two text deltas
    _install_fake_workflow(
        monkeypatch,
        chunks=[_text_chunk("Hello"), _text_chunk(" world", finish_reason="stop")],
    )

    # WHEN: execute_dragent_inline_async is called with stream=True
    result = await execute_dragent_inline_async(
        chat_completion={
            "model": "datarobot-e2e",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },  # type: ignore[arg-type]
        custom_model_dir=Path("/tmp"),
    )

    # THEN: each ChatResponseChunk is returned as a typed OpenAI ChatCompletionChunk
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(c, ChatCompletionChunk) for c in result)
    assert result[0].choices[0].delta.content == "Hello"
    assert result[1].choices[0].delta.content == " world"
    assert result[1].choices[0].finish_reason == "stop"


async def test_execute_dragent_inline_merges_tool_call_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GIVEN: a workflow that emits a tool_call name then incremental argument shards
    _install_fake_workflow(
        monkeypatch,
        chunks=[
            _make_chunk(
                choices=[
                    ChatResponseChunkChoice(
                        index=0,
                        delta=ChoiceDelta(
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=0,
                                    id="call_abc",
                                    type="function",
                                    function=ChoiceDeltaToolCallFunction(
                                        name="lookup", arguments='{"q":'
                                    ),
                                )
                            ]
                        ),
                    )
                ]
            ),
            _make_chunk(
                choices=[
                    ChatResponseChunkChoice(
                        index=0,
                        delta=ChoiceDelta(
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=0,
                                    function=ChoiceDeltaToolCallFunction(arguments='"weather"}'),
                                )
                            ]
                        ),
                        finish_reason="tool_calls",  # type: ignore[arg-type]
                    )
                ]
            ),
        ],
    )

    # WHEN: aggregated into a single completion
    result = await execute_dragent_inline_async(
        chat_completion={
            "model": "datarobot-e2e",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },  # type: ignore[arg-type]
        custom_model_dir=Path("/tmp"),
    )

    # THEN: the tool_call fragments collapse into one OpenAI tool_calls entry
    assert isinstance(result, ChatCompletion)
    assert result.choices[0].finish_reason == "tool_calls"
    tool_calls = result.choices[0].message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call_abc"
    assert tool_calls[0].function.name == "lookup"
    assert tool_calls[0].function.arguments == '{"q":"weather"}'


async def test_execute_dragent_inline_empty_stream_returns_empty_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GIVEN: a workflow that emits no chunks
    _install_fake_workflow(monkeypatch, chunks=[])

    # WHEN: non-streaming aggregation runs over an empty list
    result = await execute_dragent_inline_async(
        chat_completion={
            "model": "fallback-model",
            "messages": [{"role": "user", "content": "hi"}],
        },  # type: ignore[arg-type]
        custom_model_dir=Path("/tmp"),
    )

    # THEN: a well-formed empty ChatCompletion is returned (no choices, model preserved)
    assert isinstance(result, ChatCompletion)
    assert result.choices == []
    assert result.model == "fallback-model"


# --- Config resolution ----------------------------------------------------


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


# --- Sync wrapper ---------------------------------------------------------


def test_execute_dragent_inline_sync_wrapper_delegates_to_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GIVEN: a workflow that emits a single text chunk
    _install_fake_workflow(monkeypatch, chunks=[_text_chunk("Sync", finish_reason="stop")])

    # WHEN: calling the synchronous wrapper
    result = execute_dragent_inline(
        chat_completion={
            "model": "datarobot-e2e",
            "messages": [{"role": "user", "content": "hi"}],
        },  # type: ignore[arg-type]
        custom_model_dir=Path("/tmp"),
    )

    # THEN: the synchronous wrapper returns the same shape as the async function
    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content == "Sync"


# --- Header forwarding ----------------------------------------------------


async def test_execute_dragent_inline_forwards_default_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GIVEN: a fake load_workflow that records the headers it was invoked with
    captured: dict[str, object] = {}
    fake = _build_fake_load_workflow([_text_chunk("ok", finish_reason="stop")])

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


# --- Front-end import side effect ----------------------------------------


async def test_execute_dragent_inline_imports_register_for_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GIVEN: the inline module is invoked
    _install_fake_workflow(monkeypatch, chunks=[_text_chunk("ok", finish_reason="stop")])

    # WHEN: execute_dragent_inline_async runs (importing register inside the function)
    with patch("datarobot_genai.dragent.frontends.register") as register_module:
        # Ensure the symbol is importable but the patch records the import.
        register_module.__name__ = "datarobot_genai.dragent.frontends.register"
        await execute_dragent_inline_async(
            chat_completion={
                "model": "datarobot-e2e",
                "messages": [{"role": "user", "content": "hi"}],
            },  # type: ignore[arg-type]
            custom_model_dir=Path("/tmp"),
        )

    # THEN: no exception was raised — the lazy import resolved successfully.
    # (The body's side-effect import registers global type converters; we cover
    # the resolution flow at import time end-to-end via the e2e test.)
