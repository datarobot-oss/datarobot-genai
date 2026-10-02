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

import asyncio
import inspect
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from nat.data_models.api_server import ResponsePayloadOutput
from nat.front_ends.fastapi import response_helpers

from datarobot_genai.dragent.frontends import stream_cancellation
from datarobot_genai.dragent.frontends.stream_cancellation import _EXPECTED_PARAMETERS
from datarobot_genai.dragent.frontends.stream_cancellation import generate_streaming_response
from datarobot_genai.dragent.frontends.stream_cancellation import (
    patch_nat_streaming_disconnect_cancellation,
)


class _SlowRun:
    """A workflow run that emits *chunks*, then optionally hangs like a slow LLM call."""

    def __init__(self, chunks: list[str], hang: bool) -> None:
        self.chunks = chunks
        self.hang = hang
        self.cancelled = asyncio.Event()
        self.finished = asyncio.Event()

    async def result_stream(self, to_type: Any = None) -> AsyncIterator[str]:
        try:
            for chunk in self.chunks:
                yield chunk
            if self.hang:
                await asyncio.sleep(3600)
            self.finished.set()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


def _session_for(run: _SlowRun) -> Any:
    @asynccontextmanager
    async def _run(payload: Any) -> AsyncIterator[_SlowRun]:
        yield run

    return SimpleNamespace(run=_run, workflow=SimpleNamespace(has_streaming_output=True))


@pytest.fixture(autouse=True)
def _no_intermediate_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip NAT's intermediate-step subscription, which needs a live NAT context."""

    async def _pull_intermediate(q: Any, adapter: Any) -> asyncio.Event:
        done = asyncio.Event()
        done.set()
        return done

    monkeypatch.setattr(stream_cancellation, "pull_intermediate", _pull_intermediate)


def _stream(run: _SlowRun) -> Any:
    return generate_streaming_response(None, session=_session_for(run), streaming=True)


def test_copy_matches_installed_nat_signature() -> None:
    """The replacement is a copy of NAT's body; a NAT bump that changes it must fail here."""
    original = inspect.signature(stream_cancellation.NAT_GENERATE_STREAMING_RESPONSE)
    assert tuple(original.parameters) == _EXPECTED_PARAMETERS


async def test_streams_every_chunk_when_run_completes() -> None:
    run = _SlowRun(["a", "b"], hang=False)

    items = [item async for item in _stream(run)]

    assert [item.payload for item in items] == ["a", "b"]
    assert all(isinstance(item, ResponsePayloadOutput) for item in items)
    assert run.finished.is_set()
    assert not run.cancelled.is_set()


async def test_aclose_cancels_the_running_workflow() -> None:
    run = _SlowRun(["a"], hang=True)
    stream = _stream(run)

    first = await stream.__anext__()
    await stream.aclose()

    assert first.payload == "a"
    assert run.cancelled.is_set()


async def test_cancelling_the_consumer_cancels_the_running_workflow() -> None:
    """Starlette's disconnect path: the task iterating the response is cancelled."""
    run = _SlowRun(["a"], hang=True)
    got_first = asyncio.Event()

    async def _consume() -> None:
        async for _ in _stream(run):
            got_first.set()

    consumer = asyncio.create_task(_consume())
    await got_first.wait()
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert run.cancelled.is_set()


async def test_producer_that_ignores_cancel_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stream_cancellation, "PRODUCER_CANCEL_GRACE_SECONDS", 0.05)
    release = asyncio.Event()

    async def _stubborn() -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await release.wait()

    task = asyncio.create_task(_stubborn())
    await asyncio.sleep(0)

    await asyncio.wait_for(stream_cancellation._stop_producer(task), timeout=1)

    assert not task.done()
    release.set()
    await task


def test_patch_replaces_nat_helper_and_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(stream_cancellation._PATCH_STATE, "patched", False)
    monkeypatch.setattr(
        response_helpers,
        "generate_streaming_response",
        response_helpers.generate_streaming_response,
    )

    patch_nat_streaming_disconnect_cancellation()
    patch_nat_streaming_disconnect_cancellation()

    assert response_helpers.generate_streaming_response is generate_streaming_response


def test_patch_skips_unknown_nat_signature(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def _changed(payload: Any, *, session: Any, new_param: int) -> AsyncIterator[Any]:
        yield None

    monkeypatch.setitem(stream_cancellation._PATCH_STATE, "patched", False)
    monkeypatch.setattr(response_helpers, "generate_streaming_response", _changed)

    patch_nat_streaming_disconnect_cancellation()

    assert response_helpers.generate_streaming_response is _changed
    assert "signature changed" in caplog.text
