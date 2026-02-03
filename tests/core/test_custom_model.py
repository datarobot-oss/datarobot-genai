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
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import AsyncGenerator
from collections.abc import Iterator
from typing import Any

from datarobot_genai.core.chat import CustomModelChatResponse
from datarobot_genai.core.chat import CustomModelStreamingResponse
from datarobot_genai.core.custom_model import chat_entrypoint
from datarobot_genai.core.custom_model import load_model


class _DummyAgent:
    def __init__(self, **_: Any) -> None:
        pass

    async def invoke(self, completion_create_params: Any) -> tuple[str, None, dict[str, int]]:
        assert "messages" in completion_create_params
        return ("ok", None, {})


class _DummyStreamingAgent:
    def __init__(self, **_: Any) -> None:
        pass

    async def invoke(
        self, completion_create_params: Any
    ) -> AsyncGenerator[tuple[str, None, dict[str, int]], None]:
        assert "messages" in completion_create_params

        async def _gen() -> AsyncGenerator[tuple[str, None, dict[str, int]], None]:
            yield ("a", None, {})
            yield ("b", None, {})
            return

        return _gen()


def _minimal_params() -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "test-model",
        # populate explicit empty auth context to avoid any external coupling
        "authorization_context": {},
    }


def test_chat_entrypoint_non_streaming() -> None:
    pool, loop = load_model()
    try:
        resp = chat_entrypoint(
            _DummyAgent,
            _minimal_params(),
            (pool, loop),
            work_dir=".",
            runtime_parameter_keys=[],
        )
        assert isinstance(resp, CustomModelChatResponse)
        assert resp.choices[0].message.content == "ok"
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass


def test_chat_entrypoint_streaming() -> None:
    pool, loop = load_model()
    try:
        stream: Iterator[CustomModelStreamingResponse] | CustomModelChatResponse = chat_entrypoint(
            _DummyStreamingAgent,
            _minimal_params(),
            (pool, loop),
            work_dir=".",
            runtime_parameter_keys=[],
        )
        assert not isinstance(stream, CustomModelChatResponse)
        chunks = list(stream)  # type: ignore[arg-type]
        # Expect at least 2 deltas and one terminal chunk
        assert any(c.choices[0].delta.content == "a" for c in chunks if c.choices[0].delta)
        assert any(c.choices[0].delta.content == "b" for c in chunks if c.choices[0].delta)
        assert chunks[-1].choices[0].finish_reason == "stop"
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass


def test_chat_entrypoint_keeps_allowed_headers() -> None:
    """Test that allowed headers are kept in forwarded_headers."""
    pool, loop = load_model()
    captured_params: dict[str, Any] = {}

    class _CapturingAgent:
        def __init__(self, **kwargs: Any) -> None:
            captured_params.update(kwargs)

        async def invoke(self, completion_create_params: Any) -> tuple[str, None, dict[str, int]]:
            return ("ok", None, {})

    try:
        headers = {
            "x-datarobot-api-key": "scoped-token-123",  # Should be kept
            "x-datarobot-api-token": "scoped-token-456",  # Should be kept
            "x-datarobot-identity-token": "identity-token-123",  # Should be kept
            "x-custom-header": "custom-value",  # Should be filtered
            "Authorization": "Bearer user-provided-token",  # Should be filtered
            "authorization": "Bearer another-token",  # Should be filtered
            "X-DataRobot-Authorization-Context": "fake-jwt-token",  # Should be filtered
            "x-datarobot-authorization-context": "another-fake-token",  # Should be filtered
        }

        resp = chat_entrypoint(
            _CapturingAgent,
            _minimal_params(),
            (pool, loop),
            work_dir=".",
            runtime_parameter_keys=[],
            headers=headers,
        )
        assert isinstance(resp, CustomModelChatResponse)

        # Verify only allowed headers are kept in forwarded_headers
        forwarded_headers = captured_params.get("forwarded_headers", {})
        assert "x-datarobot-api-key" in forwarded_headers
        assert forwarded_headers["x-datarobot-api-key"] == "scoped-token-123"
        assert "x-datarobot-api-token" in forwarded_headers
        assert forwarded_headers["x-datarobot-api-token"] == "scoped-token-456"
        assert "x-datarobot-identity-token" in forwarded_headers
        assert forwarded_headers["x-datarobot-identity-token"] == "identity-token-123"
        # Verify other headers are filtered out
        assert "x-custom-header" not in forwarded_headers
        assert "Authorization" not in forwarded_headers
        assert "authorization" not in forwarded_headers
        assert "X-DataRobot-Authorization-Context" not in forwarded_headers
        assert "x-datarobot-authorization-context" not in forwarded_headers
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass


def test_chat_entrypoint_only_allows_specific_headers() -> None:
    """Test that only x-datarobot-api-key and x-datarobot-api-token are allowed."""
    pool, loop = load_model()
    captured_params: dict[str, Any] = {}

    class _CapturingAgent:
        def __init__(self, **kwargs: Any) -> None:
            captured_params.update(kwargs)

        async def invoke(self, completion_create_params: Any) -> tuple[str, None, dict[str, int]]:
            return ("ok", None, {})

    try:
        headers = {
            "x-datarobot-api-key": "scoped-token-123",
            "x-datarobot-api-token": "scoped-token-456",
            "x-datarobot-identity-token": "identity-token-123",
        }

        resp = chat_entrypoint(
            _CapturingAgent,
            _minimal_params(),
            (pool, loop),
            work_dir=".",
            runtime_parameter_keys=[],
            headers=headers,
        )
        assert isinstance(resp, CustomModelChatResponse)

        # Verify both allowed headers are present
        forwarded_headers = captured_params.get("forwarded_headers", {})
        assert len(forwarded_headers) == 3
        assert forwarded_headers["x-datarobot-api-key"] == "scoped-token-123"
        assert forwarded_headers["x-datarobot-api-token"] == "scoped-token-456"
        assert forwarded_headers["x-datarobot-identity-token"] == "identity-token-123"
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass


def test_chat_entrypoint_forwarded_headers_empty_when_no_headers() -> None:
    """Test that forwarded_headers is empty dict when no headers are provided."""
    pool, loop = load_model()
    captured_params: dict[str, Any] = {}

    class _CapturingAgent:
        def __init__(self, **kwargs: Any) -> None:
            captured_params.update(kwargs)

        async def invoke(self, completion_create_params: Any) -> tuple[str, None, dict[str, int]]:
            return ("ok", None, {})

    try:
        resp = chat_entrypoint(
            _CapturingAgent,
            _minimal_params(),
            (pool, loop),
            work_dir=".",
            runtime_parameter_keys=[],
        )
        assert isinstance(resp, CustomModelChatResponse)

        # Verify forwarded_headers is empty dict when no headers provided
        forwarded_headers = captured_params.get("forwarded_headers", {})
        assert forwarded_headers == {}
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass
