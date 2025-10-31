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

# Tests for datarobot_genai.core.utils.b64
# One test class per helper, pytest fixtures for setup, DRY and pythonic style.
from __future__ import annotations

from typing import Any

import pytest

from datarobot_genai.core.utils import b64


@pytest.fixture
def sample_obj() -> dict[str, Any]:
    """Return a small JSON-serializable object used across tests."""
    return {"id": "foo", "title": "bar", "count": 12, "active": True}


@pytest.fixture
def sample_b64(sample_obj: dict[str, Any]) -> str:
    """Return a valid base64 encoding of `sample_obj` produced by the helper.

    This fixture asserts the encoder succeeded so tests can rely on a valid
    base64 string.
    """
    encoded = b64.encode_json_to_b64(sample_obj)
    assert isinstance(encoded, str) and encoded, "encoder fixture must produce a string"
    return encoded


class TestEncodeJsonToB64:
    def test_encode_happy_path_roundtrip(self, sample_obj: dict[str, Any], sample_b64: str) -> None:
        decoded = b64.decode_b64_to_json(sample_b64)
        assert decoded == sample_obj

    def test_encode_non_serializable_returns_none(self) -> None:
        # sets are not JSON serializable by default
        class NotSerializable:
            pass

        result = b64.encode_json_to_b64(NotSerializable())
        assert result is None

    def test_encode_respects_max_json_bytes(self, sample_obj: dict[str, Any]) -> None:
        tiny_limit = 1
        result = b64.encode_json_to_b64(sample_obj, max_json_bytes=tiny_limit)
        assert result is None

    def test_encode_with_custom_separators_and_ascii(self, sample_obj: dict[str, Any]) -> None:
        result = b64.encode_json_to_b64(sample_obj, ensure_ascii=True, separators=(",", ":"))
        assert isinstance(result, str) and result


class TestDecodeB64ToJson:
    def test_decode_happy_path(self, sample_obj: dict[str, Any], sample_b64: str) -> None:
        decoded = b64.decode_b64_to_json(sample_b64)
        assert decoded == sample_obj

    def test_decode_handles_whitespace_padding(self, sample_obj: dict[str, Any]) -> None:
        encoded = b64.encode_json_to_b64(sample_obj)
        assert isinstance(encoded, str)
        padded = f"  {encoded}\n"
        decoded = b64.decode_b64_to_json(padded)
        assert decoded == sample_obj

    @pytest.mark.parametrize("bad_input", [None, "", 123, b"abc", [1, 2, 3]])  # type: ignore[list-item]
    def test_decode_invalid_types_return_none(self, bad_input: Any) -> None:
        assert b64.decode_b64_to_json(bad_input) is None

    def test_decode_malformed_base64_returns_none(self) -> None:
        assert b64.decode_b64_to_json("!!!not-base64!!!") is None
