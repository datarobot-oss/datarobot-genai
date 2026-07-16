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

"""Unit tests for the chat-history constants and pure helpers."""

from __future__ import annotations

from uuid import UUID
from uuid import uuid4

import pytest

from datarobot_genai.application_utils.chat_history.constants import _ZW_PLACEHOLDER
from datarobot_genai.application_utils.chat_history.constants import app_str
from datarobot_genai.application_utils.chat_history.constants import chat_deduplication_key
from datarobot_genai.application_utils.chat_history.constants import emitter_for_role
from datarobot_genai.application_utils.chat_history.constants import normalize_participant_id
from datarobot_genai.application_utils.chat_history.constants import participant_id
from datarobot_genai.application_utils.chat_history.constants import session_deduplication_key
from datarobot_genai.application_utils.chat_history.constants import wire_non_empty_str

# ── Placeholder codec ──────────────────────────────────────────────────────────


def test_wire_non_empty_str_encodes_empty_and_none() -> None:
    """GIVEN empty/None WHEN encoded THEN the placeholder is substituted."""
    assert wire_non_empty_str("") == _ZW_PLACEHOLDER
    assert wire_non_empty_str(None) == _ZW_PLACEHOLDER


def test_wire_non_empty_str_preserves_non_empty() -> None:
    """GIVEN a non-empty string WHEN encoded THEN it is returned unchanged."""
    assert wire_non_empty_str("hello") == "hello"


def test_app_str_strips_placeholder() -> None:
    """GIVEN the placeholder or empty WHEN decoded THEN the empty string is returned."""
    assert app_str(_ZW_PLACEHOLDER) == ""
    assert app_str("") == ""
    assert app_str(None) == ""


def test_app_str_preserves_real_content() -> None:
    """GIVEN real content WHEN decoded THEN it is returned unchanged."""
    assert app_str("hello") == "hello"


@pytest.mark.parametrize("value", ["", "hello", "{}", "multi\nline"])
def test_placeholder_codec_round_trips(value: str) -> None:
    """GIVEN any string WHEN wire-encoded then app-decoded THEN the original is recovered."""
    assert app_str(wire_non_empty_str(value)) == value


# ── Deduplication keys ──────────────────────────────────────────────────────────


def test_session_deduplication_key_is_deterministic_nul_joined() -> None:
    """GIVEN a namespace and parts WHEN hashed THEN it matches a NUL-joined SHA-256 prefix."""
    import hashlib

    key = session_deduplication_key("chat", "u", "t")
    expected = hashlib.sha256(b"chat\0u\0t").hexdigest()[:64]
    assert key == expected
    assert len(key) == 64


def test_chat_deduplication_key_matches_namespaced_helper() -> None:
    """GIVEN a user and thread WHEN keyed THEN it equals the "chat"-namespaced key."""
    user = uuid4()
    thread = "thread-42"
    assert chat_deduplication_key(user, thread) == session_deduplication_key(
        "chat", str(user), thread
    )


def test_chat_deduplication_key_is_idempotent_and_thread_sensitive() -> None:
    """GIVEN the same (user, thread) WHEN keyed twice THEN keys match; a new thread differs."""
    user = uuid4()
    assert chat_deduplication_key(user, "t1") == chat_deduplication_key(user, "t1")
    assert chat_deduplication_key(user, "t1") != chat_deduplication_key(user, "t2")


# ── Participant id ──────────────────────────────────────────────────────────────


def test_participant_id_is_deterministic_24_hex() -> None:
    """GIVEN a user UUID WHEN derived twice THEN a stable 24-hex id is produced."""
    user = UUID("12345678-1234-5678-1234-567812345678")
    first = participant_id(user)
    assert first == participant_id(user)
    assert len(first) == 24
    int(first, 16)  # is valid hex


def test_participant_id_uses_valid_override() -> None:
    """GIVEN a valid 24-hex override WHEN derived THEN the normalised override is returned."""
    user = uuid4()
    override = "AABBCCDDEEFF001122334455"
    assert participant_id(user, override=override) == override.lower()


def test_participant_id_ignores_invalid_override() -> None:
    """GIVEN an invalid override WHEN derived THEN the derived value is used instead."""
    user = uuid4()
    assert participant_id(user, override="not-hex") == participant_id(user)


def test_normalize_participant_id_validates_shape() -> None:
    """GIVEN candidate ids WHEN normalised THEN only 24-hex values survive."""
    assert normalize_participant_id("AABBCCDDEEFF001122334455") == "aabbccddeeff001122334455"
    assert normalize_participant_id("tooshort") is None
    assert normalize_participant_id("zzzzzzzzzzzzzzzzzzzzzzzz") is None
    assert normalize_participant_id(None) is None


# ── Emitter derivation ──────────────────────────────────────────────────────────


def test_emitter_for_user_role_carries_participant_id() -> None:
    """GIVEN a user message WHEN deriving the emitter THEN it is ("user", participant_id)."""
    assert emitter_for_role("user", "pid-123") == ("user", "pid-123")


@pytest.mark.parametrize("role", ["assistant", "tool", "reasoning", "system", "developer"])
def test_emitter_for_non_user_roles_is_anonymous_agent(role: str) -> None:
    """GIVEN a non-user message WHEN deriving the emitter THEN it is ("agent", None)."""
    assert emitter_for_role(role, "pid-123") == ("agent", None)
