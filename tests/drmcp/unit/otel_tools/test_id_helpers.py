# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for require_object_id and require_trace_id (drtools/core/utils.py).

Style: GIVEN preconditions / WHEN behavior under test / THEN expected outcomes.

These are the local-validation helpers the OTel tools (traces.py et al., not yet
built) will use to reject a malformed id before it reaches the server, per §2's
"Local validation before the call" convention. Housed alongside the OTel client
tests because this round is what first needs them; require_id itself predates
this package and is only exercised indirectly here.
"""

import pytest

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.core.utils import require_object_id
from datarobot_genai.drtools.core.utils import require_trace_id

_VALID_OBJECT_ID = "0123456789abcdef01234567"  # 24 hex chars
_VALID_TRACE_ID = "0123456789abcdef" * 2  # 32 hex chars


# ------------------------------------------------------------------ #
# require_object_id                                                    #
# ------------------------------------------------------------------ #


def test_require_object_id_accepts_24_lowercase_hex_chars() -> None:
    # GIVEN a well-formed 24-char hex id
    # WHEN require_object_id validates it
    result = require_object_id(_VALID_OBJECT_ID, "entity_id")

    # THEN it is returned unchanged
    assert result == _VALID_OBJECT_ID


def test_require_object_id_accepts_uppercase_hex_chars() -> None:
    # GIVEN a 24-char id using uppercase hex digits (MongoIdField accepts both cases)
    # WHEN require_object_id validates it
    result = require_object_id("ABCDEF0123456789ABCDEF01", "entity_id")

    # THEN it is accepted and returned unchanged
    assert result == "ABCDEF0123456789ABCDEF01"


def test_require_object_id_strips_surrounding_whitespace() -> None:
    # GIVEN a valid id with surrounding whitespace
    # WHEN require_object_id validates it
    result = require_object_id(f"  {_VALID_OBJECT_ID}  ", "entity_id")

    # THEN the whitespace is stripped before length/pattern checks
    assert result == _VALID_OBJECT_ID


def test_require_object_id_rejects_wrong_length() -> None:
    # GIVEN a 23-char (too short) hex string
    # WHEN require_object_id validates it
    with pytest.raises(ToolError) as exc_info:
        require_object_id(_VALID_OBJECT_ID[:-1], "entity_id")

    # THEN a VALIDATION error names the argument and the expected shape
    assert exc_info.value.kind is ToolErrorKind.VALIDATION
    assert "entity_id" in str(exc_info.value)
    assert "24-character hex" in str(exc_info.value)


def test_require_object_id_rejects_non_hex_characters() -> None:
    # GIVEN a 24-char string containing a non-hex character
    not_hex = "g" + _VALID_OBJECT_ID[1:]

    # WHEN require_object_id validates it
    with pytest.raises(ToolError) as exc_info:
        require_object_id(not_hex, "entity_id")

    # THEN it is rejected as a VALIDATION error
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


def test_require_object_id_rejects_empty_value() -> None:
    # GIVEN an empty string
    # WHEN require_object_id validates it
    with pytest.raises(ToolError) as exc_info:
        require_object_id("", "entity_id")

    # THEN it fails via require_id's own empty check, naming the argument
    assert exc_info.value.kind is ToolErrorKind.VALIDATION
    assert "entity_id" in str(exc_info.value)


def test_require_object_id_rejects_non_string_value() -> None:
    # GIVEN a non-str value (e.g. a caller passed an int id)
    # WHEN require_object_id validates it
    with pytest.raises(ToolError) as exc_info:
        require_object_id(123456789012345678901234, "entity_id")  # type: ignore[arg-type]

    # THEN it fails with a readable VALIDATION error, not an AttributeError
    # from require_id's unguarded `.strip()` on a non-str value
    assert exc_info.value.kind is ToolErrorKind.VALIDATION
    assert "entity_id" in str(exc_info.value)


# ------------------------------------------------------------------ #
# require_trace_id                                                     #
# ------------------------------------------------------------------ #


def test_require_trace_id_accepts_32_lowercase_hex_chars() -> None:
    # GIVEN a well-formed 32-char hex trace id
    # WHEN require_trace_id validates it
    result = require_trace_id(_VALID_TRACE_ID)

    # THEN it is returned unchanged
    assert result == _VALID_TRACE_ID


def test_require_trace_id_lowercases_uppercase_hex() -> None:
    # GIVEN a 32-char trace id pasted with uppercase hex digits
    # WHEN require_trace_id validates it
    result = require_trace_id(_VALID_TRACE_ID.upper())

    # THEN it is accepted and lowercased — the server emits and matches lowercase
    # hex, so forwarding the uppercase form verbatim would miss an existing trace
    assert result == _VALID_TRACE_ID


def test_require_trace_id_strips_surrounding_whitespace() -> None:
    # GIVEN a valid trace id with surrounding whitespace
    # WHEN require_trace_id validates it
    result = require_trace_id(f"  {_VALID_TRACE_ID}  ")

    # THEN the whitespace is stripped before length/pattern checks
    assert result == _VALID_TRACE_ID


def test_require_trace_id_rejects_24_char_object_id_shaped_value() -> None:
    # GIVEN a value shaped like a 24-char object id, not a 32-char trace id
    # WHEN require_trace_id validates it
    with pytest.raises(ToolError) as exc_info:
        require_trace_id(_VALID_OBJECT_ID)

    # THEN it is rejected — the two id shapes are not interchangeable
    assert exc_info.value.kind is ToolErrorKind.VALIDATION
    assert "trace_id" in str(exc_info.value)
    assert "32-character hex" in str(exc_info.value)


def test_require_trace_id_rejects_wrong_length() -> None:
    # GIVEN a 33-char (too long) hex string
    # WHEN require_trace_id validates it
    with pytest.raises(ToolError) as exc_info:
        require_trace_id(_VALID_TRACE_ID + "0")

    # THEN it is rejected as a VALIDATION error
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


def test_require_trace_id_rejects_non_hex_characters() -> None:
    # GIVEN a 32-char string containing a non-hex character
    not_hex = "z" + _VALID_TRACE_ID[1:]

    # WHEN require_trace_id validates it
    with pytest.raises(ToolError) as exc_info:
        require_trace_id(not_hex)

    # THEN it is rejected as a VALIDATION error
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


def test_require_trace_id_rejects_empty_value() -> None:
    # GIVEN an empty string
    # WHEN require_trace_id validates it
    with pytest.raises(ToolError) as exc_info:
        require_trace_id("")

    # THEN it fails via require_id's own empty check, naming 'trace_id'
    assert exc_info.value.kind is ToolErrorKind.VALIDATION
    assert "trace_id" in str(exc_info.value)


def test_require_trace_id_rejects_non_string_value() -> None:
    # GIVEN a non-str value (e.g. a caller passed an int id)
    # WHEN require_trace_id validates it
    with pytest.raises(ToolError) as exc_info:
        require_trace_id(12345678901234567890123456789012)  # type: ignore[arg-type]

    # THEN it fails with a readable VALIDATION error, not an AttributeError
    # from require_id's unguarded `.strip()` on a non-str value
    assert exc_info.value.kind is ToolErrorKind.VALIDATION
    assert "trace_id" in str(exc_info.value)
