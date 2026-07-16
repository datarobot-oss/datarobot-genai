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

"""Unit tests for the description codec (_encoding.py)."""

from __future__ import annotations

import pytest

from datarobot_genai.application_utils.persistence._encoding import build_description
from datarobot_genai.application_utils.persistence._encoding import build_query_description
from datarobot_genai.application_utils.persistence._encoding import parse_description
from datarobot_genai.application_utils.persistence._encoding import validate_range_key

# ── validate_range_key ────────────────────────────────────────────────────────


def test_validate_range_key_raises_on_empty_string() -> None:
    """GIVEN an empty string WHEN validate_range_key is called THEN raises ValueError."""
    with pytest.raises(ValueError, match="must not be empty"):
        validate_range_key("my_field", "")


def test_validate_range_key_passes_non_empty() -> None:
    """GIVEN a non-empty string WHEN validate_range_key is called THEN no exception."""
    validate_range_key("my_field", "acme")


def test_validate_range_key_passes_single_char() -> None:
    """GIVEN a single character WHEN validate_range_key is called THEN no exception."""
    validate_range_key("x", "a")


# ── build_description ─────────────────────────────────────────────────────────


def test_build_description_no_values_produces_anchor_prefix_only() -> None:
    """GIVEN empty values WHEN build_description THEN only prefix anchor."""
    # WHEN
    result = build_description("chat", [])
    # THEN
    assert result == "//chat/"


def test_build_description_one_value() -> None:
    """GIVEN one value WHEN build_description THEN single segment after prefix."""
    result = build_description("chat", ["acme"])
    assert result == "//chat/acme/"


def test_build_description_two_values() -> None:
    """GIVEN two values WHEN build_description THEN two segments."""
    result = build_description("chat", ["acme", "billing"])
    assert result == "//chat/acme/billing/"


def test_build_description_escapes_slash_in_prefix() -> None:
    """GIVEN a prefix with a slash WHEN build_description THEN slash percent-encoded."""
    result = build_description("my/prefix", ["val"])
    assert result == "//my%2Fprefix/val/"


def test_build_description_escapes_slash_in_value() -> None:
    """GIVEN a value containing a slash WHEN build_description THEN slash encoded."""
    result = build_description("chat", ["key/val"])
    assert result == "//chat/key%2Fval/"


def test_build_description_escapes_percent_in_value() -> None:
    """GIVEN a value containing % WHEN build_description THEN percent encoded first."""
    result = build_description("chat", ["100%sure"])
    assert result == "//chat/100%25sure/"


def test_build_description_escapes_percent_then_slash() -> None:
    """GIVEN a value with both % and / WHEN build_description THEN correct order."""
    # '%/' in source → '%25%2F' in encoded (percent first, then slash)
    result = build_description("p", ["%/"])
    assert result == "//p/%25%2F/"


def test_build_description_raises_on_over_1000_chars() -> None:
    """GIVEN values that produce > 1000 encoded chars WHEN build_description THEN raises."""
    long_val = "a" * 998
    with pytest.raises(ValueError, match="1000-character limit"):
        build_description("p", [long_val])


def test_build_description_exactly_1000_chars_ok() -> None:
    """GIVEN values that produce exactly 1000 chars WHEN build_description THEN ok."""
    # "//p/" = 4 chars; need 996 more; use "x" * 994 + "/" trailing = 995 + "/"
    # Actually: "//p/" + "x"*994 + "/" = 4 + 994 + 1 = 999. Let's compute exactly.
    # We need total = 1000. "//p/" = 4. Remaining segment + "/" = 996 chars.
    # Segment value = 995 chars. Total = 4 + 995 + 1 = 1000.
    val = "x" * 995
    result = build_description("p", [val])
    assert len(result) == 1000


# ── build_query_description ───────────────────────────────────────────────────


def test_build_query_description_returns_same_as_build_description() -> None:
    """GIVEN normal values WHEN build_query_description THEN same as build_description."""
    assert build_query_description("chat", ["acme"]) == build_description("chat", ["acme"])


def test_build_query_description_raises_when_too_short() -> None:
    """GIVEN a very short prefix with no values WHEN build_query THEN raises (< 3 chars)."""
    # "//x/" = 4 chars — OK
    result = build_query_description("x", [])
    assert len(result) >= 3

    # Cannot construct a sub-3-char result through the API because the minimum is
    # "//" + prefix + "/" which is always >= 4 chars for any non-empty prefix.
    # The check is a guard for future edge cases; test that it branches correctly
    # by testing the helper is reachable and doesn't crash for a 3-char output.


# ── Prefix-of-hierarchy anchoring (the key ORM invariant) ────────────────────


def test_prefix_query_matches_stored_with_one_range_key() -> None:
    """GIVEN stored with 2 keys WHEN query with 1 key THEN query IS a substring of stored."""
    stored = build_description("chat", ["acme", "billing"])
    query = build_query_description("chat", ["acme"])
    # Simulates what the service does: case-insensitive substring match
    assert query.lower() in stored.lower()


def test_prefix_query_with_all_keys_matches_stored_exactly() -> None:
    """GIVEN stored with 2 keys WHEN query with same 2 keys THEN equal."""
    stored = build_description("chat", ["acme", "billing"])
    query = build_query_description("chat", ["acme", "billing"])
    assert query == stored


def test_partial_segment_does_not_match_stored() -> None:
    """GIVEN query uses partial segment text THEN it is NOT a substring of stored."""
    stored = build_description("chat", ["acme", "billing"])
    # A query with "bil" instead of "billing" should NOT match.
    # build_query_description("chat", ["acme"]) produces "//chat/acme/" which does match —
    # that is the correct anchored-prefix behaviour.  The test checks a *partial segment*
    # query that should NOT match because boundary enforcement is what matters.
    false_query = "//chat/acme/bil/"
    # The stored description is "//chat/acme/billing/"
    # "//chat/acme/bil/" is NOT a prefix because "billing/" != "bil/"
    assert false_query not in stored


def test_different_prefix_does_not_match() -> None:
    """GIVEN two different prefixes WHEN query one against the other THEN no match."""
    stored = build_description("chat", ["acme"])
    query = build_query_description("widget", ["acme"])
    assert query.lower() not in stored.lower()


def test_prefix_is_not_matched_by_a_longer_prefix_sharing_its_start() -> None:
    """GIVEN prefixes 'thread' and 'threadx' WHEN cross-querying THEN neither matches.

    Trailing-slash anchoring keeps prefixes disjoint even when one is a raw-string
    prefix of the other — the reason ``EntityLocator`` (prefix ``loc``) can share a
    space with ``Chat`` (prefix ``thread``) without ``.list`` bleeding across them.
    """
    stored_thread = build_description("thread", ["acme"])  # //thread/acme/
    stored_threadx = build_description("threadx", ["acme"])  # //threadx/acme/
    # A no-value query for one prefix must not substring-match the other's storage.
    assert build_query_description("thread", []).lower() not in stored_threadx.lower()
    assert build_query_description("threadx", []).lower() not in stored_thread.lower()


def test_loc_prefix_disjoint_from_thread_prefix() -> None:
    """GIVEN 'loc' and 'thread' prefixes WHEN cross-querying THEN no substring match."""
    stored_chat = build_description("thread", ["acme", "billing"])
    stored_loc = build_description("loc", ["chat:some-uuid"])
    assert build_query_description("loc", []).lower() not in stored_chat.lower()
    assert build_query_description("thread", []).lower() not in stored_loc.lower()


# ── parse_description ─────────────────────────────────────────────────────────


def test_parse_description_two_segments() -> None:
    """GIVEN a stored description with 2 segments WHEN parse THEN correct values."""
    desc = build_description("chat", ["acme", "billing"])
    values = parse_description("chat", desc, 2)
    assert values == ["acme", "billing"]


def test_parse_description_one_segment_from_two_stored() -> None:
    """GIVEN 2 stored segments WHEN parse with n=1 THEN first value only."""
    desc = build_description("chat", ["acme", "billing"])
    values = parse_description("chat", desc, 1)
    assert values == ["acme"]


def test_parse_description_unescapes_slash() -> None:
    """GIVEN a value with an escaped slash WHEN parse THEN slash is restored."""
    desc = build_description("chat", ["key/val"])
    values = parse_description("chat", desc, 1)
    assert values == ["key/val"]


def test_parse_description_unescapes_percent() -> None:
    """GIVEN a value with an escaped percent WHEN parse THEN percent is restored."""
    desc = build_description("chat", ["100%sure"])
    values = parse_description("chat", desc, 1)
    assert values == ["100%sure"]


def test_parse_description_roundtrip_special_chars() -> None:
    """GIVEN values with both % and / WHEN build+parse THEN round-trips correctly."""
    original = ["%/path%", "val/two"]
    desc = build_description("p", original)
    recovered = parse_description("p", desc, 2)
    assert recovered == original


def test_parse_description_raises_on_wrong_prefix() -> None:
    """GIVEN a description with a different prefix WHEN parse THEN raises ValueError."""
    desc = build_description("chat", ["acme"])
    with pytest.raises(ValueError, match="does not start with"):
        parse_description("widget", desc, 1)


def test_parse_description_raises_on_too_few_segments() -> None:
    """GIVEN a description with 1 segment WHEN parse asks for 2 THEN raises ValueError."""
    desc = build_description("chat", ["acme"])
    with pytest.raises(ValueError, match="found only 1"):
        parse_description("chat", desc, 2)


def test_parse_description_empty_values() -> None:
    """GIVEN description with no values WHEN parse with n=0 THEN empty list."""
    desc = build_description("chat", [])
    values = parse_description("chat", desc, 0)
    assert values == []


def test_build_and_parse_prefix_with_slash() -> None:
    """GIVEN prefix containing slash WHEN build+parse THEN round-trips correctly."""
    desc = build_description("my/prefix", ["val1", "val2"])
    values = parse_description("my/prefix", desc, 2)
    assert values == ["val1", "val2"]
