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

import logging

import pytest

from datarobot_genai.core.agents.reasoning import flatten_to_text
from datarobot_genai.core.agents.reasoning import iter_content_blocks


@pytest.mark.parametrize(
    "content, expected",
    [
        (None, []),
        ("", []),
        ([], []),
        ("hi", [("text", "hi")]),
        (["a", "b"], [("text", "a"), ("text", "b")]),
        (["", "b"], [("text", "b")]),
        ([{"type": "text", "text": "hi"}], [("text", "hi")]),
        ([{"type": "thinking", "thinking": "t"}], [("thinking", "t")]),
        ([{"type": "reasoning", "reasoning": "r"}], [("thinking", "r")]),
        ([{"type": "thinking", "thinking": ""}], []),
        ([{"type": "text", "text": ""}], []),
        (
            [
                {"type": "thinking", "thinking": "think"},
                {"type": "text", "text": "say"},
            ],
            [("thinking", "think"), ("text", "say")],
        ),
        ([{"type": "unknown_future", "value": "x"}], []),
        ([{"no_type_key": "x"}], []),
        (
            [
                {"type": "text", "text": "a"},
                "b",
                {"type": "thinking", "thinking": "t"},
            ],
            [("text", "a"), ("text", "b"), ("thinking", "t")],
        ),
    ],
)
def test_iter_content_blocks(content, expected):
    assert list(iter_content_blocks(content)) == expected


@pytest.mark.parametrize(
    "content, expected",
    [
        (None, ""),
        ("", ""),
        ("hi", "hi"),
        (["a", "b"], "ab"),
        ([{"type": "text", "text": "hi"}], "hi"),
        ([{"type": "thinking", "thinking": "t"}], ""),
        (
            [
                {"type": "thinking", "thinking": "think"},
                {"type": "text", "text": "say"},
            ],
            "say",
        ),
        (
            [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            "ab",
        ),
    ],
)
def test_flatten_to_text(content, expected):
    assert flatten_to_text(content) == expected


def test_malformed_item_raises():
    """A non-str, non-dict list item raises to surface an unexpected content format."""
    with pytest.raises(ValueError, match="Unparseable content item"):
        list(iter_content_blocks([123]))


def test_unrouted_block_type_skips_without_raising(caplog):
    """A recognized-but-unrouted block (e.g. tool_use) is skipped at debug, never raised."""
    with caplog.at_level(logging.DEBUG, logger="datarobot_genai.core.agents.reasoning"):
        assert list(iter_content_blocks([{"type": "tool_use", "name": "search"}])) == []
    assert [r for r in caplog.records if r.levelno == logging.DEBUG]
