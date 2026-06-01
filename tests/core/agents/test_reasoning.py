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

from types import SimpleNamespace

import pytest

from datarobot_genai.core.agents.reasoning import _flatten_to_text
from datarobot_genai.core.agents.reasoning import _iter_content_blocks
from datarobot_genai.core.agents.reasoning import _iter_message_blocks


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
    assert list(_iter_content_blocks(content)) == expected


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
    assert _flatten_to_text(content) == expected


@pytest.mark.parametrize(
    "message, expected",
    [
        # Plain text content, no reasoning.
        (SimpleNamespace(content="hi", additional_kwargs={}), [("text", "hi")]),
        # OpenAI-compatible flat shape: text in content, reasoning in additional_kwargs.
        # Reasoning is yielded BEFORE text so AG-UI emits REASONING_* before TEXT_*.
        (
            SimpleNamespace(content="say", additional_kwargs={"reasoning_content": "think"}),
            [("thinking", "think"), ("text", "say")],
        ),
        # Native list-form content (Anthropic/Bedrock blocks).
        (
            SimpleNamespace(
                content=[
                    {"type": "thinking", "thinking": "t"},
                    {"type": "text", "text": "say"},
                ],
                additional_kwargs={},
            ),
            [("thinking", "t"), ("text", "say")],
        ),
        # Pure-reasoning chunk: empty content, only reasoning_content delta.
        (
            SimpleNamespace(content="", additional_kwargs={"reasoning_content": "delta"}),
            [("thinking", "delta")],
        ),
        # Empty additional_kwargs reasoning_content is ignored.
        (
            SimpleNamespace(content="hi", additional_kwargs={"reasoning_content": ""}),
            [("text", "hi")],
        ),
        # Object without an additional_kwargs attribute at all: no crash.
        (SimpleNamespace(content="hi"), [("text", "hi")]),
        # Gateway sends the SAME reasoning in BOTH list-form content AND
        # reasoning_content. The flat copy must be dropped (no double-emit).
        (
            SimpleNamespace(
                content=[
                    {"type": "thinking", "thinking": "t"},
                    {"type": "text", "text": "say"},
                ],
                additional_kwargs={"reasoning_content": "t"},
            ),
            [("thinking", "t"), ("text", "say")],
        ),
    ],
)
def test_iter_message_blocks(message, expected):
    assert list(_iter_message_blocks(message)) == expected
