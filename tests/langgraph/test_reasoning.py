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

import pytest
from langchain_core.messages import AIMessageChunk

from datarobot_genai.langgraph.reasoning import iter_message_blocks


@pytest.mark.parametrize(
    "message, expected",
    [
        # Plain text content, no reasoning.
        (AIMessageChunk(content="hi"), [("text", "hi")]),
        # OpenAI-compatible flat shape: text in content, reasoning in additional_kwargs.
        # Reasoning is yielded BEFORE text so AG-UI emits REASONING_* before TEXT_*.
        (
            AIMessageChunk(content="say", additional_kwargs={"reasoning_content": "think"}),
            [("thinking", "think"), ("text", "say")],
        ),
        # Native list-form content (Anthropic/Bedrock blocks).
        (
            AIMessageChunk(
                content=[
                    {"type": "thinking", "thinking": "t"},
                    {"type": "text", "text": "say"},
                ],
            ),
            [("thinking", "t"), ("text", "say")],
        ),
        # Pure-reasoning chunk: empty content, only reasoning_content delta.
        (
            AIMessageChunk(content="", additional_kwargs={"reasoning_content": "delta"}),
            [("thinking", "delta")],
        ),
        # Empty additional_kwargs reasoning_content is ignored.
        (
            AIMessageChunk(content="hi", additional_kwargs={"reasoning_content": ""}),
            [("text", "hi")],
        ),
        # Gateway sends the SAME reasoning in BOTH list-form content AND
        # reasoning_content. The flat copy must be dropped (no double-emit).
        (
            AIMessageChunk(
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
    assert list(iter_message_blocks(message)) == expected
