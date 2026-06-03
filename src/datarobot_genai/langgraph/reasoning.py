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
"""LangChain-message reasoning normalization for the LangGraph adapter.

Wraps the framework-agnostic ``iter_content_blocks`` (in ``core``) with the
LangChain ``BaseMessage`` surface so it can type the message directly instead of
duck-typing. Lives in the ``langgraph`` package because it depends on LangChain;
``core`` stays framework-free.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

from langchain_core.messages import BaseMessage

from datarobot_genai.core.agents.reasoning import iter_content_blocks


def iter_message_blocks(
    message: BaseMessage,
) -> Iterator[tuple[Literal["text", "thinking"], str]]:
    """Yield typed (kind, delta) pairs for any AIMessage/AIMessageChunk shape.

    Combines two surfaces LangChain/LiteLLM produce for reasoning models:

    - **Native list-form content**: e.g. Anthropic/Bedrock blocks
      ``[{"type": "thinking", ...}, {"type": "text", ...}]`` — delegated to
      ``iter_content_blocks(message.content)``.
    - **OpenAI-compatible flat shape**: text in ``message.content`` (string) and
      reasoning hoisted to ``message.additional_kwargs["reasoning_content"]``
      (string). This is what the DataRobot LLM gateway returns over its
      OpenAI-style HTTP API, even when the underlying model is Anthropic.

    These shapes are not mutually exclusive: with extended thinking enabled the
    DataRobot gateway emits the *same* reasoning delta in BOTH
    ``additional_kwargs["reasoning_content"]`` and as a native ``content``
    thinking block. To avoid double-emitting, the flat ``reasoning_content`` is
    only used as a fallback when ``content`` does not already carry thinking.

    Reasoning is yielded before content so consumers can route REASONING_*
    events ahead of the matching text in the AG-UI stream.
    """
    content_pairs = list(iter_content_blocks(message.content))
    reasoning_text = message.additional_kwargs.get("reasoning_content")
    if reasoning_text and not any(kind == "thinking" for kind, _ in content_pairs):
        yield "thinking", reasoning_text
    yield from content_pairs
