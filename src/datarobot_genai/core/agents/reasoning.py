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
"""Framework-agnostic normalization of reasoning/thinking content.

Reasoning models (Claude extended thinking, Qwen, OpenAI o1, GPT-OSS, DeepSeek)
expose reasoning in shapes that recur across LangChain/LiteLLM-based adapters:

- native **list-form content blocks** — ``[{"type": "thinking", ...}, {"type": "text", ...}]``
  (Anthropic/Bedrock SDK pass-through), and
- the **OpenAI-compatible flat shape** — text on ``content`` with reasoning hoisted to
  ``additional_kwargs["reasoning_content"]`` (DataRobot LLM gateway and other proxies).

These helpers normalize both into typed ``(kind, delta)`` pairs so each framework
adapter can route thinking to AG-UI Reasoning events and text to text events without
re-implementing the parsing. They live in ``core`` (rather than a single framework
package) so independent adapters can share them without leaf-to-leaf imports. The
functions are duck-typed and carry no framework dependency.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any
from typing import Literal

logger = logging.getLogger(__name__)


def _iter_content_blocks(
    content: str | list[Any] | None,
) -> Iterator[tuple[Literal["text", "thinking"], str]]:
    """Yield typed (kind, delta) pairs for any AIMessage/ToolMessage content shape.

    Normalizes the union of shapes LangChain/LiteLLM produce for ``AIMessage.content``:
    plain string, list of strings, or list of structured blocks. Thinking and
    reasoning blocks both map to ``("thinking", delta)``. Empty deltas are filtered
    so callers never need to emit zero-content events. Unknown block shapes are
    skipped with a debug log to keep the stream resilient.
    """
    if not content:
        return
    if isinstance(content, str):
        yield "text", content
        return
    for block in content:
        if isinstance(block, str):
            if block:
                yield "text", block
            continue
        if not isinstance(block, dict):
            logger.debug("Skipping unknown content item: %r", block)
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text", "")
            if text:
                yield "text", text
        elif block_type == "thinking":
            thinking = block.get("thinking", "")
            if thinking:
                yield "thinking", thinking
        elif block_type == "reasoning":
            reasoning = block.get("reasoning", "")
            if reasoning:
                yield "thinking", reasoning
        else:
            logger.debug("Skipping unknown content block type: %r", block_type)


def _flatten_to_text(content: str | list[Any] | None) -> str:
    """Collapse any content shape to its text portion, dropping thinking blocks."""
    return "".join(delta for kind, delta in _iter_content_blocks(content) if kind == "text")


def _iter_message_blocks(
    message: Any,
) -> Iterator[tuple[Literal["text", "thinking"], str]]:
    """Yield typed (kind, delta) pairs for any AIMessage/AIMessageChunk shape.

    Combines two surfaces LangChain/LiteLLM produce for reasoning models:

    - **Native list-form content**: e.g. Anthropic/Bedrock blocks
      ``[{"type": "thinking", ...}, {"type": "text", ...}]`` — delegated to
      ``_iter_content_blocks(message.content)``.
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
    content_pairs = list(_iter_content_blocks(getattr(message, "content", None)))
    ak = getattr(message, "additional_kwargs", None) or {}
    reasoning_text = ak.get("reasoning_content")
    if reasoning_text and not any(kind == "thinking" for kind, _ in content_pairs):
        yield "thinking", reasoning_text
    yield from content_pairs
