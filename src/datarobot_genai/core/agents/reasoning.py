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
"""Framework-agnostic normalization of reasoning/thinking content blocks.

Reasoning models (Claude extended thinking, Qwen, OpenAI o1, GPT-OSS, DeepSeek)
expose reasoning as native **list-form content blocks** —
``[{"type": "thinking", ...}, {"type": "text", ...}]`` (Anthropic/Bedrock SDK
pass-through). These helpers normalize that (and plain-string or list content)
into typed ``(kind, delta)`` pairs so each framework adapter can route thinking
to AG-UI Reasoning events and text to text events without re-implementing the
parsing. They live in ``core`` (rather than a single framework package) so
independent adapters can share them without leaf-to-leaf imports, and they carry
no framework dependency.

The OpenAI-compatible flat shape (reasoning hoisted to
``additional_kwargs["reasoning_content"]``) is framework-specific and handled by
the per-adapter wrappers, e.g. ``datarobot_genai.langgraph.reasoning``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any
from typing import Literal

logger = logging.getLogger(__name__)


def iter_content_blocks(
    content: str | list[Any] | None,
) -> Iterator[tuple[Literal["text", "thinking"], str]]:
    """Yield typed (kind, delta) pairs for any AIMessage/ToolMessage content shape.

    Normalizes the union of shapes LangChain/LiteLLM produce for ``AIMessage.content``:
    plain string, list of strings, or list of structured blocks. Thinking and
    reasoning blocks both map to ``("thinking", delta)``. Empty deltas are filtered
    so callers never need to emit zero-content events.

    A list item that is neither a string nor a block dict is malformed for every
    framework we support, so it raises ``ValueError`` to surface the unexpected
    format immediately rather than silently dropping content. A well-formed block
    whose ``type`` we do not route (e.g. ``tool_use``, which agents handle via
    their own tool-call path) is skipped at ``debug``: those are recognized,
    non-renderable blocks, not format errors.
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
            # Neither a string nor a block dict: malformed for every framework we
            # support. Fail fast so an unexpected content format surfaces here
            # instead of being silently dropped.
            raise ValueError(f"Unparseable content item (expected str or dict): {block!r}")
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
            # Blocks we recognize but don't route here (e.g. tool_use) arrive
            # routinely via flatten_to_text on tool-calling messages, so this
            # stays at debug — warning/raising would fire on healthy runs.
            logger.debug("Skipping block type: %r", block_type)


def flatten_to_text(content: str | list[Any] | None) -> str:
    """Collapse any content shape to its text portion, dropping thinking blocks."""
    return "".join(delta for kind, delta in iter_content_blocks(content) if kind == "text")


# AG-UI ``AssistantMessage`` has no reasoning field, so prior-turn reasoning is folded
# into the assistant ``content`` as a plain-text sentinel block by ``wrap_reasoning``
# (used during history extraction); the sentinels are defined once here.
REASONING_OPEN = "<reasoning>"
REASONING_CLOSE = "</reasoning>"


def wrap_reasoning(reasoning: str, content: str | None) -> str:
    """Prepend ``reasoning`` to ``content`` as a ``<reasoning>...</reasoning>`` block.

    Returns the reasoning block followed by the answer text. ``content`` may be ``None``
    or empty (e.g. a tool-call-only assistant turn), in which case only the reasoning
    block remains (trailing whitespace stripped).
    """
    return f"{REASONING_OPEN}\n{reasoning}\n{REASONING_CLOSE}\n{content or ''}".rstrip()
