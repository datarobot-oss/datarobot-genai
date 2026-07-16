# Copyright 2025 DataRobot, Inc.
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
"""Pipeline-interaction primitives for agent runs.

Each framework agent records its run as a ``MultiTurnSample``: an ordered list of
``HumanMessage`` / ``AIMessage`` / ``ToolMessage`` turns. The chat layer serialises
that sample to JSON and ships it as the ``pipeline_interactions`` field, which
DataRobot moderations reads back to score the run (e.g. agent goal accuracy).

These types used to come from ``ragas`` (``ragas.MultiTurnSample`` and
``ragas.messages``). ragas is an unmaintained package with a large, CVE-heavy
dependency chain, and we only ever used it to build and JSON-serialise this
payload. datarobot-moderations now ships dependency-light equivalents of the
message primitives, so we reuse those and keep only a slim, locally-owned
``MultiTurnSample`` here. That lets us drop the ragas dependency entirely.
"""

from __future__ import annotations

from typing import Any
from typing import TypeAlias

# Reuse the conversation-message primitives owned and exported by
# datarobot-moderations. They are plain pydantic models (content/type/metadata,
# plus tool_calls on AIMessage) that match the shape moderations reads back, so
# the serialised payload stays byte-for-byte compatible with the old ragas one
# apart from a few always-null fields we no longer emit.
from datarobot_dome.guards.agent_goal_accuracy import AIMessage
from datarobot_dome.guards.agent_goal_accuracy import HumanMessage
from datarobot_dome.guards.agent_goal_accuracy import ToolCall
from datarobot_dome.guards.agent_goal_accuracy import ToolMessage
from pydantic import BaseModel

# Plain (non-discriminated) union, matching the old ragas ``MultiTurnSample``.
# We only ever *build and serialise* a sample, never parse one back, so we don't
# need the discriminator moderations uses for round-trip decoding -- and a
# discriminated union would reject the type-less AG-UI message dicts the base
# agent may hand us. Pydantic's smart union keeps the old behaviour: a dict
# without a ``type`` field validates as a ``HumanMessage`` (its ``type`` default).
Message: TypeAlias = HumanMessage | AIMessage | ToolMessage


class MultiTurnSample(BaseModel):
    """A multi-turn conversation between the human, the agent and its tools.

    Slim replacement for ragas' ``MultiTurnSample``. The only field the downstream
    moderations consumer reads is ``user_input``; ``reference`` is kept for parity
    with the moderations model.
    """

    user_input: list[Message]
    reference: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict, dropping unset/None fields."""
        return self.model_dump(exclude_none=True)


__all__ = [
    "AIMessage",
    "HumanMessage",
    "ToolMessage",
    "ToolCall",
    "Message",
    "MultiTurnSample",
]
