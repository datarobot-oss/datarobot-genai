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

from datarobot_dome.guards.agent_goal_accuracy import AIMessage
from datarobot_dome.guards.agent_goal_accuracy import HumanMessage
from datarobot_dome.guards.agent_goal_accuracy import ToolMessage
from pydantic import BaseModel


class MultiTurnSample(BaseModel):
    """A multi-turn conversation between the human, the agent and its tools.

    Slim replacement for ragas' ``MultiTurnSample``. The only field the downstream
    moderations consumer reads is ``user_input``; ``reference`` is kept for parity
    with the moderations model.
    """

    # Plain (non-discriminated) union, matching the old ragas ``MultiTurnSample``.
    user_input: list[HumanMessage | AIMessage | ToolMessage]
    reference: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict, dropping unset/None fields."""
        return self.model_dump(exclude_none=True)
