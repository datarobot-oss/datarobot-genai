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
#
# The one module of the v0.3 compatibility suite that depends on the SDK's own types.
#
# The tests assert on what a v0.3 caller sees, expressed in ``v03`` types, and build their
# inputs through the helpers below. Upgrading the SDK means re-pointing this file, and nothing
# else in this package: if a test file has to change too, a v0.3 caller would notice the upgrade.
#
# On a2a-sdk 1.x, ``v03`` is the SDK's v0.3 compatibility module, and inputs are built from the
# SDK's native (protobuf) types, the way the 1.x client hands them to dragent.

from typing import Any

from a2a.compat.v0_3 import types as v03
from a2a.helpers import new_text_message as _new_text_message
from a2a.types import AgentCard
from a2a.types import Artifact
from a2a.types import Message
from a2a.types import Part
from a2a.types import Role
from a2a.types import StreamResponse
from a2a.types import Task
from a2a.types import TaskArtifactUpdateEvent
from a2a.types import TaskState
from a2a.types import TaskStatus
from a2a.types import TaskStatusUpdateEvent

from datarobot_genai.dragent.a2a import cards

__all__ = [
    "authorization_code_token_url",
    "card_rpc_url",
    "client_message_event",
    "client_task",
    "client_task_events",
    "new_text_message",
    "parse_card",
    "v03",
]

_TASK_ID = "task-1"
_CONTEXT_ID = "ctx-1"


def new_text_message(text: str, *, context_id: str | None, task_id: str | None) -> Any:
    """Build an agent text message for an executor to enqueue."""
    return _new_text_message(text, context_id=context_id, task_id=task_id)


def parse_card(data: dict[str, Any]) -> AgentCard:
    """Read agent card JSON the way dragent reads one it fetched or cached."""
    return cards.parse_agent_card(data)


def card_rpc_url(card: AgentCard) -> str:
    """Return the URL dragent calls for JSON-RPC on this card."""
    return cards.card_rpc_url(card)


def authorization_code_token_url(card: AgentCard, scheme: str = "oauth2") -> str | None:
    """Return the token URL of the authorization-code flow under *scheme*, if kept."""
    flows = card.security_schemes[scheme].oauth2_security_scheme.flows
    if flows.WhichOneof("flow") != "authorization_code":
        return None
    return str(flows.authorization_code.token_url)


def client_task() -> Any:
    """Build a working task, as the base A2A client returns it from get_task / cancel_task."""
    return Task(
        id=_TASK_ID,
        context_id=_CONTEXT_ID,
        status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
    )


def client_task_events() -> list[Any]:
    """Build the events the base client yields for a task reply, ending in completion."""
    artifact = Artifact(artifact_id="a1", parts=[Part(text="Hello")])
    return [
        StreamResponse(task=client_task()),
        StreamResponse(
            artifact_update=TaskArtifactUpdateEvent(
                task_id=_TASK_ID, context_id=_CONTEXT_ID, artifact=artifact
            )
        ),
        StreamResponse(
            status_update=TaskStatusUpdateEvent(
                task_id=_TASK_ID,
                context_id=_CONTEXT_ID,
                status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
            )
        ),
    ]


def client_message_event(text: str) -> Any:
    """Build a plain agent message, as the base client yields it for a message reply."""
    return StreamResponse(
        message=Message(role=Role.ROLE_AGENT, parts=[Part(text=text)], message_id="m-1")
    )
