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
# Today ``v03`` is the SDK's own type module, because the SDK speaks v0.3 natively.

from typing import Any

from a2a import types as v03
from a2a.types import AgentCard
from a2a.utils import new_agent_text_message

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
    return new_agent_text_message(text, context_id=context_id, task_id=task_id)


def parse_card(data: dict[str, Any]) -> AgentCard:
    """Read agent card JSON the way dragent reads one it fetched or cached."""
    return AgentCard.model_validate(data)


def card_rpc_url(card: AgentCard) -> str:
    """Return the URL dragent calls for JSON-RPC on this card."""
    return card.url


def authorization_code_token_url(card: AgentCard, scheme: str = "oauth2") -> str | None:
    """Return the token URL of the authorization-code flow under *scheme*, if kept."""
    flows = card.security_schemes[scheme].root.flows  # type: ignore[index]
    return flows.authorization_code.token_url if flows.authorization_code else None


def client_task() -> Any:
    """Build a working task, as the base A2A client returns it from get_task / cancel_task."""
    return v03.Task(
        id=_TASK_ID, context_id=_CONTEXT_ID, status=v03.TaskStatus(state=v03.TaskState.working)
    )


def client_task_events() -> list[Any]:
    """Build the events the base client yields for a task reply, ending in completion."""
    artifact = v03.Artifact(artifact_id="a1", parts=[v03.Part(root=v03.TextPart(text="Hello"))])
    working = client_task()
    with_artifact = working.model_copy(update={"artifacts": [artifact]})
    completed = with_artifact.model_copy(
        update={"status": v03.TaskStatus(state=v03.TaskState.completed)}
    )
    return [
        (working, None),
        (
            with_artifact,
            v03.TaskArtifactUpdateEvent(
                task_id=_TASK_ID, context_id=_CONTEXT_ID, artifact=artifact
            ),
        ),
        (
            completed,
            v03.TaskStatusUpdateEvent(
                task_id=_TASK_ID,
                context_id=_CONTEXT_ID,
                status=v03.TaskStatus(state=v03.TaskState.completed),
                final=True,
            ),
        ),
    ]


def client_message_event(text: str) -> Any:
    """Build a plain agent message, as the base client yields it for a message reply."""
    return v03.Message(
        role=v03.Role.agent, parts=[v03.Part(root=v03.TextPart(text=text))], message_id="m-1"
    )
