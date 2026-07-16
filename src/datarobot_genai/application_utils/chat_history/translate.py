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

"""Translate stored :class:`Message` records into AG-UI wire messages.

:func:`translate_messages` rebuilds the AG-UI conversation history that is fed
back into an agent at the start of a run.  Each stored message is emitted as one
message, immediately followed by its tool-call *results* (role ``tool``) and then
its reasoning steps (role ``reasoning``) — matching the order a live agent stream
would have produced them.

:class:`ExtendedBaseMessage` widens :class:`ag_ui.core.BaseMessage` with the
extra bookkeeping fields the chat layer tracks (``in_progress`` / ``status`` /
``error`` / ``tool_calls`` / ``tool_call_id`` / ``uuid``).  It relies on the
loose ``extra="allow"`` behaviour of the AG-UI base message, so downstream
consumers that only understand plain AG-UI messages ignore the extra keys.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Iterator

from ag_ui.core import AssistantMessage
from ag_ui.core import BaseMessage
from ag_ui.core import FunctionCall
from ag_ui.core import ToolCall

from datarobot_genai.application_utils.chat_history.models import Message
from datarobot_genai.application_utils.chat_history.models import Role


class ExtendedBaseMessage(BaseMessage):
    """AG-UI base message widened with in-progress, status and error bookkeeping.

    The extra fields are carried through the AG-UI base message's loose
    ``extra="allow"`` configuration; consumers that only understand plain AG-UI
    messages ignore them.

    Attributes
    ----------
    in_progress : bool
        Whether the underlying record is still streaming.
    status : str | None
        Lifecycle status (see :class:`~.models.MessageStatus`), if known.
    error : str | None
        Error text captured for the record, if any.
    tool_calls : list[ag_ui.core.ToolCall] | None
        OpenAI-style tool calls attached to an assistant message.
    tool_call_id : str | None
        For a tool-*result* message, the AG-UI id of the call it answers.
    uuid : str | None
        The internal database identifier, distinct from ``id`` (which carries the
        AG-UI id).  Lets clients call message-scoped endpoints keyed on the uuid.
    """

    in_progress: bool
    status: str | None = None
    error: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    uuid: str | None = None


def translate_messages(messages: Iterable[Message]) -> Iterator[ExtendedBaseMessage]:
    """Transform stored messages into AG-UI messages, oldest first.

    For every message the order is: the message itself, then its tool-call
    results (role ``tool``), then its reasoning steps (role ``reasoning``).
    Messages are ordered by their application ``timestamp``; a message's tool
    calls and reasoning steps are each ordered by their ``created_at``.

    Parameters
    ----------
    messages : Iterable[Message]
        Stored messages (e.g. from
        :meth:`~.repositories.MessageRepository.get_chat_messages`).

    Yields
    ------
    ExtendedBaseMessage
        The AG-UI messages reconstructing the conversation history.
    """
    for message in sorted(messages, key=lambda m: m.timestamp):
        sorted_tool_calls = sorted(message.tool_calls, key=lambda tc: tc.created_at)

        out: BaseMessage
        if message.role == Role.ASSISTANT.value:
            out = AssistantMessage(
                id=message.agui_id or str(message.message_uuid),
                role="assistant",
                content=message.content,
                name=message.name,
                tool_calls=[
                    ToolCall(
                        id=tc.agui_id or str(tc.uuid),
                        function=FunctionCall(name=tc.name, arguments=tc.arguments),
                    )
                    for tc in sorted_tool_calls
                ],
            )
        else:
            out = BaseMessage(
                id=message.agui_id or str(message.message_uuid),
                role=message.role,
                content=message.content,
                name=message.name,
            )
        yield ExtendedBaseMessage(
            **out.model_dump(),
            in_progress=message.in_progress,
            status=message.status,
            error=message.error,
            uuid=str(message.message_uuid),
        )

        # Tool-call *results* follow the message they belong to.  The public
        # message id is the tool_call_id while tool_call_id carries the AG-UI id —
        # the reverse of the assistant message's tool_calls entries above.
        for tc in sorted_tool_calls:
            yield ExtendedBaseMessage(
                id=tc.tool_call_id or str(tc.uuid),
                role=Role.TOOL.value,
                name=tc.name,
                content=tc.content if tc.content else f"Completed {tc.name}",
                tool_call_id=tc.agui_id or str(tc.uuid),
                in_progress=tc.in_progress,
                error=tc.error,
            )

        for reasoning in sorted(message.reasonings, key=lambda r: r.created_at):
            yield ExtendedBaseMessage(
                id=reasoning.agui_id or str(reasoning.uuid),
                role=Role.REASONING.value,
                name=reasoning.name,
                content=reasoning.content,
                in_progress=reasoning.in_progress,
                error=reasoning.error,
            )
