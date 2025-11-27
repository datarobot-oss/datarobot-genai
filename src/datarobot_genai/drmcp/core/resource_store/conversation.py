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

"""Conversation State front door for ResourceStore."""

import json
import logging
from typing import Any

from .models import Scope
from .store import ResourceStore

logger = logging.getLogger(__name__)


class ConversationState:
    """Internal API for managing conversation state."""

    def __init__(self, store: ResourceStore) -> None:
        self.store = store

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> str:
        scope = Scope(type="conversation", id=conversation_id)
        message_data: dict[str, Any] = {"role": role, "content": content}
        if tool_calls:
            message_data["tool_calls"] = tool_calls

        resource = await self.store.put(
            scope=scope,
            kind="message",
            data=json.dumps(message_data),
            lifetime="ephemeral",
            contentType="application/json",
            ttlSeconds=86400,
            metadata={"role": role},
        )
        return resource.id

    async def get_history(
        self,
        conversation_id: str,
        max_messages: int | None = None,
    ) -> list[dict[str, Any]]:
        scope = Scope(type="conversation", id=conversation_id)
        resources = await self.store.query(scope=scope)
        resources.sort(key=lambda r: r.createdAt)
        messages = []
        for resource in resources:
            if resource.kind == "message":
                result = await self.store.get(resource.id)
                if result:
                    _, data = result
                    if isinstance(data, str):
                        messages.append(json.loads(data))
        if max_messages:
            messages = messages[-max_messages:]
        return messages

    async def clear_history(self, conversation_id: str) -> None:
        scope = Scope(type="conversation", id=conversation_id)
        resources = await self.store.query(scope=scope)
        for resource in resources:
            await self.store.delete(resource.id)

