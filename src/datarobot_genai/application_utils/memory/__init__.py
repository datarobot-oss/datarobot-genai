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

"""Memory Service light ORM — public surface.

Install with ``pip install datarobot-genai[application-utils]``.

Quick start
-----------
.. code-block:: python

    import asyncio
    from typing import Annotated
    from datarobot_genai.application_utils.memory import (
        DRMemorySpace,
        DRSession,
        DREvent,
        DRDeduplicationKey,
        DRRangeKey,
        DRConcurrencyField,
        MemoryServiceClient,
        SYSTEM_PARTICIPANT,
    )

    class ChatSession(DRSession):
        __description_prefix__ = "chat"
        tenant: Annotated[str, DRRangeKey]
        chat_id: Annotated[str, DRDeduplicationKey]
        rev: Annotated[int, DRConcurrencyField]
        title: str  # -> metadata

    class ChatMessage(DREvent, session=ChatSession):
        __event_type__ = "message"
        score: float

    async def main() -> None:
        async with MemoryServiceClient() as client:
            space = await DRMemorySpace.post(client, description="my-space")
            session = await ChatSession.post(space, tenant="acme", chat_id="c1", title="Hello")
            msg = await ChatMessage.post(
                session=session, content="Hi", emitter_type="agent", score=0.9
            )
            print(msg.sequence_id)

    asyncio.run(main())
"""

from datarobot_genai.application_utils.memory._client import MemoryServiceClient
from datarobot_genai.application_utils.memory.event import DREvent
from datarobot_genai.application_utils.memory.exceptions import MemoryBadRequestError
from datarobot_genai.application_utils.memory.exceptions import MemoryConflictError
from datarobot_genai.application_utils.memory.exceptions import MemoryNotFoundError
from datarobot_genai.application_utils.memory.exceptions import MemoryServiceError
from datarobot_genai.application_utils.memory.exceptions import MemoryValidationError
from datarobot_genai.application_utils.memory.exceptions import MemoryVersionConflictError
from datarobot_genai.application_utils.memory.markers import SYSTEM_PARTICIPANT
from datarobot_genai.application_utils.memory.markers import DRConcurrencyField
from datarobot_genai.application_utils.memory.markers import DRDeduplicationKey
from datarobot_genai.application_utils.memory.markers import DRRangeKey
from datarobot_genai.application_utils.memory.session import DRSession
from datarobot_genai.application_utils.memory.space import DRMemorySpace

__all__ = [
    "DRMemorySpace",
    "DRSession",
    "DREvent",
    "DRDeduplicationKey",
    "DRRangeKey",
    "DRConcurrencyField",
    "MemoryServiceClient",
    "SYSTEM_PARTICIPANT",
    "MemoryServiceError",
    "MemoryNotFoundError",
    "MemoryBadRequestError",
    "MemoryValidationError",
    "MemoryConflictError",
    "MemoryVersionConflictError",
]
