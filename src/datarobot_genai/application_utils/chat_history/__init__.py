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

"""Chat-history layer over the Memory Service persistence ORM.

Public API re-exports for the chat models, DTOs and low-level helpers.  This
package extends :mod:`datarobot_genai.application_utils.persistence`; the
persistence sub-package itself never imports from here (or from ``ag_ui``).
"""

from datarobot_genai.application_utils.chat_history.constants import LOCATOR_KIND_CHAT
from datarobot_genai.application_utils.chat_history.constants import LOCATOR_KIND_MESSAGE
from datarobot_genai.application_utils.chat_history.constants import LOCATOR_KIND_REASONING
from datarobot_genai.application_utils.chat_history.constants import LOCATOR_KIND_TOOL_CALL
from datarobot_genai.application_utils.chat_history.constants import MEMORY_CHAT_MESSAGE_EVENT_TYPE
from datarobot_genai.application_utils.chat_history.constants import PAYLOAD_VERSION
from datarobot_genai.application_utils.chat_history.constants import app_str
from datarobot_genai.application_utils.chat_history.constants import chat_deduplication_key
from datarobot_genai.application_utils.chat_history.constants import emitter_for_role
from datarobot_genai.application_utils.chat_history.constants import locator_key
from datarobot_genai.application_utils.chat_history.constants import normalize_participant_id
from datarobot_genai.application_utils.chat_history.constants import participant_id
from datarobot_genai.application_utils.chat_history.constants import session_deduplication_key
from datarobot_genai.application_utils.chat_history.constants import wire_non_empty_str
from datarobot_genai.application_utils.chat_history.models import Chat
from datarobot_genai.application_utils.chat_history.models import ChatCreate
from datarobot_genai.application_utils.chat_history.models import EntityLocator
from datarobot_genai.application_utils.chat_history.models import Message
from datarobot_genai.application_utils.chat_history.models import MessageCreate
from datarobot_genai.application_utils.chat_history.models import MessagePublic
from datarobot_genai.application_utils.chat_history.models import MessageReasoningCreate
from datarobot_genai.application_utils.chat_history.models import MessageReasoningUpdate
from datarobot_genai.application_utils.chat_history.models import MessageStatus
from datarobot_genai.application_utils.chat_history.models import MessageToolCallCreate
from datarobot_genai.application_utils.chat_history.models import MessageToolCallUpdate
from datarobot_genai.application_utils.chat_history.models import MessageUpdate
from datarobot_genai.application_utils.chat_history.models import Reasoning
from datarobot_genai.application_utils.chat_history.models import Role
from datarobot_genai.application_utils.chat_history.models import ToolCall
from datarobot_genai.application_utils.chat_history.repositories import ChatRepository
from datarobot_genai.application_utils.chat_history.repositories import ChatRepositoryLike
from datarobot_genai.application_utils.chat_history.repositories import ChatSessionRegistry
from datarobot_genai.application_utils.chat_history.repositories import LocatorIndex
from datarobot_genai.application_utils.chat_history.repositories import MessageRepository
from datarobot_genai.application_utils.chat_history.repositories import MessageRepositoryLike

__all__ = [
    # constants / helpers
    "MEMORY_CHAT_MESSAGE_EVENT_TYPE",
    "PAYLOAD_VERSION",
    "app_str",
    "wire_non_empty_str",
    "session_deduplication_key",
    "chat_deduplication_key",
    "normalize_participant_id",
    "participant_id",
    "emitter_for_role",
    "locator_key",
    "LOCATOR_KIND_CHAT",
    "LOCATOR_KIND_MESSAGE",
    "LOCATOR_KIND_TOOL_CALL",
    "LOCATOR_KIND_REASONING",
    # enums
    "Role",
    "MessageStatus",
    # nested models
    "ToolCall",
    "Reasoning",
    # ORM models
    "Chat",
    "Message",
    "EntityLocator",
    # DTOs
    "ChatCreate",
    "MessageCreate",
    "MessageUpdate",
    "MessageToolCallCreate",
    "MessageToolCallUpdate",
    "MessageReasoningCreate",
    "MessageReasoningUpdate",
    "MessagePublic",
    # repositories
    "ChatRepositoryLike",
    "MessageRepositoryLike",
    "ChatSessionRegistry",
    "LocatorIndex",
    "ChatRepository",
    "MessageRepository",
]
