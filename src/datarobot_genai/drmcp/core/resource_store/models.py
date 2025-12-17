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

"""Core data models for ResourceStore."""

from datetime import datetime
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import Field

# Type definitions
ScopeType = Literal["conversation", "memory", "resource", "custom"]
Lifetime = Literal["ephemeral", "persistent"]


class Scope(BaseModel):
    """Scope for organizing resources."""

    type: ScopeType = Field(
        ..., description="Type of scope: conversation, memory, resource, or custom"
    )
    id: str = Field(..., description="Scope identifier (conversationId, userId, agentId, etc.)")


class Resource(BaseModel):
    """Unified resource model for all storage types."""

    id: str = Field(..., description="Unique resource identifier")
    scope: Scope = Field(..., description="Scope this resource belongs to")
    kind: str = Field(
        ...,
        description="Resource kind: 'message', 'tool-call', 'tool-result', 'note', 'blob', etc.",
    )
    lifetime: Lifetime = Field(..., description="Resource lifetime: 'ephemeral' or 'persistent'")
    createdAt: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")  # noqa: N815
    ttlSeconds: int | None = Field(  # noqa: N815
        None, description="Time-to-live in seconds for ephemeral resources"
    )
    contentType: str = Field(  # noqa: N815
        ...,
        description="MIME type: 'application/json', 'text/markdown', etc.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (tags, version, embeddings, etc.)"
    )  # noqa: E501
    contentRef: str = Field(  # noqa: N815
        ..., description="Reference to content: file path, blob key, DB pointer, etc."
    )

    class Config:
        """Pydantic config."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
