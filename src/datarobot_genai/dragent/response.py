# Copyright 2025 DataRobot, Inc. and its affiliates.
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

from ag_ui.core import Event
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import ResponseBaseModelOutput


class DRAgentEventResponse(ResponseBaseModelOutput):
    event: Event | None = None
    delta: str | None = None
    pipeline_interactions: str | None = None
    usage_metrics: dict[str, int] | None = None


class DRAgentChatResponseChunk(ChatResponseChunk):
    pipeline_interactions: str | None = None
    event: Event | None = None


class DRAgentChatResponse(ChatResponse):
    pipeline_interactions: str | None = None
