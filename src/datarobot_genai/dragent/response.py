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
from ag_ui.core import ReasoningMessageContentEvent
from ag_ui.core import TextMessageContentEvent
from nat.data_models.api_server import GlobalTypeConverter
from nat.data_models.api_server import ResponseBaseModelOutput


class DRAgentEventResponse(ResponseBaseModelOutput):
    events: list[Event] | None = None
    model: str | None = None
    delta: str | None = None
    usage_metrics: dict[str, int] | None = None

    def get_delta(self) -> str:
        if self.delta is not None:
            return self.delta
        if self.events is not None:
            return "\n".join([event.delta for event in self.events_with_delta()])
        return ""

    def events_with_delta(self) -> list[Event]:
        if self.events is not None:
            return [
                event
                for event in self.events
                if isinstance(event, (TextMessageContentEvent, ReasoningMessageContentEvent))
            ]
        return []


def _convert_event_response_to_str(response: DRAgentEventResponse) -> str:
    return response.get_delta()


GlobalTypeConverter.register_converter(_convert_event_response_to_str)
