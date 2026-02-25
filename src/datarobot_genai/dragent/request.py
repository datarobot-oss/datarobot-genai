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

from ag_ui.core import RunAgentInput
from pydantic import ConfigDict
from pydantic.alias_generators import to_camel


class DRAgentRunAgentInput(RunAgentInput):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "threadId": "1",
                "runId": "1",
                "messages": [{"role": "user", "content": "who are you?", "id": "1"}],
                "tools": [],
                "context": [],
                "forwardedProps": {},
                "state": {},
            }
        },
        alias_generator=to_camel,
        populate_by_name=True,
    )
