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

from collections.abc import Sequence

from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.workflow import Event
from llama_index.llms.litellm import LiteLLM
from ragas import MultiTurnSample
from ragas.integrations.llama_index import convert_to_ragas_messages


class DataRobotLiteLLM(LiteLLM):  # type: ignore[misc]
    """LiteLLM wrapper providing chat/function capability metadata for LlamaIndex."""

    @property
    def metadata(self) -> LLMMetadata:  # noqa: D401
        return LLMMetadata(
            context_window=128000,
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name=self.model,
        )


def create_pipeline_interactions_from_events(
    events: Sequence[Event] | None,
) -> MultiTurnSample | None:
    if not events:
        return None
    ragas_trace = convert_to_ragas_messages(events)
    return MultiTurnSample(user_input=ragas_trace)
