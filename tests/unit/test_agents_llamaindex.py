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

from datarobot_genai.agents.llamaindex import DataRobotLiteLLM
from datarobot_genai.agents.llamaindex import create_pipeline_interactions_from_events


def test_datarobot_litellm_metadata_properties() -> None:
    llm = DataRobotLiteLLM(model="dr/model", max_tokens=256)
    meta = llm.metadata

    assert meta.context_window == 128000
    assert meta.num_output == 256
    assert meta.is_chat_model is True
    assert meta.is_function_calling_model is True
    assert meta.model_name == "dr/model"


def test_create_pipeline_interactions_from_events_none() -> None:
    assert create_pipeline_interactions_from_events(None) is None
