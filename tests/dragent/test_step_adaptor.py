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

import pytest
from nat.builder.context import IntermediateStepType
from nat.data_models.step_adaptor import StepAdaptorConfig
from nat.data_models.step_adaptor import StepAdaptorMode

from datarobot_genai.dragent.step_adaptor import DRAgentNestedReasoningStepAdaptor


@pytest.mark.parametrize(
    "config",
    [
        StepAdaptorConfig(
            mode=StepAdaptorMode.CUSTOM, custom_event_types=[IntermediateStepType.CUSTOM_START]
        ),
        StepAdaptorConfig(mode=StepAdaptorMode.OFF),
    ],
)
def test_step_adaptor_init_fails_with_non_default_config(config):
    with pytest.raises(ValueError):
        DRAgentNestedReasoningStepAdaptor(config)


@pytest.mark.parametrize(
    "config",
    [
        StepAdaptorConfig(mode=StepAdaptorMode.DEFAULT),
        StepAdaptorConfig(),
    ],
)
def test_step_adaptor_init_succeeds_with_default_config(config):
    adaptor = DRAgentNestedReasoningStepAdaptor(config)
    assert adaptor.config == StepAdaptorConfig()
