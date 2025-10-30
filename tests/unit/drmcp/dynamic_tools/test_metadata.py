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

import pytest

from datarobot_genai.drmcp.core.dynamic_tools.deployment.adapters.drum import is_drum


@pytest.fixture
def drum_metadata():
    return {
        "drum_server": "server",
        "drum_version": "1.18.0",
        "target_type": "unstructured",
        "model_metadata": {
            "name": "model-name",
            "description": "model-description",
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }


@pytest.fixture
def non_drum_metadata():
    return {
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "endpoint": "/predict",
    }


def test_is_drum_true(drum_metadata):
    assert is_drum(drum_metadata) is True


def test_is_drum_false(non_drum_metadata):
    assert is_drum(non_drum_metadata) is False
