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

import pytest

from datarobot_genai.core.agents import default_usage_metrics
from datarobot_genai.core.agents import is_streaming


def test_default_usage_metrics_has_required_keys_and_zero_values():
    # GIVEN a freshly created metrics dict
    metrics = default_usage_metrics()

    # THEN it has the required keys with zero values
    assert metrics["completion_tokens"] == 0
    assert metrics["prompt_tokens"] == 0
    assert metrics["total_tokens"] == 0


def test_is_streaming_false_by_default():
    # GIVEN no explicit stream flag
    params = {"messages": []}

    # WHEN checking streaming
    # THEN default is False
    assert is_streaming(params) is False


def test_is_streaming_true_when_bool_true():
    # GIVEN a boolean True stream flag
    params = {"stream": True}

    # WHEN checking streaming
    # THEN result is True
    assert is_streaming(params) is True


@pytest.mark.parametrize("value", ["true", "TRUE", "TrUe"])  # GIVEN truthy string stream flag
def test_is_streaming_true_when_string_true_case_insensitive(value):
    params = {"stream": value}

    # WHEN checking streaming
    # THEN result is True
    assert is_streaming(params) is True


@pytest.mark.parametrize("value", ["false", "FALSE", "FaLsE"])  # GIVEN falsy string stream flag
def test_is_streaming_false_when_string_false_case_insensitive(value):
    params = {"stream": value}

    # WHEN checking streaming
    # THEN result is False
    assert is_streaming(params) is False
