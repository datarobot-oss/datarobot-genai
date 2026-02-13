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

from datarobot_genai.core.agents import default_usage_metrics


def test_default_usage_metrics_has_required_keys_and_zero_values() -> None:
    # GIVEN a freshly created metrics dict
    metrics = default_usage_metrics()

    # THEN it has the required keys with zero values
    assert metrics["completion_tokens"] == 0
    assert metrics["prompt_tokens"] == 0
    assert metrics["total_tokens"] == 0
