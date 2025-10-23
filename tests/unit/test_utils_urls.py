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

from datarobot_genai.utils.urls import get_api_base


@pytest.mark.parametrize(
    "api_base, deployment_id, expected",
    [
        (
            "https://example.com/api/v2",
            "dep-123",
            "https://example.com/api/v2/deployments/dep-123/chat/completions",
        ),
        (
            "https://example.com/api/v2/",
            "dep-123",
            "https://example.com/api/v2/deployments/dep-123/chat/completions",
        ),
        (
            "https://example.com/",
            "dep-123",
            "https://example.com/api/v2/deployments/dep-123/chat/completions",
        ),
        (
            "https://example.com",
            None,
            "https://example.com/",
        ),
        (
            "https://example.com/custom/base/",
            None,
            "https://example.com/custom/base/",
        ),
        (
            "https://example.com/custom/base",
            "dep-123",
            "https://example.com/custom/base/api/v2/deployments/dep-123/chat/completions",
        ),
        (
            "https://example.com/api/v2/deployments/dep-1/chat/completions",
            None,
            "https://example.com/api/v2/deployments/dep-1/chat/completions",
        ),
        (
            "https://example.com/genai",
            None,
            "https://example.com/genai/",
        ),
    ],
)
def test_get_api_base(api_base: str, deployment_id: str | None, expected: str) -> None:
    assert get_api_base(api_base, deployment_id) == expected
