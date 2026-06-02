# Copyright 2026 DataRobot, Inc.
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

from collections.abc import Iterator
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from datarobot_genai.drtools.core.clients.datarobot import ThreadSafeDataRobotClient


@pytest.fixture
def mock_get_client_context_with_token_from_request_header() -> Iterator[Mock]:
    with patch.object(
        ThreadSafeDataRobotClient,
        "get_client_context_with_token_from_request_header",
    ) as mock_func:
        yield mock_func
