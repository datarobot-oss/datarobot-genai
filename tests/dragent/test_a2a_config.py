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

import os
from unittest.mock import patch

from datarobot_genai.dragent.a2a_config import A2AConfig


class TestA2AConfig:
    def test_expose_a2a_server_endpoints_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EXPOSE_A2A_SERVER_ENDPOINTS", None)
            assert A2AConfig().expose_a2a_server_endpoints is True

    def test_expose_a2a_server_endpoints_disabled(self):
        with patch.dict(os.environ, {"EXPOSE_A2A_SERVER_ENDPOINTS": "false"}):
            assert A2AConfig().expose_a2a_server_endpoints is False

