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

"""Stub DataRobot credentials for integration/ETE runs that use MCP client stubs (no real API)."""

import os

from datarobot_genai.drtools.core.constants import DEFAULT_DATAROBOT_ENDPOINT

STUB_DATAROBOT_API_TOKEN = "test-token"


def apply_stub_datarobot_credentials_env() -> None:
    """Set DATAROBOT_* env only when unset (does not override a developer's real token)."""
    os.environ.setdefault("DATAROBOT_API_TOKEN", STUB_DATAROBOT_API_TOKEN)
    os.environ.setdefault("DATAROBOT_ENDPOINT", DEFAULT_DATAROBOT_ENDPOINT)
