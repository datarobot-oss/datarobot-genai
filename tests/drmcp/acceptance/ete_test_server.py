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

import sys
from pathlib import Path

# Add project root to path so imports work when running as a script
# This must be done before any other imports
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from datarobot_genai.drmcp import create_mcp_server  # noqa: E402

# Import to register test tools with the MCP server
# Import integration test tools if they exist (for integration tests)
from datarobot_genai.drmcp.test_utils.elicitation_test_tool import (  # noqa: E402
    get_user_greeting,  # noqa: F401
)
from tests.drmcp.acceptance.test_tools import get_auth_context_user_info  # noqa: E402, F401

if __name__ == "__main__":
    server = create_mcp_server()
    server.run(show_banner=True)
