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


# Try to import get_http_headers from FastMCP if available
# The deplyment is expected to have the fastmcp dependency installed.
# If not, you need to add your own implementation of get_http_headers.
try:
    from fastmcp.server.dependencies import get_http_headers

    _get_http_headers = get_http_headers
except ImportError:
    # FastMCP not available - create a stub that returns empty dict
    def _get_http_headers(include_all: bool = False) -> dict[str, str]:
        """
        Stub implementation when FastMCP is not available.
        Returns empty dict to match FastMCP behavior when no request context exists.
        """
        return {}

    get_http_headers = _get_http_headers


def get_api_key_from_headers(header_name: str) -> str | None:
    headers = _get_http_headers()

    # Normalize header name to lowercase for consistent lookup
    header_name = header_name.lower()
    candidates = [header_name]

    if header_name.startswith("x-") and not header_name.startswith("x-datarobot-"):
        candidates.append(f"x-datarobot-{header_name[2:]}")
    elif not header_name.startswith("x-datarobot-"):
        candidates.append(f"x-datarobot-{header_name}")

    for name in candidates:
        if value := headers.get(name):
            return value

    return None
