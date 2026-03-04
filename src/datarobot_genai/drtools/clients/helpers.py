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
from fastmcp.server.dependencies import get_http_headers


def get_api_key_from_headers(header_name: str) -> str | None:
    if header_name.startswith("x-datarobot-"):
        header_name = header_name[len("x-datarobot-") :]
    if header_name.startswith("x-"):
        header_name = header_name[len("x-") :]

    headers = get_http_headers()

    # Try to get from x-{header}
    if api_key := headers.get(f"x-{header_name}"):
        return api_key

    # Try to get from x-datarobot-{header}
    if api_key := headers.get(f"x-datarobot-{header_name}"):
        return api_key

    return None
