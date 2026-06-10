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

"""Base utilities for building FastMCP servers on DataRobot."""

from datarobot_genai.drmcpbase.middleware import OAuthMiddleWare
from datarobot_genai.drmcpbase.middleware import RequestHeadersMiddleware
from datarobot_genai.drmcpbase.middleware import read_http_headers
from datarobot_genai.drmcpbase.middleware import register_oauth_middleware

__all__ = [
    "OAuthMiddleWare",
    "RequestHeadersMiddleware",
    "read_http_headers",
    "register_oauth_middleware",
]
