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

from starlette.requests import Request

from .config import get_config
from .constants import OAUTH_PROTECTED_RESOURCE_METADATA_ENDPOINT


def prefix_mount_path(endpoint: str) -> str:
    config = get_config()
    mount_path = config.mount_path

    if mount_path == "/":
        return endpoint

    if mount_path.endswith("/"):
        mount_path = mount_path[:-1]

    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    return mount_path + endpoint


def oauth_protected_resource_metadata_path() -> str:
    """Return the mounted path for the OAuth protected resource metadata document."""
    return prefix_mount_path(OAUTH_PROTECTED_RESOURCE_METADATA_ENDPOINT)


def build_oauth_protected_resource_metadata_url(request: Request) -> str:
    """Build the absolute URL for the OAuth protected resource metadata document."""
    return str(request.url.replace(path=oauth_protected_resource_metadata_path(), query="", fragment=""))
