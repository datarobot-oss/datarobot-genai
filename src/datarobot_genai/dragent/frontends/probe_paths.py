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


"""Path handling shared by the routes the platform's readiness probe hits.

Registered by ``fastapi._register_health_routes``, which also uses ``route_path`` to drop
their OTel spans.
"""

from starlette.requests import Request

# The platform probes all five.
DATAROBOT_EXPECTED_HEALTH_ROUTES = ["/", "/ping", "/ping/", "/health", "/health/"]


def route_path(request: Request) -> str:
    """Request path relative to the ASGI ``root_path`` the app is mounted under.

    In a DataRobot deployment the server runs with ``--root_path /<model_id>/<lrs_id>`` and the
    LRS ingress forwards the full, prefixed path. Since Starlette 0.33 ``scope["path"]`` (and so
    ``request.url.path``) includes that prefix, so comparing against the unprefixed route paths
    NAT registers requires stripping ``root_path`` first.
    """
    path: str = request.scope["path"]
    root_path: str = request.scope.get("root_path", "").rstrip("/")
    if root_path and path.startswith(root_path):
        return path[len(root_path) :] or "/"
    return path
