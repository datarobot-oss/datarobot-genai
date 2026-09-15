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


"""One definition of "this request is a platform probe", for every layer that must skip one.

Two consumers today -- ``fastapi`` registers these paths and drops their OTel spans,
``claim_validation`` exempts them from the audience check.  Kept here rather than in
``fastapi`` because ``fastapi`` imports ``claim_validation``, so the constant cannot live
in the module that also installs the middleware without a cycle.

The probe is anonymous infrastructure traffic: it cannot obtain an agent-scoped token, so
any layer that rejects unrecognised credentials has to let it through or the workload never
reaches ready state.
"""

from starlette.requests import Request

# Registered by ``fastapi._register_health_routes``; the platform probes all five.
DATAROBOT_EXPECTED_HEALTH_ROUTES = ["/", "/ping", "/ping/", "/health", "/health/"]

# The only methods ``_register_health_routes`` registers (as ``fastapi._GET_AND_HEAD``, which
# cannot be imported here without a cycle), so anything else on those paths is a 405 from the
# router regardless.  Narrowing to them costs no probe and keeps the exemption off a route that
# merely shares a path -- ``/`` is both a health route and, to an A2A app mounted at the root,
# the execute endpoint.  If the health routes ever take another method, a probe using it is
# checked rather than wrongly exempted, so the two drifting fails safe.
_PROBE_METHODS = frozenset({"GET", "HEAD"})


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


def is_probe_request(request: Request) -> bool:
    """Whether this request is a liveness/readiness probe of one of the health routes.

    Matches the same unprefixed path the router matches, so this is true only when the request
    would reach ``health_check`` -- a prefixed probe (``/<model_id>/<lrs_id>/health``, or the
    bare deployment root) resolves through ``route_path`` just as routing does, while a path
    that merely ends in ``/health`` under some other prefix keeps that prefix and does not
    match.  Nothing outside the five registered routes is exempted by accident.
    """
    return request.method.upper() in _PROBE_METHODS and (
        route_path(request) in DATAROBOT_EXPECTED_HEALTH_ROUTES
    )
