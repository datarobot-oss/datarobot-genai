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


"""Rejects an inbound IdP token whose ``aud`` claim does not name this agent.

The DataRobot API Gateway already validated that the token is well-formed, signed and trusted,
then forwarded it as-is.  This answers what is left: was it issued for *us*?

Not an authentication check.  A request with no IdP token passes through -- an agent may be
called with a DataRobot API token instead (see ``a2a.py``'s ``bearerAuth`` scheme), and whether
a caller is authenticated is the gateway's business.  Signature, issuer and expiry are not
re-verified either; claims are decoded unverified and only read.

Covers every route with no exemptions, ``/a2a`` and agent-card discovery included: NAT copies
inbound headers into the workflow context on every route, and the cross-application-access
provider reads the token from there regardless of which route it arrived on.

Agent-card discovery needs no exemption because auth there is optional, which the pass-through
above already models: an unauthenticated request reaches ``_handle_get_agent_card``, which
applies ``enable_unauthenticated_well_known_route`` and serves a redacted card or a 401. A
request that *does* present a token gets the full check first -- a token naming another agent is
rejected rather than earning a card.

Installed by ``fastapi.DRAgentFastApiFrontEndPluginWorker.build_app``.
"""

import logging
from http import HTTPStatus

from nat.authentication.jwt_utils import decode_jwt_claims_unverified
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.responses import Response
from starlette.types import ASGIApp

from datarobot_genai.dragent.inbound_token import find_idp_token

logger = logging.getLogger(__name__)


def _audience_claim(token: str) -> list[str]:
    """Return ``aud`` as a list of strings, or ``[]``.  Compared verbatim by the caller -- no
    case or trailing-slash leniency.

    Normalized here rather than with NAT's ``UserManager._user_info_from_jwt``, which is private
    and rejects tokens carrying no identity claim.

    Raises
    ------
        ValueError: If the token is empty, malformed, or undecodable.
    """
    raw = decode_jwt_claims_unverified(token).get("aud")
    values = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []
    return [entry for entry in values if isinstance(entry, str)]


def _error(status: HTTPStatus, message: str) -> JSONResponse:
    """Rejection body; shape matches ``drmcp.core.middleware``."""
    return JSONResponse(status_code=status, content={"detail": message})


class GeneralOAuthClaimValidationMiddleware(BaseHTTPMiddleware):
    """Audience check on every inbound request.  See the module docstring for scope.

    Sibling of ``drmcp.core.middleware.GeneralOAuthClaimValidationMiddleware``, duplicated so
    ``dragent`` does not depend on the MCP subpackages.

    HTTP-only is sufficient: ``DRAgentFastApiFrontEndConfig`` overrides NAT's ``workflow``
    default without a ``websocket_path``, so no WebSocket route is registered.  If a future
    scope-validation phase needs to hand claims to the endpoint, switch to pure ASGI --
    ``BaseHTTPMiddleware`` runs ``call_next`` in a separate task, so ``ContextVar``s set here
    do not reach it.
    """

    def __init__(self, app: ASGIApp, expected_audience: str) -> None:
        super().__init__(app)
        self._expected_audience = expected_audience

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        rejection = self._reject(request)
        if rejection is not None:
            return rejection
        return await call_next(request)

    def _reject(self, request: Request) -> JSONResponse | None:
        """Response to send instead of calling the app, or ``None`` to allow."""
        token = find_idp_token(request.headers)
        if token is None:
            return None  # No claim validation When using standard datarobot api tokens

        try:
            audience = _audience_claim(token)
        except ValueError as ex:
            message = f"Malformed authorization token: {ex}"
            logger.info(message)
            return _error(HTTPStatus.UNPROCESSABLE_ENTITY, message)

        # `audience` is a list, so `in` is exact equality per entry. If _audience_claim ever
        # returned the raw claim, a str would make this a substring match and let "-aaa"
        # satisfy "aaa" -- see TestExactAudienceMatching.
        if self._expected_audience not in audience:
            message = "Authorization audience claim validation failed"  # no token/claim values
            logger.info(message)
            return _error(HTTPStatus.UNAUTHORIZED, message)

        logger.debug("OAuth audience claim validation succeeded")
        return None
