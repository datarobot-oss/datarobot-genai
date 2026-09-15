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


"""Where the caller's IdP access token arrives.

One definition, two consumers -- ``frontends/claim_validation.py`` validates the token's
``aud``, ``plugins/okta_a2a_auth.py`` exchanges it.  If they read different headers, a token
could skip validation and still be exchanged.  Not configurable: the gateway picks the header
it forwards, so naming another one would just receive nothing.

Only ``x-datarobot-external-access-token`` carries an IdP token.  The gateway populates it with
the external token it has already validated, so a value there is an IdP token by construction.
``authorization`` carries DataRobot's own credentials -- an opaque API token, or a
DataRobot-minted JWT that decodes perfectly well and has nothing to do with the caller's IdP --
so it is *not* read here unless ``DRAGENT_ALLOW_IDP_TOKEN_IN_AUTHORIZATION`` opts in.  That
switch exists for local runs with no gateway in front and nowhere else.
"""

import logging
from collections.abc import Mapping
from functools import lru_cache

from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from nat.authentication.jwt_utils import decode_jwt_claims_unverified
from pydantic import Field

# Forwarded by the DataRobot API Gateway.  Carries nothing but the IdP token.
OAUTH_ACCESS_TOKEN_HEADER = "x-datarobot-external-access-token"

# Read as an IdP-token carrier only under the opt-in below.
OAUTH_ACCESS_TOKEN_FALLBACK_HEADER = "authorization"

TOKEN_HEADERS = frozenset({OAUTH_ACCESS_TOKEN_HEADER, OAUTH_ACCESS_TOKEN_FALLBACK_HEADER})

logger = logging.getLogger(__name__)


class _InboundTokenSettings(DataRobotAppFrameworkBaseSettings):
    """Reads the local-dev carrier opt-in from env / Runtime Parameters / ``.env`` / secrets."""

    dragent_allow_idp_token_in_authorization: bool = Field(
        default=False,
        description=(
            "Local development only. Read the caller's IdP access token from the "
            "``authorization`` header when it decodes as a JWT, for runs with no DataRobot API "
            "Gateway in front to populate ``x-datarobot-external-access-token``. Leave off in "
            "a deployment."
        ),
    )


@lru_cache(maxsize=1)
def _authorization_carries_idp_token() -> bool:
    """Whether ``authorization`` is an IdP-token carrier in this process.

    Cached because it is read on every request and cannot change without a restart -- the
    settings chain walks ``.env``, file secrets and ``pulumi_config.json``.  Tests that flip
    the env var must ``cache_clear()`` (see ``tests/dragent/conftest.py``).

    A settings chain that will not load -- an unparseable value, a malformed
    ``pulumi_config.json`` -- resolves to ``False`` instead of propagating.  This runs on every
    request, including the ones carrying no IdP token at all, so letting a config typo out of
    here would turn requests this library promises to leave alone into 500s.  ``False`` is both
    the field default and the deployment posture, and the cache means the warning is logged
    once per process rather than once per request.
    """
    try:
        return _InboundTokenSettings().dragent_allow_idp_token_in_authorization
    except Exception:
        logger.warning(
            "Could not resolve DRAGENT_ALLOW_IDP_TOKEN_IN_AUTHORIZATION; treating it as off, "
            "so '%s' is not read as an IdP token carrier.",
            OAUTH_ACCESS_TOKEN_FALLBACK_HEADER,
            exc_info=True,
        )
        return False


def _without_bearer(value: str) -> str | None:
    """Return the token from ``Bearer <token>``, or ``None`` for any other scheme."""
    scheme, _, token = value.strip().partition(" ")
    return token.strip() or None if scheme.lower() == "bearer" else None


def _is_jwt(value: str) -> bool:
    """Whether ``value`` decodes as a JWT.  Asks the parser; a dot count would accept an
    opaque token containing two dots.  No signature check -- the gateway owns that.
    """
    try:
        decode_jwt_claims_unverified(value)
    except ValueError:
        return False
    return True


def find_idp_token(headers: Mapping[str, str]) -> str | None:
    """Return the caller's IdP access token, or ``None``.

    Needs only case-insensitive ``get`` -- Starlette ``Headers`` and NAT context headers both
    qualify.  The dedicated header wins and is returned whatever its shape, since it carries
    nothing else.  ``authorization`` is consulted only under the opt-in, and then only when it
    decodes as a JWT, so an opaque DataRobot API token there is left for the credential it
    belongs to.
    """
    if raw := headers.get(OAUTH_ACCESS_TOKEN_HEADER):
        return _without_bearer(raw) or raw.strip() or None  # gateway sends it bare
    if not _authorization_carries_idp_token():
        return None
    if raw := headers.get(OAUTH_ACCESS_TOKEN_FALLBACK_HEADER):
        token = _without_bearer(raw)
        if token and _is_jwt(token):
            return token
    return None
