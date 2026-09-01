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
"""

from collections.abc import Mapping

from nat.authentication.jwt_utils import decode_jwt_claims_unverified

# Forwarded by the DataRobot API Gateway.  Carries nothing but the IdP token.
OAUTH_ACCESS_TOKEN_HEADER = "x-datarobot-external-access-token"

# Fallback for local runs with no gateway.  Also carries the opaque DataRobot API token, so a
# value here counts only if it decodes as a JWT.
OAUTH_ACCESS_TOKEN_FALLBACK_HEADER = "authorization"

TOKEN_HEADERS = frozenset({OAUTH_ACCESS_TOKEN_HEADER, OAUTH_ACCESS_TOKEN_FALLBACK_HEADER})


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
    qualify.  The primary header wins and is returned whatever its shape, since it carries
    nothing else; the fallback is taken only when it decodes as a JWT, so a DataRobot API token
    there is left for the credential it belongs to.
    """
    if raw := headers.get(OAUTH_ACCESS_TOKEN_HEADER):
        return _without_bearer(raw) or raw.strip() or None  # gateway sends it bare
    if raw := headers.get(OAUTH_ACCESS_TOKEN_FALLBACK_HEADER):
        token = _without_bearer(raw)
        if token and _is_jwt(token):
            return token
    return None
