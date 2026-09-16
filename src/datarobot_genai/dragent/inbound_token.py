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
``authorization`` carries DataRobot's own credentials instead -- an opaque API token, or a
DataRobot-issued JWT that decodes perfectly well and has nothing to do with the caller's IdP --
so it is never read here.
"""

from collections.abc import Mapping

# Forwarded by the DataRobot API Gateway.  Carries nothing but the IdP token.
OAUTH_ACCESS_TOKEN_HEADER = "x-datarobot-external-access-token"

# Never read as an IdP-token carrier -- it holds DataRobot's own credentials instead.  Kept as
# a constant because ``TOKEN_HEADERS`` and the retired ``fallback_token_headers`` config field
# (see ``okta_a2a_auth``) still need to name it.
OAUTH_ACCESS_TOKEN_FALLBACK_HEADER = "authorization"

TOKEN_HEADERS = frozenset({OAUTH_ACCESS_TOKEN_HEADER, OAUTH_ACCESS_TOKEN_FALLBACK_HEADER})


def _without_bearer(value: str) -> str | None:
    """Return the token from ``Bearer <token>``, or ``None`` for any other scheme."""
    scheme, _, token = value.strip().partition(" ")
    return token.strip() or None if scheme.lower() == "bearer" else None


def find_idp_token(headers: Mapping[str, str]) -> str | None:
    """Return the caller's IdP access token, or ``None``.

    Needs only case-insensitive ``get`` -- Starlette ``Headers`` and NAT context headers both
    qualify.  Reads only ``OAUTH_ACCESS_TOKEN_HEADER`` and returns it whatever its shape, since
    that header carries nothing else; see the module docstring for why ``authorization`` is
    never consulted.
    """
    raw = headers.get(OAUTH_ACCESS_TOKEN_HEADER)
    if not raw:
        return None
    return _without_bearer(raw) or raw.strip() or None  # gateway sends it bare
