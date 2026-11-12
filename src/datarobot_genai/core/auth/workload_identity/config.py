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

"""Workload identity configuration from env.

Two auth modes:

* federation: ``DATAROBOT_WORKLOAD_TOKEN_FILE`` holds a projected Kubernetes
  ServiceAccount token, exchanged at the auth-server token bridge (RFC 8693).
  Preferred when both are present, since it needs no secret.
* private_key_jwt: ``DATAROBOT_CLIENT_PRIVATE_KEY`` (PKCS#8 PEM) and
  ``DATAROBOT_CLIENT_KEY_ID`` sign a client assertion for ``client_credentials``
  with ``DATAROBOT_CLIENT_KEY_ALG`` (default RS256; PyJWT + cryptography).

The client id is ``DATAROBOT_CLIENT_ID`` (``WORKLOAD_ID`` as a fallback). The
token endpoint comes from overrides or discovery off ``DATAROBOT_PUBLIC_API_ENDPOINT``,
which is also the default requested audience.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from urllib.parse import urlsplit

from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePrivateKey
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.hazmat.primitives.serialization import load_pem_private_key

logger = logging.getLogger(__name__)

CLIENT_ID_ENV = "DATAROBOT_CLIENT_ID"
# client_id is the workload id; used when CLIENT_ID_ENV is unset.
WORKLOAD_ID_ENV = "WORKLOAD_ID"
WORKLOAD_TOKEN_FILE_ENV = "DATAROBOT_WORKLOAD_TOKEN_FILE"
PRIVATE_KEY_ENV = "DATAROBOT_CLIENT_PRIVATE_KEY"
KEY_ID_ENV = "DATAROBOT_CLIENT_KEY_ID"
KEY_ALG_ENV = "DATAROBOT_CLIENT_KEY_ALG"
DEFAULT_KEY_ALG = "RS256"
PUBLIC_API_ENDPOINT_ENV = "DATAROBOT_PUBLIC_API_ENDPOINT"
DISCOVERY_URL_ENV = "DATAROBOT_OAUTH_DISCOVERY_URL"
TOKEN_ENDPOINT_ENV = "DATAROBOT_OAUTH_TOKEN_ENDPOINT"
ASSERTION_AUDIENCE_ENV = "DATAROBOT_OAUTH_CLIENT_ASSERTION_AUDIENCE"
REFRESH_INTERVAL_ENV = "DATAROBOT_WORKLOAD_TOKEN_REFRESH_SECONDS"

# Tokens are often long-lived to ride out an outage; refreshing more often limits exposure.
DEFAULT_REFRESH_INTERVAL_SECONDS = 3600.0

_API_V2_SUFFIX = "/api/v2"
# FIPS-approved private_key_jwt algs. EC algs pin the curve.
_RSA_ALGS = frozenset({"RS256", "RS384", "RS512", "PS256", "PS384", "PS512"})
_EC_ALG_CURVES = {"ES256": "secp256r1", "ES384": "secp384r1", "ES512": "secp521r1"}


class AuthMode(StrEnum):
    FEDERATION = "federation"
    PRIVATE_KEY_JWT = "private_key_jwt"


class WorkloadTokenError(Exception):
    """A workload token request failed. Messages never carry the key or a token."""


@dataclass(frozen=True)
class WorkloadIdentityConfig:
    mode: AuthMode
    client_id: str
    api_endpoint: str
    token_file: str = ""
    private_key: str = field(default="", repr=False)
    key_id: str = ""
    key_alg: str = DEFAULT_KEY_ALG
    discovery_url: str = ""
    token_endpoint: str = ""
    assertion_audience: str = ""
    refresh_interval: float = DEFAULT_REFRESH_INTERVAL_SECONDS

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> WorkloadIdentityConfig | None:
        """Read the workload identity env; ``None`` when it is absent or incomplete."""
        env = os.environ if environ is None else environ
        client_id = (env.get(CLIENT_ID_ENV) or env.get(WORKLOAD_ID_ENV) or "").strip()
        api_endpoint = env.get(PUBLIC_API_ENDPOINT_ENV, "").strip()
        token_file = env.get(WORKLOAD_TOKEN_FILE_ENV, "").strip()
        private_key = env.get(PRIVATE_KEY_ENV, "").strip()
        key_id = env.get(KEY_ID_ENV, "").strip()
        if not (token_file or private_key) or not (client_id and api_endpoint):
            return None
        if token_file:
            mode = AuthMode.FEDERATION
            if private_key:
                logger.warning("Workload identity: both auth modes configured; using federation")
            private_key = key_id = ""
        else:
            mode = AuthMode.PRIVATE_KEY_JWT
            if not key_id:
                logger.warning("Workload identity disabled: %s is not set", KEY_ID_ENV)
                return None
        return cls(
            mode=mode,
            client_id=client_id,
            api_endpoint=api_endpoint,
            token_file=token_file,
            private_key=private_key,
            key_id=key_id,
            key_alg=env.get(KEY_ALG_ENV, "").strip() or DEFAULT_KEY_ALG,
            discovery_url=env.get(DISCOVERY_URL_ENV, "").strip(),
            token_endpoint=env.get(TOKEN_ENDPOINT_ENV, "").strip(),
            assertion_audience=env.get(ASSERTION_AUDIENCE_ENV, "").strip(),
            refresh_interval=_refresh_interval_from_env(env),
        )

    def discovery_urls(self) -> list[str]:
        """Metadata URLs to try in order: the override, else RFC 8414 then OIDC."""
        if self.discovery_url:
            return [self.discovery_url]
        # Strip a literal /api/v2; urljoin("../..") breaks path-prefixed installs.
        base = self.api_endpoint.rstrip("/")
        if base.endswith(_API_V2_SUFFIX):
            base = base[: -len(_API_V2_SUFFIX)]
        parts = urlsplit(base)
        origin = f"{parts.scheme}://{parts.netloc}"
        return [
            f"{origin}/.well-known/oauth-authorization-server{parts.path}",
            f"{base}/.well-known/openid-configuration",
        ]


def _refresh_interval_from_env(env: Mapping[str, str]) -> float:
    raw = env.get(REFRESH_INTERVAL_ENV, "").strip()
    if not raw:
        return DEFAULT_REFRESH_INTERVAL_SECONDS
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    if value > 0:
        return value
    logger.warning(
        "Ignoring %s=%r; using %.0fs", REFRESH_INTERVAL_ENV, raw, DEFAULT_REFRESH_INTERVAL_SECONDS
    )
    return DEFAULT_REFRESH_INTERVAL_SECONDS


def load_signing_key(private_key: str, alg: str) -> RSAPrivateKey | EllipticCurvePrivateKey:
    """Load the PKCS#8 PEM private key and check it fits ``alg``."""
    if alg not in _RSA_ALGS and alg not in _EC_ALG_CURVES:
        raise WorkloadTokenError(f"{KEY_ALG_ENV}={alg!r} is not a supported signing alg")
    try:
        key = load_pem_private_key(private_key.encode(), password=None)
    except Exception:
        # Drop the cause: some parser errors echo key material.
        raise WorkloadTokenError(f"cannot load {PRIVATE_KEY_ENV} as a PKCS#8 PEM") from None
    if alg in _RSA_ALGS:
        if not isinstance(key, RSAPrivateKey):
            raise WorkloadTokenError(f"{PRIVATE_KEY_ENV} is not an RSA key, as {alg} needs")
    elif not isinstance(key, EllipticCurvePrivateKey) or key.curve.name != _EC_ALG_CURVES[alg]:
        raise WorkloadTokenError(
            f"{PRIVATE_KEY_ENV} is not an EC {_EC_ALG_CURVES[alg]} key, as {alg} needs"
        )
    return key
