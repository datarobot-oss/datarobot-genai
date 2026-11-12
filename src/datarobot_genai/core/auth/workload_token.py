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

"""Workload identity token requester.

Two auth modes, picked from env:

* federation: ``DATAROBOT_WORKLOAD_TOKEN_FILE`` holds a projected Kubernetes
  ServiceAccount token, exchanged at the auth-server token bridge (RFC 8693).
  Preferred when both are present, since it needs no secret.
* private_key_jwt: ``DATAROBOT_CLIENT_PRIVATE_KEY`` (PKCS#8 PEM) and
  ``DATAROBOT_CLIENT_KEY_ID`` sign a client assertion for ``client_credentials``
  with ``DATAROBOT_CLIENT_KEY_ALG`` (default RS256; PyJWT + cryptography).

The client id is ``DATAROBOT_CLIENT_ID`` (``WORKLOAD_ID`` as a fallback). The
token endpoint comes from overrides or discovery off ``DATAROBOT_PUBLIC_API_ENDPOINT``.

Transport-neutral: ``WorkloadTokenSource.token()``, plus thin ``requests`` and
``httpx`` auth adapters. It imports nothing from datarobot_genai, so it can move
to a shared library as is. Keys and tokens are never logged.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Mapping
from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from urllib.parse import urlsplit

import httpx
import requests

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


_API_V2_SUFFIX = "/api/v2"
_ASSERTION_LIFETIME_SECONDS = 300
# Algs Hydra accepts for private_key_jwt, all FIPS-approved. EC algs pin the curve.
_RSA_ALGS = frozenset({"RS256", "RS384", "RS512", "PS256", "PS384", "PS512"})
_EC_ALG_CURVES = {"ES256": "secp256r1", "ES384": "secp384r1", "ES512": "secp521r1"}
_TOKEN_EXCHANGE_GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"
_JWT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:jwt"
_JWT_BEARER_ASSERTION = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
_HTTP_TIMEOUT_SECONDS = 10.0
# Refresh hourly; the 7-day lifetime only exists to ride out a hub outage.
DEFAULT_REFRESH_INTERVAL_SECONDS = 3600.0
_MIN_RETRY_SECONDS = 30.0
_MAX_RETRY_SECONDS = 300.0
# A token this close to expiry counts as expired.
_EXPIRY_SKEW_SECONDS = 30.0


class AuthMode(StrEnum):
    FEDERATION = "federation"
    PRIVATE_KEY_JWT = "private_key_jwt"


class WorkloadTokenError(Exception):
    """A workload token request failed. Messages never carry the key or a token."""


@dataclass(frozen=True)
class WorkloadIdentityConfig:
    mode: AuthMode
    client_id: str
    audience: str
    token_file: str = ""
    private_key: str = field(default="", repr=False)
    key_id: str = ""
    key_alg: str = DEFAULT_KEY_ALG
    discovery_url: str = ""
    token_endpoint: str = ""
    assertion_audience: str = ""

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> WorkloadIdentityConfig | None:
        """Read the workload identity env; ``None`` when it is absent or incomplete."""
        env = os.environ if environ is None else environ
        client_id = (env.get(CLIENT_ID_ENV) or env.get(WORKLOAD_ID_ENV) or "").strip()
        audience = env.get(PUBLIC_API_ENDPOINT_ENV, "").strip()
        token_file = env.get(WORKLOAD_TOKEN_FILE_ENV, "").strip()
        private_key = env.get(PRIVATE_KEY_ENV, "").strip()
        key_id = env.get(KEY_ID_ENV, "").strip()
        if not (token_file or private_key) or not (client_id and audience):
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
            audience=audience,
            token_file=token_file,
            private_key=private_key,
            key_id=key_id,
            key_alg=env.get(KEY_ALG_ENV, "").strip() or DEFAULT_KEY_ALG,
            discovery_url=env.get(DISCOVERY_URL_ENV, "").strip(),
            token_endpoint=env.get(TOKEN_ENDPOINT_ENV, "").strip(),
            assertion_audience=env.get(ASSERTION_AUDIENCE_ENV, "").strip(),
        )

    def discovery_urls(self) -> list[str]:
        """Metadata URLs to try in order: the override, else RFC 8414 then OIDC."""
        if self.discovery_url:
            return [self.discovery_url]
        # Strip a literal /api/v2; urljoin("../..") breaks path-prefixed installs.
        base = self.audience.rstrip("/")
        if base.endswith(_API_V2_SUFFIX):
            base = base[: -len(_API_V2_SUFFIX)]
        parts = urlsplit(base)
        origin = f"{parts.scheme}://{parts.netloc}"
        return [
            f"{origin}/.well-known/oauth-authorization-server{parts.path}",
            f"{base}/.well-known/openid-configuration",
        ]


def _load_signing_key(private_key: str, alg: str) -> object:
    """Load the PKCS#8 PEM private key and check it fits ``alg``."""
    from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePrivateKey
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

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


class WorkloadTokenSource:
    """Holds a workload token and refreshes it in a background thread.

    ``token()`` fetches synchronously on first use and then starts the refresher.
    On refresh failure the current token is kept until it expires.
    """

    def __init__(
        self,
        config: WorkloadIdentityConfig,
        *,
        session: requests.Session | None = None,
        refresh_interval: float = DEFAULT_REFRESH_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._config = config
        self._http = session or requests.Session()
        self._refresh_interval = refresh_interval
        self._clock = clock
        self._signing_key = (
            _load_signing_key(config.private_key, config.key_alg)
            if config.mode is AuthMode.PRIVATE_KEY_JWT
            else None
        )
        self._discovered_endpoint = ""
        # _refresh_lock serializes token requests; _lock guards state and is never held over HTTP.
        self._refresh_lock = threading.Lock()
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires_at = 0.0
        self._next_refresh_at = 0.0
        self._retry_delay = _MIN_RETRY_SECONDS
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def token(self) -> str | None:
        """Return the current unexpired token, or ``None`` if none could be obtained."""
        with self._lock:
            needs_first = self._token is None
        if needs_first:
            self.refresh_if_due()
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="datarobot-workload-token", daemon=True
                )
                self._thread.start()
            if self._token is not None and self._clock() < self._expires_at - _EXPIRY_SKEW_SECONDS:
                return self._token
            return None

    def refresh_if_due(self) -> None:
        with self._refresh_lock:
            if self.seconds_until_refresh() > 0:
                return
            now = self._clock()
            try:
                token, expires_in = self._request_token()
            except Exception as exc:
                self._record_failure(now, exc)
                return
            with self._lock:
                self._token = token
                self._expires_at = now + expires_in
                # Hourly by default, or at 80% of the lifetime if sooner.
                self._next_refresh_at = now + min(self._refresh_interval, expires_in * 0.8)
                self._retry_delay = _MIN_RETRY_SECONDS
            logger.debug("Workload token refreshed; expires in %.0fs", expires_in)

    def seconds_until_refresh(self) -> float:
        with self._lock:
            return max(0.0, self._next_refresh_at - self._clock())

    def close(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.seconds_until_refresh()):
            self.refresh_if_due()

    def _record_failure(self, now: float, exc: Exception) -> None:
        with self._lock:
            delay = self._retry_delay
            self._next_refresh_at = now + delay
            self._retry_delay = min(delay * 2, _MAX_RETRY_SECONDS)
            kept = self._token is not None and now < self._expires_at
        logger.error(
            "Workload token refresh failed (%s); %s; retrying in %.0fs",
            exc if isinstance(exc, WorkloadTokenError) else type(exc).__name__,
            "keeping current token" if kept else "no valid token",
            delay,
        )

    def _discovered_token_endpoint(self) -> str:
        if self._discovered_endpoint:
            return self._discovered_endpoint
        urls = self._config.discovery_urls()
        for i, url in enumerate(urls):
            last = i + 1 == len(urls)
            try:
                resp = self._http.get(url, timeout=_HTTP_TIMEOUT_SECONDS)
            except requests.RequestException as exc:
                if last:
                    raise WorkloadTokenError(
                        f"discovery at {url} failed ({type(exc).__name__})"
                    ) from None
                logger.debug("Discovery at %s failed (%s)", url, type(exc).__name__)
                continue
            if resp.status_code != 200:
                if not last:
                    # Hydra serves only the OIDC document; a 404 here is the normal path.
                    logger.debug(
                        "No authorization server metadata at %s (HTTP %s)", url, resp.status_code
                    )
                    continue
                raise WorkloadTokenError(f"discovery at {url} returned HTTP {resp.status_code}")
            endpoint = resp.json().get("token_endpoint")
            if not isinstance(endpoint, str) or not endpoint:
                raise WorkloadTokenError(f"discovery at {url} has no token_endpoint")
            self._discovered_endpoint = endpoint
            return endpoint
        raise WorkloadTokenError("no discovery URL")  # pragma: no cover

    def _client_assertion(self) -> str:
        import jwt

        now = int(self._clock())
        claims = {
            "iss": self._config.client_id,
            "sub": self._config.client_id,
            # The token endpoint override never changes the aud.
            "aud": self._config.assertion_audience or self._discovered_token_endpoint(),
            "jti": str(uuid.uuid4()),
            "iat": now,
            "exp": now + _ASSERTION_LIFETIME_SECONDS,
        }
        return jwt.encode(
            claims,
            self._signing_key,
            algorithm=self._config.key_alg,
            headers={"kid": self._config.key_id},
        )

    def _read_subject_token(self) -> str:
        # Re-read on every fetch: the kubelet rotates the projected token.
        try:
            with open(self._config.token_file, encoding="utf-8") as f:
                token = f.read().strip()
        except OSError as exc:
            raise WorkloadTokenError(
                f"cannot read {WORKLOAD_TOKEN_FILE_ENV} ({type(exc).__name__})"
            ) from None
        if not token:
            raise WorkloadTokenError(f"{WORKLOAD_TOKEN_FILE_ENV} is empty")
        return token

    def _request_form(self) -> dict[str, str]:
        if self._config.mode is AuthMode.FEDERATION:
            # The bridge refuses client credentials and 'resource'; the target goes in 'audience'.
            return {
                "grant_type": _TOKEN_EXCHANGE_GRANT,
                "subject_token": self._read_subject_token(),
                "subject_token_type": _JWT_TOKEN_TYPE,
                "client_id": self._config.client_id,
                "audience": self._config.audience,
            }
        return {
            "grant_type": "client_credentials",
            "client_id": self._config.client_id,
            "client_assertion_type": _JWT_BEARER_ASSERTION,
            "client_assertion": self._client_assertion(),
            "audience": self._config.audience,
        }

    def _request_token(self) -> tuple[str, float]:
        form = self._request_form()
        url = self._config.token_endpoint or self._discovered_token_endpoint()
        resp = self._http.post(url, data=form, timeout=_HTTP_TIMEOUT_SECONDS)
        if resp.status_code != 200:
            raise WorkloadTokenError(
                f"token endpoint returned HTTP {resp.status_code} ({_oauth_error(resp)})"
            )
        body = resp.json()
        token = body.get("access_token")
        if not isinstance(token, str) or not token:
            raise WorkloadTokenError("token response has no access_token")
        try:
            expires_in = float(body.get("expires_in", 0))
        except (TypeError, ValueError):
            expires_in = 0.0
        if expires_in <= 0:
            raise WorkloadTokenError("token response has no usable expires_in")
        return token, expires_in


def _oauth_error(resp: requests.Response) -> str:
    try:
        return str(resp.json().get("error", "unknown"))
    except ValueError:
        return "unknown"


class WorkloadTokenAuth(requests.auth.AuthBase):
    """Sets ``Authorization: Bearer <workload token>`` on each request when a token is held."""

    def __init__(self, source: WorkloadTokenSource) -> None:
        self._source = source

    def __call__(self, r: requests.PreparedRequest) -> requests.PreparedRequest:
        _set_bearer(r.headers, self._source.token())
        return r


class WorkloadTokenHttpxAuth(httpx.Auth):
    """httpx ``Authorization: Bearer <workload token>`` for sync and async clients."""

    def __init__(self, source: WorkloadTokenSource) -> None:
        self._source = source

    def sync_auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        _set_bearer(request.headers, self._source.token())
        yield request

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response, None]:
        # token() may block on the first fetch; keep it off the event loop.
        _set_bearer(request.headers, await asyncio.to_thread(self._source.token))
        yield request


def _set_bearer(headers: MutableMapping[str, str], token: str | None) -> None:
    if token:
        headers["Authorization"] = f"Bearer {token}"


_SOURCE_LOCK = threading.Lock()
_SOURCE_STATE: dict[str, WorkloadTokenSource | None | bool] = {"source": None, "resolved": False}


def get_workload_token_source() -> WorkloadTokenSource | None:
    """Process-wide token source from env; ``None`` when workload identity is not configured."""
    with _SOURCE_LOCK:
        if not _SOURCE_STATE["resolved"]:
            _SOURCE_STATE["resolved"] = True
            config = WorkloadIdentityConfig.from_env()
            if config is not None:
                try:
                    _SOURCE_STATE["source"] = WorkloadTokenSource(config)
                except WorkloadTokenError as exc:
                    logger.error("Workload identity disabled: %s", exc)
        source = _SOURCE_STATE["source"]
        return source if isinstance(source, WorkloadTokenSource) else None


def _reset_workload_token_source() -> None:
    """Test hook: forget the cached process-wide source."""
    with _SOURCE_LOCK:
        source = _SOURCE_STATE["source"]
        if isinstance(source, WorkloadTokenSource):
            source.close()
        _SOURCE_STATE.update(source=None, resolved=False)
