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

"""Workload token requests and background refresh.

The token refreshes every ``DATAROBOT_WORKLOAD_TOKEN_REFRESH_SECONDS`` (default 3600),
or sooner when the token's lifetime is shorter. Keys and tokens are never logged.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from collections.abc import Sequence

import jwt
import requests

from .config import WORKLOAD_TOKEN_FILE_ENV
from .config import AuthMode
from .config import WorkloadIdentityConfig
from .config import WorkloadTokenError
from .config import load_signing_key

logger = logging.getLogger(__name__)

_ASSERTION_LIFETIME_SECONDS = 300
_TOKEN_EXCHANGE_GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"
_JWT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:jwt"
_JWT_BEARER_ASSERTION = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
_HTTP_TIMEOUT_SECONDS = 10.0
# Refresh by this fraction of the token lifetime when that comes before the interval.
_REFRESH_LIFETIME_FRACTION = 0.8
_MIN_RETRY_SECONDS = 30.0
_MAX_RETRY_SECONDS = 300.0
# A token this close to expiry counts as expired. Capped at a fraction of the lifetime
# so short tokens stay usable past the scheduled refresh.
_EXPIRY_SKEW_SECONDS = 30.0
_EXPIRY_SKEW_LIFETIME_FRACTION = 0.1


def _scope_param(scopes: Sequence[str]) -> str:
    """RFC 6749 ``scope``: space-separated, order kept, duplicates dropped."""
    if isinstance(scopes, str):
        raise TypeError("scopes must be a sequence of strings, not a str")
    out: list[str] = []
    for scope in scopes:
        # scope-token = 1*( %x21 / %x23-5B / %x5D-7E )
        if not scope or any(not 0x21 <= ord(c) <= 0x7E or c in '"\\' for c in scope):
            raise ValueError(f"invalid OAuth scope: {scope!r}")
        if scope not in out:
            out.append(scope)
    return " ".join(out)


class WorkloadTokenSource:
    """Holds a workload token and refreshes it in a background thread.

    ``token()`` fetches synchronously on first use and then starts the refresher.
    On refresh failure the current token is kept until it expires; after that
    ``token()`` raises until a background retry succeeds.
    """

    def __init__(
        self,
        config: WorkloadIdentityConfig,
        *,
        session: requests.Session | None = None,
        audience: str | None = None,
        scopes: Sequence[str] = (),
        refresh_interval: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Fix ``audience`` and ``scopes`` per source, so each source caches one token.

        ``audience`` defaults to ``config.api_endpoint``; no ``scopes`` sends no ``scope``.
        """
        self._config = config
        self._audience = (audience or config.api_endpoint).strip()
        self._scope = _scope_param(scopes)
        self._http = session or requests.Session()
        self._refresh_interval = refresh_interval or config.refresh_interval
        # Monotonic: scheduling is relative to expires_in, so wall-clock steps must not skew it.
        self._clock = clock
        self._signing_key = (
            load_signing_key(config.private_key, config.key_alg)
            if config.mode is AuthMode.PRIVATE_KEY_JWT
            else None
        )
        self._discovered_endpoint = ""
        # _refresh_lock serializes token requests; _lock guards state and is never held over HTTP.
        self._refresh_lock = threading.Lock()
        self._lock = threading.Lock()
        self._token: str | None = None
        self._valid_until = 0.0
        self._next_refresh_at = 0.0
        self._retry_delay = _MIN_RETRY_SECONDS
        self._last_error = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def token(self) -> str:
        """Return the current unexpired token.

        Raises ``WorkloadTokenError`` with the last refresh failure when no valid token is held.
        """
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
            if self._token is not None and self._clock() < self._valid_until:
                return self._token
            reason = self._last_error or "token expired"
        raise WorkloadTokenError(f"No valid workload token: {reason}")

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
                skew = min(_EXPIRY_SKEW_SECONDS, expires_in * _EXPIRY_SKEW_LIFETIME_FRACTION)
                self._valid_until = now + expires_in - skew
                # Never rely on the lifetime: refresh at the interval or before expiry.
                self._next_refresh_at = now + min(
                    self._refresh_interval, expires_in * _REFRESH_LIFETIME_FRACTION
                )
                self._retry_delay = _MIN_RETRY_SECONDS
                self._last_error = ""
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
        reason = str(exc) if isinstance(exc, WorkloadTokenError) else type(exc).__name__
        with self._lock:
            self._last_error = reason
            delay = self._retry_delay
            self._next_refresh_at = now + delay
            self._retry_delay = min(delay * 2, _MAX_RETRY_SECONDS)
            kept = self._token is not None and now < self._valid_until
        logger.error(
            "Workload token refresh failed (%s); %s; retrying in %.0fs",
            reason,
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
                    # Many servers publish only the OIDC document; a 404 here is normal.
                    logger.debug(
                        "No authorization server metadata at %s (HTTP %s)", url, resp.status_code
                    )
                    continue
                raise WorkloadTokenError(f"discovery at {url} returned HTTP {resp.status_code}")
            try:
                endpoint = resp.json().get("token_endpoint")
            except (ValueError, AttributeError):
                endpoint = None
            if not isinstance(endpoint, str) or not endpoint:
                # Some gateways answer unknown paths with 200; try the next document.
                if not last:
                    logger.debug("No token_endpoint in %s", url)
                    continue
                raise WorkloadTokenError(f"discovery at {url} has no token_endpoint")
            self._discovered_endpoint = endpoint
            return endpoint
        raise WorkloadTokenError("no discovery URL")  # pragma: no cover

    def _client_assertion(self) -> str:
        assert self._signing_key is not None  # only called in private_key_jwt mode
        # iat/exp are absolute claims, so these need wall-clock time.
        now = int(time.time())
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
            form = {
                "grant_type": _TOKEN_EXCHANGE_GRANT,
                "subject_token": self._read_subject_token(),
                "subject_token_type": _JWT_TOKEN_TYPE,
                "client_id": self._config.client_id,
                "audience": self._audience,
            }
        else:
            form = {
                "grant_type": "client_credentials",
                "client_id": self._config.client_id,
                "client_assertion_type": _JWT_BEARER_ASSERTION,
                "client_assertion": self._client_assertion(),
                "audience": self._audience,
            }
        if self._scope:
            form["scope"] = self._scope
        return form

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
