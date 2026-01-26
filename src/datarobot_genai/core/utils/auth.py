# Copyright 2025 DataRobot, Inc. and its affiliates.
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
import logging
import os
import warnings
from collections.abc import Sequence
from typing import Any
from typing import Protocol

import aiohttp
import jwt
from datarobot.auth.datarobot.oauth import AsyncOAuth as DatarobotOAuthClient
from datarobot.auth.exceptions import OAuthProviderNotFound
from datarobot.auth.exceptions import OAuthValidationErr
from datarobot.auth.identity import Identity
from datarobot.auth.oauth import OAuthToken
from datarobot.auth.session import AuthCtx
from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from datarobot.models.genai.agent.auth import ToolAuth
from datarobot.models.genai.agent.auth import get_authorization_context
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AuthContextConfig(DataRobotAppFrameworkBaseSettings):
    session_secret_key: str = ""


class DRAppCtx(BaseModel):
    """DataRobot application context from authorization metadata."""

    email: str | None = None
    api_key: str | None = None


class AuthContextHeaderHandler:
    """Manages encoding and decoding of authorization context into JWT tokens.

    This class provides a consistent interface for encoding auth context into JWT tokens
    and exchanging them via HTTP headers across multiple applications.
    """

    HEADER_NAME = "X-DataRobot-Authorization-Context"
    DEFAULT_ALGORITHM = "HS256"

    def __init__(
        self,
        secret_key: str | None = None,
        algorithm: str = DEFAULT_ALGORITHM,
        validate_signature: bool = True,
    ) -> None:
        """Initialize the handler.

        Parameters
        ----------
        secret_key : Optional[str]
            Secret key for JWT encoding/decoding. If None, tokens will be unsigned (insecure).
        algorithm : str
            JWT algorithm. Default is "HS256".
        validate_signature : bool
            Whether to validate JWT signatures. Default is True.

        Raises
        ------
        ValueError
            If algorithm is 'none' (insecure).
        """
        if algorithm is None:
            raise ValueError("Algorithm None is not allowed. Use a secure algorithm like HS256.")

        # Get secret key from parameter, config, or environment variable
        # Handle the case where AuthContextConfig() initialization fails due to
        # a bug in the datarobot package when SESSION_SECRET_KEY is not set
        if secret_key:
            self.secret_key = secret_key
        else:
            try:
                config = AuthContextConfig()
                self.secret_key = config.session_secret_key or ""
            except (TypeError, AttributeError, Exception):
                # Fallback to reading environment variable directly if config initialization fails
                # This can happen when SESSION_SECRET_KEY is not set and the datarobot package's
                # getenv function encounters a bug with None values
                # it tries to check if "apiToken" in payload: when payload is None
                self.secret_key = ""

        self.algorithm = algorithm
        self.validate_signature = validate_signature

    @property
    def header(self) -> str:
        """Get the header name for authorization context."""
        return self.HEADER_NAME

    def get_header(self, authorization_context: dict[str, Any] | None = None) -> dict[str, str]:
        """Get the authorization context header with encoded JWT token."""
        token = self.encode(authorization_context)
        if not token:
            return {}

        return {self.header: token}

    def encode(self, authorization_context: dict[str, Any] | None = None) -> str | None:
        """Encode the current authorization context into a JWT token."""
        auth_context = authorization_context or get_authorization_context()
        if not auth_context:
            return None

        if not self.secret_key:
            warnings.warn(
                "No secret key provided. Please make sure SESSION_SECRET_KEY is set. "
                "JWT tokens will be signed with an empty key. This is insecure and should "
                "only be used for testing."
            )

        return jwt.encode(auth_context, self.secret_key, algorithm=self.algorithm)

    def decode(self, token: str) -> dict[str, Any] | None:
        """Decode a JWT token into the authorization context."""
        if not token:
            return None

        if not self.secret_key and self.validate_signature:
            logger.error(
                "No secret key provided. Cannot validate signature. "
                "Provide a secret key or set validate_signature to False."
            )
            return None

        try:
            decoded = jwt.decode(
                jwt=token,
                key=self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_signature": self.validate_signature},
            )
        except jwt.ExpiredSignatureError:
            logger.info("JWT token has expired.")
            return None
        except jwt.InvalidTokenError:
            logger.warning("JWT token is invalid or malformed.")
            return None

        if not isinstance(decoded, dict):
            logger.warning("Decoded JWT token is not a dictionary.")
            return None

        return decoded

    def get_context(self, headers: dict[str, str]) -> AuthCtx | None:
        """Extract and validate authorization context from headers.

        Parameters
        ----------
        headers : Dict[str, str]
            HTTP headers containing the authorization context.

        Returns
        -------
        Optional[AuthCtx]
            Validated authorization context or None if validation fails.
        """
        token = headers.get(self.header) or headers.get(self.header.lower())
        if not token:
            logger.debug("No authorization context header found")
            return None

        auth_ctx_dict = self.decode(token)
        if not auth_ctx_dict:
            logger.debug("Failed to decode auth context from token")
            return None

        try:
            return AuthCtx(**auth_ctx_dict)
        except Exception as e:
            logger.error(f"Failed to create AuthCtx from decoded token: {e}", exc_info=True)
            return None


# --- OAuth Token Provider Implementation ---


class TokenRetriever(Protocol):
    """Protocol for OAuth token retrievers."""

    def filter_identities(self, identities: Sequence[Identity]) -> list[Identity]:
        """Filter identities to only those valid for this retriever implementation."""
        ...

    async def refresh_access_token(self, identity: Identity) -> OAuthToken:
        """Refresh the access token for the given identity ID.

        Parameters
        ----------
        identity_id : str
            The provider identity ID to refresh the token for.

        Returns
        -------
        OAuthToken
            The refreshed OAuth token.
        """
        ...


class DatarobotTokenRetriever:
    """Retrieves OAuth tokens using the DataRobot platform."""

    def __init__(self) -> None:
        self._client = DatarobotOAuthClient()

    def filter_identities(self, identities: Sequence[Identity]) -> list[Identity]:
        """Filter oauth2 identities to only those with provider_identity_id.

        The `provider_identity_id` is required in order to identify the provider
        and retrieve the access token from DataRobot OAuth Providers service.
        """
        return [i for i in identities if i.type == "oauth2" and i.provider_identity_id]

    async def refresh_access_token(self, identity: Identity) -> OAuthToken:
        """Refresh the access token using DataRobot's OAuth client."""
        return await self._client.refresh_access_token(identity_id=identity.provider_identity_id)


class AuthlibTokenRetriever:
    """Retrieves OAuth tokens from a generic Authlib-based endpoint."""

    def __init__(self, application_endpoint: str) -> None:
        if not application_endpoint:
            raise ValueError("AuthlibTokenRetriever requires 'application_endpoint'.")
        self.application_endpoint = application_endpoint.rstrip("/")

    def filter_identities(self, identities: Sequence[Identity]) -> list[Identity]:
        """Filter identities to only OAuth2 identities."""
        return [i for i in identities if i.type == "oauth2" and i.provider_identity_id is None]

    async def refresh_access_token(self, identity: Identity) -> OAuthToken:
        """Retrieve an OAuth token via an HTTP POST request.

        Parameters
        ----------
        identity : Identity
            The identity to retrieve the token for.

        Returns
        -------
        OAuthToken
            The retrieved OAuth token.
        """
        api_token = os.environ.get("DATAROBOT_API_TOKEN")
        if not api_token:
            raise ValueError("DATAROBOT_API_TOKEN environment variable is required but not set.")

        token_url = f"{self.application_endpoint}/oauth/token/"
        headers = {"Authorization": f"Bearer {api_token}"}
        payload = {"identity_id": identity.id}
        timeout = aiohttp.ClientTimeout(total=30)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(token_url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
                    logger.debug(f"Retrieved access token from {token_url}")
                    return OAuthToken(**data)
        except aiohttp.ClientError as e:
            logger.error(f"Error retrieving token from {token_url}: {e}")
            raise


class OAuthConfig(BaseModel):
    """Configuration extracted from AuthCtx metadata for OAuth operations."""

    implementation: str = "datarobot"
    application_endpoint: str | None = None

    @classmethod
    def from_auth_ctx(cls, auth_ctx: AuthCtx) -> "OAuthConfig":
        metadata = auth_ctx.metadata or {}
        return cls(
            implementation=metadata.get("oauth_implementation", "datarobot"),
            application_endpoint=metadata.get("application_endpoint"),
        )


def create_token_retriever(config: OAuthConfig) -> TokenRetriever:
    """Create a token retriever based on the OAuth configuration.

    Parameters
    ----------
    config : OAuthConfig
        The OAuth configuration specifying implementation type and endpoints.

    Returns
    -------
    TokenRetriever
        The configured token retriever instance.
    """
    if config.implementation == "datarobot":
        return DatarobotTokenRetriever()

    if config.implementation == "authlib":
        if not config.application_endpoint:
            raise ValueError("Required 'application_endpoint' not found in metadata.")
        return AuthlibTokenRetriever(config.application_endpoint)

    raise ValueError(
        f"Unsupported OAuth implementation: '{config.implementation}'. "
        f"Supported values: datarobot, authlib."
    )


class AsyncOAuthTokenProvider:
    """Provides OAuth tokens for authorized users.

    This class manages OAuth token retrieval for users with multiple identity providers.
    It uses either DataRobot or Authlib as the OAuth token storage and refresh backend
    based on the auth context metadata.
    """

    def __init__(self, auth_ctx: AuthCtx) -> None:
        """Initialize the provider with an authorization context.

        Parameters
        ----------
        auth_ctx : AuthCtx
            The authorization context containing user identities and metadata.
        """
        self.auth_ctx = auth_ctx
        config = OAuthConfig.from_auth_ctx(auth_ctx)
        self._retriever = create_token_retriever(config)

    def _get_identity(self, provider_type: str | None) -> Identity:
        """Get identity from auth context, filtered by provider_type if specified."""
        oauth_identities = self._retriever.filter_identities(self.auth_ctx.identities)
        if not oauth_identities:
            raise OAuthProviderNotFound("No OAuth provider found.")

        if provider_type is None:
            if len(oauth_identities) > 1:
                raise OAuthValidationErr(
                    "Multiple OAuth providers found. Specify 'provider_type' parameter."
                )
            return oauth_identities[0]

        identity = next((i for i in oauth_identities if i.provider_type == provider_type), None)
        if identity is None:
            raise OAuthValidationErr(f"No identity found for provider '{provider_type}'.")
        return identity

    async def get_token(self, auth_type: ToolAuth, provider_type: str | None = None) -> str:
        """Get an OAuth access token for the specified auth type and provider.

        Parameters
        ----------
        auth_type : ToolAuth
            Authentication type (only OBO is supported).
        provider_type : str, optional
            The specific provider to use (e.g., 'google'). Required if multiple
            identities are available.

        Returns
        -------
        str
            The retrieved OAuth access token.

        Raises
        ------
        ValueError
            If the auth type is unsupported or if a suitable identity cannot be found.
        """
        if auth_type != ToolAuth.OBO:
            raise ValueError(
                f"Unsupported auth type: {auth_type}. Only OBO (on-behalf-of) is supported."
            )

        identity = self._get_identity(provider_type)
        token_data = await self._retriever.refresh_access_token(identity)
        return token_data.access_token
