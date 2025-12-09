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
import warnings
from typing import Any

import jwt
from datarobot.auth.datarobot.oauth import AsyncOAuth as DatarobotAsyncOAuthClient
from datarobot.auth.identity import Identity
from datarobot.auth.oauth import AsyncOAuthComponent
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


class AsyncOAuthTokenProvider:
    """Manages OAuth access tokens using generic OAuth client."""

    def __init__(self, auth_ctx: AuthCtx) -> None:
        self.auth_ctx = auth_ctx
        self.oauth_client = self._create_oauth_client()

    def _get_identity(self, provider_type: str | None) -> Identity:
        """Retrieve the appropriate identity from the authentication context."""
        identities = [x for x in self.auth_ctx.identities if x.provider_identity_id is not None]

        if not identities:
            raise ValueError("No identities found in authorization context.")

        if provider_type is None:
            if len(identities) > 1:
                raise ValueError(
                    "Multiple identities found. Please specify 'provider_type' parameter."
                )
            return identities[0]

        identity = next((id for id in identities if id.provider_type == provider_type), None)

        if identity is None:
            raise ValueError(f"No identity found for provider '{provider_type}'.")

        return identity

    async def get_token(self, auth_type: ToolAuth, provider_type: str | None = None) -> str:
        """Get OAuth access token using the specified method."""
        if auth_type != ToolAuth.OBO:
            raise ValueError(
                f"Unsupported auth type: {auth_type}. Only {ToolAuth.OBO} is supported."
            )

        identity = self._get_identity(provider_type)
        token_data = await self.oauth_client.refresh_access_token(
            identity_id=identity.provider_identity_id
        )
        return token_data.access_token

    def _create_oauth_client(self) -> AsyncOAuthComponent:
        """Create either DataRobot or Authlib OAuth client based on
        authorization context.

        Note: at the moment, only DataRobot OAuth client is supported.
        """
        logger.debug("Using DataRobot OAuth client")
        return DatarobotAsyncOAuthClient()
