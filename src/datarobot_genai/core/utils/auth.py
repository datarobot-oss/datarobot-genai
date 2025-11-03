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
from typing import Any

import jwt
from datarobot.auth.session import AuthCtx
from datarobot.models.genai.agent.auth import get_authorization_context

logger = logging.getLogger(__name__)


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
        token_expiration_seconds: int | None = None,
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

        self.secret_key = secret_key or os.getenv("SESSION_SECRET_KEY")
        self.algorithm = algorithm
        self.validate_signature = validate_signature

    @property
    def header(self) -> str:
        """Get the header name for authorization context."""
        return self.HEADER_NAME

    def get_header(self) -> dict[str, str]:
        """Get the authorization context header with encoded JWT token."""
        token = self.encode()
        if not token:
            return {}

        return {self.header: token}

    def encode(self) -> str | None:
        """Encode the current authorization context into a JWT token."""
        auth_context = get_authorization_context()
        if not auth_context:
            return None

        if self.secret_key is None:
            logger.warning(
                "No secret key provided. JWT tokens will be unsigned. "
                "This is insecure and should only be used for testing."
            )

        return jwt.encode(auth_context, self.secret_key, algorithm=self.algorithm)

    def decode(self, token: str) -> dict[str, Any] | None:
        """Decode a JWT token into the authorization context."""
        if not token:
            return None

        if self.secret_key is None and self.validate_signature:
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
        token = headers.get(self.header)
        if not token:
            logger.debug("No authorization context header found")
            return None

        auth_ctx_dict = self.decode(token)
        if not auth_ctx_dict:
            return None

        try:
            return AuthCtx(**auth_ctx_dict)
        except Exception as e:
            logger.error(f"Failed to create AuthCtx from decoded token: {e}", exc_info=True)
            return None
