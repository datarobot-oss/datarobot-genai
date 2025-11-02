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
from typing import Any, Dict, Optional

import jwt

from datarobot.auth.session import AuthCtx
from datarobot.models.genai.agent.auth import get_authorization_context, set_authorization_context

logger = logging.getLogger(__name__)


class AuthContextHeaderHandler:
    """Manages encoding and decoding of authorization context into JWT tokens.

    Parameters
    ----------
    secret_key : Optional[str]
        Secret key used for encoding and decoding JWT tokens.
    algorithm : str
        Algorithm used for JWT encoding/decoding. Default is "HS256".
    validate_signature : bool
        Whether to validate the JWT signature during decoding. Default is True.
        Disabling signature verification is insecure and should only be used for
        testing or controlled non-production scenarios.
    """

    header = "X-DataRobot-Authorization-Context"

    def __init__(self, secret_key: Optional[str] = None, algorithm: str = "HS256", validate_signature: bool = True) -> None:
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.validate_signature = validate_signature

    def get_header(self) -> dict[str, str]:
        """Get the authorization context header with encoded JWT token."""
        token = self.encode()
        if not token:
            return {}

        return {self.header: token}

    def encode(self) -> Optional[str]:
        """Encode the current authorization context into a JWT token."""
        auth_context = get_authorization_context()
        if not auth_context:
            return None

        if self.secret_key is None:
            logger.warning(
                "No secret key provided for AuthContextHeaderHandler; JWT will be unsigned. "
                "This is insecure and should only be used for testing or controlled "
                "non-production scenarios."
            )
        return jwt.encode(auth_context, self.secret_key, algorithm=self.algorithm)

    def decode(self, token: str) -> Optional[Dict[str, Any]]:
        """Decode a JWT token into the authorization context."""
        if not token:
            return None

        try:
            options = {"verify_signature": self.validate_signature}
            decoded = jwt.decode(
                jwt=token,
                key=self.secret_key,
                algorithms=[self.algorithm],
                options=options
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

    def get_context(self, headers: dict[str, str]) -> Optional[AuthCtx]:
        """Set the authorization context from the provided headers."""
        auth_ctx_dict = self.decode(headers.get(self.header))
        if not auth_ctx_dict:
            return

        return AuthCtx(**auth_ctx_dict)


