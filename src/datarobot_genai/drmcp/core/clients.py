# Copyright 2025 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Any
from typing import cast

import datarobot as dr
from datarobot.context import Context as DRContext
from datarobot.rest import RESTClientObject
from fastmcp.server.dependencies import get_http_headers

from datarobot_genai.core.utils.auth import AuthContextHeaderHandler
from datarobot_genai.core.utils.auth import DRAppCtx

from .credentials import get_credentials

logger = logging.getLogger(__name__)

# Header names to check for authorization tokens (in order of preference)
HEADER_TOKEN_CANDIDATE_NAMES = [
    "authorization",
    "x-datarobot-api-token",
    "x-datarobot-api-key",
]


def _extract_token_from_headers(headers: dict[str, str]) -> str | None:
    """
    Extract a Bearer token from headers by checking multiple header name candidates.

    Args:
        headers: Dictionary of headers (keys should be lowercase)

    Returns
    -------
        The extracted token string, or None if not found
    """
    for candidate_name in HEADER_TOKEN_CANDIDATE_NAMES:
        auth_header = headers.get(candidate_name)
        if not auth_header:
            continue

        if not isinstance(auth_header, str):
            continue

        # Handle Bearer token format
        bearer_prefix = "bearer "
        if auth_header.lower().startswith(bearer_prefix):
            token = auth_header[len(bearer_prefix) :].strip()
        else:
            # Assume it's a plain token
            token = auth_header.strip()

        if token:
            return token

    return None


def _extract_token_from_auth_context(headers: dict[str, str]) -> str | None:
    """
    Extract API token from authorization context metadata as a fallback.

    Args:
        headers: Dictionary of headers (keys should be lowercase)

    Returns
    -------
        The extracted API key from auth context metadata, or None if not found
    """
    try:
        auth_handler = AuthContextHeaderHandler()

        auth_ctx = auth_handler.get_context(headers)
        if not auth_ctx or not auth_ctx.metadata:
            return None

        metadata = auth_ctx.metadata
        if not isinstance(metadata, dict):
            return None

        dr_ctx: DRAppCtx = DRAppCtx(**metadata.get("dr_ctx", {}))
        if dr_ctx.api_key:
            logger.debug("Extracted token from auth context")
            return dr_ctx.api_key

        return None

    except Exception as e:
        logger.debug(f"Failed to get token from auth context: {e}")
        return None


def extract_token_from_headers(headers: dict[str, str]) -> str | None:
    """
    Extract a token from headers with multiple fallback strategies.

    This function attempts to extract a token in the following order:
    1. From standard authorization headers (Bearer token, x-datarobot-api-token, etc.)
    2. From authorization context metadata (dr_ctx.api_key)

    Args:
        headers: Dictionary of headers (keys should be lowercase)

    Returns
    -------
        The extracted token string, or None if not found
    """
    if token := _extract_token_from_headers(headers):
        return token

    if token := _extract_token_from_auth_context(headers):
        return token

    return None


def get_sdk_client() -> Any:
    """
    Get a DataRobot SDK client, using the user's Bearer token from the request.

    This function attempts to extract the Bearer token from the HTTP request headers
    with fallback strategies:
    1. Standard authorization headers (Bearer token, x-datarobot-api-token, etc.)
    2. Authorization context metadata (dr_ctx.api_key)
    3. Application credentials as final fallback
    """
    token = None

    try:
        headers = get_http_headers()
        if headers:
            token = extract_token_from_headers(headers)
            if token:
                logger.debug("Using API token found in HTTP headers")
    except Exception:
        # No HTTP context e.g. stdio transport
        logger.warning(
            "Could not get HTTP headers, falling back to application credentials", exc_info=True
        )

    credentials = get_credentials()

    # Fallback: Use application token
    if not token:
        token = credentials.datarobot.application_api_token
        logger.debug("Using application API token from credentials")

    dr.Client(token=token, endpoint=credentials.datarobot.endpoint)
    # The trafaret setting up a use case in the context, seem to mess up the tool calls
    DRContext.use_case = None
    return dr


def get_api_client() -> RESTClientObject:
    """Get a DataRobot SDK api client using application credentials."""
    dr = get_sdk_client()

    return cast(RESTClientObject, dr.client.get_client())


def get_s3_bucket_info() -> dict[str, str]:
    """Get S3 bucket configuration."""
    credentials = get_credentials()
    return {
        "bucket": credentials.aws_predictions_s3_bucket,
        "prefix": credentials.aws_predictions_s3_prefix,
    }
