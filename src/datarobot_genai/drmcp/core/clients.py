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


def get_sdk_client() -> Any:
    """
    Get a DataRobot SDK client, using the user's Bearer token from the request.

    This function attempts to extract the Bearer token from the HTTP request headers.
    If no token is found, it uses the application credentials.
    """
    token = None

    try:
        headers = get_http_headers()
        if headers:
            token = _extract_token_from_headers(headers)
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
