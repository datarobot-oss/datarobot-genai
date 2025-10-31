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

from typing import Any
from typing import cast

import datarobot as dr
from datarobot.context import Context as DRContext
from datarobot.rest import RESTClientObject

from .credentials import get_credentials


def get_sdk_client(ctx: Any | None = None) -> Any:
    """
    Get a DataRobot SDK client, using the user's Bearer token from the request if available.
    Args:
        ctx: Optional FastMCP Context object. If provided, will attempt to extract the
            Bearer token from the request headers.

    Returns
    -------
        datarobot module with authenticated client.
    """
    token = None
    endpoint = None
    if ctx is not None:
        # Try to get the Bearer token from the request headers
        auth_header = None
        # FastMCP context may have .request or .request_headers
        if hasattr(ctx, "request") and hasattr(ctx.request, "headers"):
            headers = ctx.request.headers
            # headers may be a dict or a case-insensitive dict
            for k, v in headers.items():
                if k.lower() == "authorization":
                    auth_header = v
                    break
        if auth_header and auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
    if not token:
        credentials = get_credentials()
        token = credentials.datarobot.application_api_token
        endpoint = credentials.datarobot.endpoint
    else:
        credentials = get_credentials()
        endpoint = credentials.datarobot.endpoint
    dr.Client(token=token, endpoint=endpoint)
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
