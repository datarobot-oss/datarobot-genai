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

import re
from urllib.parse import urlparse
from urllib.parse import urlunparse


def get_api_base(api_base: str, deployment_id: str | None) -> str:
    """
    Construct the LiteLLM API base URL for a deployment.

    Parameters
    ----------
    api_base : str
        Base URL for the LiteLLM API.
    deployment_id : str | None
        Deployment identifier. When provided, a chat/completions URL is produced.

    Returns
    -------
    str
        Normalized URL for the given deployment. Ensures a trailing slash unless the path
        ends with "chat/completions" or already has a meaningful path component.
    """
    # Normalize the URL and drop a trailing /api/v2 if present
    parsed = urlparse(api_base)
    path = re.sub(r"/api/v2/?$", "", parsed.path)
    base_url = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )
    base_url = base_url.rstrip("/")

    # If the base_url already ends with chat/completions, return it.
    if base_url.endswith("chat/completions"):
        return base_url

    # If the path contains deployments or genai, it's already a complete API path preserve it.
    if path and ("deployments" in path or "genai" in path):
        return f"{base_url}/" if not base_url.endswith("/") else base_url

    # For all other cases (including custom base paths), apply deployment logic if needed.
    if deployment_id:
        return f"{base_url}/api/v2/deployments/{deployment_id}/chat/completions"
    # Otherwise, just return the base URL with a trailing slash for normalization.
    return f"{base_url}/"
