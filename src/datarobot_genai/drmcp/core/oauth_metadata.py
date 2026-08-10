# Copyright 2026 DataRobot, Inc.
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

"""Assemble the OAuth protected-resource metadata this server publishes.

The document itself lives in
:mod:`datarobot_genai.drmcpbase.oauth_protected_resource_metadata`, which
knows nothing about where settings come from. This module is the adapter that
reads them off :class:`MCPServerConfig` — one function shared by the
well-known route and the startup validation pass, so what is checked at boot
can never drift from what is served.
"""

from datarobot_genai.drmcpbase.oauth_protected_resource_metadata.entities import (
    MCPOAuthProtectedResourceMetadataConfig,
)

from .config import get_config


def build_protected_resource_metadata_config() -> MCPOAuthProtectedResourceMetadataConfig:
    """Assemble the published protected-resource metadata from this server's settings.

    Building the config is what validates it — an incomplete
    Cross-Application Access block is dropped with a warning naming the
    missing variables.
    """
    config = get_config()
    return MCPOAuthProtectedResourceMetadataConfig.from_settings(
        resource=config.mcp_oauth_resource,
        authorization_servers=config.mcp_oauth_authorization_servers,
        scopes_supported=config.mcp_oauth_scopes_supported,
        xaa_trusted_issuer=config.mcp_xaa_trusted_issuer,
        xaa_exchange_audience=config.mcp_xaa_exchange_audience,
        xaa_token_url=config.mcp_xaa_token_url,
        xaa_token_audience=config.mcp_xaa_token_audience,
        xaa_scopes=config.mcp_xaa_scopes,
        xaa_token_endpoint_auth_method=config.mcp_xaa_token_endpoint_auth_method,
    )
