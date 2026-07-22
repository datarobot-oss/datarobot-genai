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

"""MSF service configuration for MongoDB."""

from __future__ import annotations

from urllib.parse import urlparse

from msf.config import ServiceConfig
from msf.databases.mongodb.connection import MongodbConfig
from pydantic import Field

from datarobot_genai.drmcp.platform.__service__ import __service__


class DRMcpMongoConfig(MongodbConfig):
    """MongoDB connection settings for the MCP server."""

    transactions_enabled: bool = Field(default=False, env="MONGO_TRANSACTIONS_ENABLED")


class DRMcpServiceConfig(ServiceConfig):
    """Declarative MSF config for drmcp MongoDB access and migrations."""

    mongo: DRMcpMongoConfig = Field(default_factory=DRMcpMongoConfig, no_nesting=True)

    class Config:
        namespace = __service__
        secrets_dir = "/var/run/drmcp"

    @property
    def database_name(self) -> str:
        """Resolve the logical database name from ``MONGO_URI`` or MSF fallback."""
        uri_path = urlparse(self.mongo.mongo_uri.get_secret_value()).path.lstrip("/")
        if uri_path:
            return uri_path
        return self.mongo.database_name
