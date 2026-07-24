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

"""Migration runner entry point for the MCP server.

Invoke via ``sbin/drmcp-migrate-db`` (which delegates here) or directly:

    uv run python -m datarobot_genai.drmcp.migrations          # apply pending migrations
    uv run python -m datarobot_genai.drmcp.migrations list     # list migrations
    uv run python -m datarobot_genai.drmcp.migrations latest   # show latest applied

Configuration is resolved via ``DRMcpServiceConfig``.
See ``drmcp/platform/mongo_config.py`` and ``sbin/drmcp-migrate-db`` for prerequisites.
"""

import logging
import sys
from pathlib import Path

from dr_mongo_migrations.cli import cli

from datarobot_genai.drmcp.platform.mongo_config import DRMcpServiceConfig

logger = logging.getLogger(__name__)


def main() -> None:
    """Resolve MSF configuration and delegate to the dr-mongo-migrations CLI."""
    config = DRMcpServiceConfig()
    mongo_uri = config.mongo.mongo_uri.get_secret_value().strip()
    if not mongo_uri:
        sys.exit("MONGO_URI environment variable is required")

    db_name = config.database_name
    migrations_dir = Path(__file__).resolve().parent.parent / "db_migrations"

    logger.info("Running migrations (db=%s, dir=%s)", db_name, migrations_dir)

    args = (
        "--migrations-dir",
        str(migrations_dir),
        "--mongo-uri",
        mongo_uri,
        "--migration-db",
        db_name,
        "--db-name",
        db_name,
        *sys.argv[1:],
    )
    sys.exit(cli(args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
