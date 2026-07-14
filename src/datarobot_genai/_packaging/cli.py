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

"""Expose the canonical uv consumer configuration on the command line.

uv reads ``override-dependencies`` / ``exclude-dependencies`` /
``constraint-dependencies`` only from the workspace-root ``pyproject.toml``, so
these settings never propagate through package metadata. This command prints the
authoritative block shipped inside the wheel so a consumer can paste it into
their own ``[tool.uv]`` section.
"""

from __future__ import annotations

import sys
from importlib.resources import files

CONFIG_RESOURCE = "uv_consumer_config.toml"


def get_config_text() -> str:
    """Return the packaged uv consumer configuration as text."""
    return files(__package__).joinpath(CONFIG_RESOURCE).read_text(encoding="utf-8")


def main() -> int:
    """Print the packaged uv consumer configuration to stdout."""
    sys.stdout.write(get_config_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
