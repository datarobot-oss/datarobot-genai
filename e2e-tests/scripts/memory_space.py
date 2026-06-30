# Copyright 2026 DataRobot, Inc. and its affiliates.
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

"""Create or delete an ephemeral DataRobot MemorySpace for e2e tests.

Used by ``run_local.py`` and the GitHub Actions e2e workflow when a case sets
``E2E_PROVISION_MEMORY_SPACE=true``. Requires ``DATAROBOT_ENDPOINT`` and
``DATAROBOT_API_TOKEN`` in the environment.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid

import datarobot as dr
from datarobot.models.memory import MemorySpace


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"error: {name} is required", file=sys.stderr)
        raise SystemExit(2)
    return value


def _configure_client() -> None:
    dr.Client(
        endpoint=_require_env("DATAROBOT_ENDPOINT"),
        token=_require_env("DATAROBOT_API_TOKEN"),
    )


def create_space() -> str:
    """Create a MemorySpace and print its id to stdout."""
    _configure_client()
    test_id = uuid.uuid4().hex
    space = MemorySpace.create(description=f"datarobot-genai e2e memory {test_id}")
    print(space.id)
    return space.id


def delete_space(space_id: str) -> None:
    """Delete a MemorySpace by id."""
    _configure_client()
    MemorySpace(id=space_id).delete()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="memory_space.py")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("create", help="Create a MemorySpace and print its id.")
    delete_parser = sub.add_parser("delete", help="Delete a MemorySpace by id.")
    delete_parser.add_argument("space_id")
    args = parser.parse_args(argv)

    try:
        if args.cmd == "create":
            create_space()
        else:
            delete_space(args.space_id)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
