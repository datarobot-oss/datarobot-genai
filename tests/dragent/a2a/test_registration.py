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
#
# dragent ships its own copy of the A2A layer it used to take from nvidia-nat-a2a.
# These tests pin the parts of that move a user could notice: the ``a2a_client``
# workflow type still resolves, and nothing reaches back into the removed package.

import ast
from importlib.metadata import entry_points
from pathlib import Path

from nat.runtime.loader import load_config

import datarobot_genai.dragent
from datarobot_genai.dragent.a2a.client.client_config import A2AClientConfig
from datarobot_genai.dragent.plugins.auth_a2a_client import AuthenticatedA2AClientConfig

REMOVED_PACKAGE = "nat.plugins.a2a"


def test_a2a_client_entry_point_points_at_dragent_copy() -> None:
    # GIVEN the installed distribution's NAT plugin entry points
    plugins = {ep.name: ep.value for ep in entry_points(group="nat.plugins")}

    # WHEN looking up the plain A2A client plugin
    target = plugins.get("a2a_client")

    # THEN it is registered from dragent, not from nvidia-nat-a2a
    assert target == "datarobot_genai.dragent.a2a.client.client_impl"


def test_workflow_with_both_a2a_client_types_loads(tmp_path: Path) -> None:
    # GIVEN a workflow that uses the plain and the authenticated A2A client types
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        "function_groups:\n"
        "  plain_agent:\n"
        "    _type: a2a_client\n"
        '    url: "http://agent.example.com:8080"\n'
        "  authenticated_agent:\n"
        "    _type: authenticated_a2a_client\n"
        '    url: "http://agent.example.com:8080"\n'
    )

    # WHEN NAT loads it, discovering plugins through entry points
    config = load_config(workflow)

    # THEN each group resolves to dragent's config class for its type
    plain = config.function_groups["plain_agent"]
    authenticated = config.function_groups["authenticated_agent"]
    assert type(plain) is A2AClientConfig
    assert type(authenticated) is AuthenticatedA2AClientConfig


def test_dragent_does_not_import_removed_nat_a2a_package() -> None:
    # GIVEN every Python module in the dragent package
    package_root = Path(datarobot_genai.dragent.__file__).parent
    modules = sorted(package_root.rglob("*.py"))
    assert modules

    # WHEN collecting the modules each one imports, including function-local imports
    offenders = []
    for module in modules:
        tree = ast.parse(module.read_text(), filename=str(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            for name in names:
                if name == REMOVED_PACKAGE or name.startswith(REMOVED_PACKAGE + "."):
                    offenders.append(f"{module.relative_to(package_root)}:{node.lineno} {name}")

    # THEN none of them reach into nvidia-nat-a2a
    assert offenders == []
