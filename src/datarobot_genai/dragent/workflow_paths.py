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

"""Locate ``workflow.yaml`` for DRAgent runtimes and deployments."""

from __future__ import annotations

import os
from pathlib import Path

from datarobot_genai.dragent.constants import DRAGENT_CONFIG_FILE_ENV

WORKFLOW_FILENAME = "workflow.yaml"
CODE_DIR_ENV = "CODE_DIR"


def _existing_workflow(path: Path) -> Path | None:
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        return None
    return resolved if resolved.is_file() else None


def discover_workflow_yaml() -> Path | None:
    """Return ``workflow.yaml`` when it can be located on disk.

    Resolution order:

    1. ``DRAGENT_CONFIG_FILE`` when it points at an existing file
    2. ``$CODE_DIR/workflow.yaml`` (DataRobot custom-model deployments set ``CODE_DIR``)
    3. ``workflow.yaml`` in the current working directory or any parent directory
    """
    config_file = os.environ.get(DRAGENT_CONFIG_FILE_ENV)
    if config_file:
        for candidate in _workflow_candidates_from_hint(config_file):
            if (found := _existing_workflow(candidate)) is not None:
                return found

    code_dir = os.environ.get(CODE_DIR_ENV)
    if code_dir and (found := _existing_workflow(Path(code_dir) / WORKFLOW_FILENAME)) is not None:
        return found

    for directory in [Path.cwd(), *Path.cwd().parents]:
        if (found := _existing_workflow(directory / WORKFLOW_FILENAME)) is not None:
            return found

    return None


def _workflow_candidates_from_hint(config_file: str) -> list[Path]:
    hinted = Path(config_file).expanduser()
    candidates = [hinted, Path.cwd() / hinted]
    code_dir = os.environ.get(CODE_DIR_ENV)
    if code_dir is not None:
        code_root = Path(code_dir).expanduser()
        candidates.extend([code_root / hinted, code_root / WORKFLOW_FILENAME])
    return candidates


def publish_dragent_config_file_env(
    workflow_path: Path | str | None = None,
) -> Path | None:
    """Set ``DRAGENT_CONFIG_FILE`` to an absolute ``workflow.yaml`` path when found."""
    resolved = (
        _existing_workflow(Path(workflow_path))
        if workflow_path is not None
        else discover_workflow_yaml()
    )
    if resolved is not None:
        os.environ[DRAGENT_CONFIG_FILE_ENV] = str(resolved)
    return resolved
