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

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from datarobot_genai.dragent.constants import DRAGENT_CONFIG_FILE_ENV
from datarobot_genai.dragent.workflow_paths import CODE_DIR_ENV
from datarobot_genai.dragent.workflow_paths import discover_workflow_yaml
from datarobot_genai.dragent.workflow_paths import publish_dragent_config_file_env


def test_discover_workflow_yaml_from_absolute_dragent_config_file(tmp_path: Path) -> None:
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text("workflow: {}\n", encoding="utf-8")
    with patch.dict(os.environ, {DRAGENT_CONFIG_FILE_ENV: str(workflow)}, clear=False):
        assert discover_workflow_yaml() == workflow.resolve()


def test_discover_workflow_yaml_from_relative_dragent_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent_dir = tmp_path / "opt" / "code"
    agent_dir.mkdir(parents=True)
    workflow = agent_dir / "workflow.yaml"
    workflow.write_text("workflow: {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with patch.dict(
        os.environ,
        {DRAGENT_CONFIG_FILE_ENV: "opt/code/workflow.yaml"},
        clear=False,
    ):
        assert discover_workflow_yaml() == workflow.resolve()


def test_discover_workflow_yaml_from_code_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent_dir = tmp_path / "opt" / "code"
    agent_dir.mkdir(parents=True)
    workflow = agent_dir / "workflow.yaml"
    workflow.write_text("workflow: {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "other").mkdir()
    monkeypatch.chdir(tmp_path / "other")
    monkeypatch.delenv(DRAGENT_CONFIG_FILE_ENV, raising=False)
    with patch.dict(os.environ, {CODE_DIR_ENV: str(agent_dir)}, clear=False):
        assert discover_workflow_yaml() == workflow.resolve()


def test_discover_workflow_yaml_walks_cwd_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent_dir = tmp_path / "project" / "agent"
    agent_dir.mkdir(parents=True)
    workflow = agent_dir / "workflow.yaml"
    workflow.write_text("workflow: {}\n", encoding="utf-8")
    nested = agent_dir / "nested"
    nested.mkdir()
    monkeypatch.chdir(nested)
    monkeypatch.delenv(DRAGENT_CONFIG_FILE_ENV, raising=False)
    monkeypatch.delenv(CODE_DIR_ENV, raising=False)
    assert discover_workflow_yaml() == workflow.resolve()


def test_publish_dragent_config_file_env_sets_absolute_path(tmp_path: Path) -> None:
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text("workflow: {}\n", encoding="utf-8")
    published = publish_dragent_config_file_env(workflow)
    assert published == workflow.resolve()
    assert os.environ[DRAGENT_CONFIG_FILE_ENV] == str(workflow.resolve())


def test_discover_workflow_yaml_relative_hint_with_code_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deployment layout: CWD is not ``CODE_DIR`` but hint is ``workflow.yaml``."""
    agent_dir = tmp_path / "opt" / "code"
    agent_dir.mkdir(parents=True)
    workflow = agent_dir / "workflow.yaml"
    workflow.write_text("workflow: {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with patch.dict(
        os.environ,
        {DRAGENT_CONFIG_FILE_ENV: "workflow.yaml", CODE_DIR_ENV: str(agent_dir)},
        clear=False,
    ):
        assert discover_workflow_yaml() == workflow.resolve()
