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

import logging
from unittest.mock import MagicMock

import pytest
from colorama import Fore

from datarobot_genai.dragent.frontends.console import DRAgentConsoleFrontEndPlugin
from datarobot_genai.dragent.frontends.step_adaptor import DRAgentNestedReasoningStepAdaptor

# --- _print_result ---


def test_print_result_prints_when_log_level_too_high(capsys):
    """When root logger handlers are above INFO, _print_result falls back to print()."""
    # GIVEN a root logger with only a WARNING-level handler
    root = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)
    root.addHandler(handler)
    try:
        # WHEN we print a result
        DRAgentConsoleFrontEndPlugin._print_result("test output")
        # THEN the output goes through print() fallback
        out = capsys.readouterr().out
        assert "Workflow Result:" in out
        assert "test output" in out
    finally:
        root.removeHandler(handler)


def test_print_result_does_not_double_print_when_info_handler_exists(capsys):
    """When a StreamHandler at INFO level exists, _print_result should not use print()."""
    # GIVEN a root logger with an INFO-level StreamHandler
    root = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    root.addHandler(handler)
    try:
        # WHEN we print a result
        DRAgentConsoleFrontEndPlugin._print_result("test output")
        # THEN the print() fallback does NOT fire (output only via logger)
        out = capsys.readouterr().out
        assert "test output" not in out
    finally:
        root.removeHandler(handler)


def test_print_result_includes_colorama_formatting(capsys):
    # GIVEN a logger that triggers the print() fallback
    root = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)
    root.addHandler(handler)
    try:
        # WHEN we print a result
        DRAgentConsoleFrontEndPlugin._print_result("result")
        # THEN output contains colorama escape codes
        out = capsys.readouterr().out
        assert Fore.GREEN in out
        assert Fore.RESET in out
    finally:
        root.removeHandler(handler)


# --- _get_step_adaptor ---


def test_get_step_adaptor_returns_correct_type():
    # GIVEN a plugin instance (mocked)
    plugin = MagicMock(spec=DRAgentConsoleFrontEndPlugin)
    # WHEN we call _get_step_adaptor
    adaptor = DRAgentConsoleFrontEndPlugin._get_step_adaptor(plugin)
    # THEN it returns a DRAgentNestedReasoningStepAdaptor
    assert isinstance(adaptor, DRAgentNestedReasoningStepAdaptor)


# --- run_workflow ---


async def test_run_workflow_raises_if_session_manager_is_none():
    # GIVEN a plugin with run_workflow bound
    plugin = MagicMock(spec=DRAgentConsoleFrontEndPlugin)
    plugin.run_workflow = DRAgentConsoleFrontEndPlugin.run_workflow.__get__(plugin)
    # WHEN we pass None as session_manager
    # THEN it raises RuntimeError
    with pytest.raises(RuntimeError, match="Session manager must be provided"):
        await plugin.run_workflow(None)


async def test_run_workflow_raises_if_no_input():
    # GIVEN a plugin with no input_query and no input_file
    plugin = MagicMock(spec=DRAgentConsoleFrontEndPlugin)
    plugin.run_workflow = DRAgentConsoleFrontEndPlugin.run_workflow.__get__(plugin)
    plugin._get_step_adaptor = DRAgentConsoleFrontEndPlugin._get_step_adaptor.__get__(plugin)
    plugin.front_end_config.input_query = None
    plugin.front_end_config.input_file = None
    # WHEN we call run_workflow
    # THEN it raises RuntimeError about missing input
    with pytest.raises(RuntimeError, match="No input provided"):
        await plugin.run_workflow(MagicMock())
