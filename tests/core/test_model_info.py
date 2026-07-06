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

"""Tests for :mod:`datarobot_genai.core.model_info`."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from datarobot_genai.core.model_info import get_model_info

# These tests patch ``litellm.get_model_info``; litellm is a framework extra, not a
# core dependency, so skip the module when it is absent (e.g. the core-only CI job).
pytest.importorskip("litellm")


def test_non_datarobot_model_passes_through() -> None:
    with patch("litellm.get_model_info", return_value={"ok": True}) as m:
        assert get_model_info("gpt-4o") == {"ok": True}
    m.assert_called_once_with("gpt-4o")


def test_datarobot_prefix_is_stripped_before_lookup() -> None:
    with patch("litellm.get_model_info", return_value={"ok": True}) as m:
        get_model_info("datarobot/vertex_ai/gemini-2.5-flash")
    m.assert_called_once_with("vertex_ai/gemini-2.5-flash")


@pytest.mark.parametrize(
    ("model", "looked_up"),
    [
        ("datarobot/azure/gpt-5-1-2025-11-13", "azure/gpt-5.1-2025-11-13"),  # dash version -> dot
        ("datarobot/azure/gpt-4-1-2025-04-14", "azure/gpt-4.1-2025-04-14"),
        ("datarobot/azure/gpt-4o-2024-11-20", "azure/gpt-4o-2024-11-20"),  # unchanged
        ("datarobot/azure/gpt-4-32k", "azure/gpt-4-32k"),  # size, not version -> unchanged
    ],
)
def test_azure_deployment_name_is_normalized(model: str, looked_up: str) -> None:
    """Azure dashed versions are dotted to litellm keys before lookup."""
    with patch("litellm.get_model_info", return_value={"ok": True}) as m:
        get_model_info(model)
    m.assert_called_once_with(looked_up)


def test_lookup_failure_propagates() -> None:
    with (
        patch("litellm.get_model_info", side_effect=Exception("not mapped")),
        pytest.raises(Exception, match="not mapped"),
    ):
        get_model_info("datarobot/vertex_ai/brand-new-model")
