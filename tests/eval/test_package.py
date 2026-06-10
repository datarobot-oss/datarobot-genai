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

"""Temporary stub tests for the eval subpackage until the full implementation lands."""

import datarobot_genai.eval


def test_eval_package_imports() -> None:
    assert datarobot_genai.eval.__name__ == "datarobot_genai.eval"


def test_eval_extra_dependencies_import() -> None:
    """The eval extra must be self-contained: its declared deps importable in isolation."""
    import nemo_evaluator_launcher
    import yaml

    assert nemo_evaluator_launcher is not None
    assert yaml is not None
