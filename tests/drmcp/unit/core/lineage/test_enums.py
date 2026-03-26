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
import os
from collections.abc import Iterator
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from datarobot_genai.drmcp.core.lineage.enums import LRSEnvVarIsNotSetError
from datarobot_genai.drmcp.core.lineage.enums import LRSEnvVars


class TestLRSEnvVars:
    @pytest.fixture
    def mock_os_getenv(self) -> Iterator[Mock]:
        with patch.object(os, "getenv") as mock_func:
            yield mock_func

    @pytest.mark.parametrize("lrs_env_var", [lrs_env_var for lrs_env_var in LRSEnvVars])
    def test_get_os_env_value(self, mock_os_getenv: Mock, lrs_env_var: LRSEnvVars) -> None:
        lrs_env_var.get_os_env_value()

        mock_os_getenv.assert_called_with(lrs_env_var.name)

    @pytest.mark.parametrize("lrs_env_var", [lrs_env_var for lrs_env_var in LRSEnvVars])
    def test_get_os_env_value_raise_error(
        self, mock_os_getenv: Mock, lrs_env_var: LRSEnvVars
    ) -> None:
        mock_os_getenv.return_value = None

        with pytest.raises(LRSEnvVarIsNotSetError):
            lrs_env_var.get_os_env_value()
