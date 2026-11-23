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
import pytest

# The quickstart notebook loads a hardcoded PromptTemplate owned by the E2E account that
# IR-991 disabled; the replacement account gets a 404. Strict, so a pass turns the run red.
QUICKSTART_XFAIL = pytest.mark.xfail(
    reason="ALIFE-2051: PromptTemplate 69eb56c5cc06eff822e8439f is unreachable after IR-991",
    strict=True,
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.path.name == "quickstart.ipynb":
            item.add_marker(QUICKSTART_XFAIL)
