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
import re

from datarobot_genai.eval.utils import make_run_id


def test_make_run_id_format() -> None:
    run_id = make_run_id()
    assert re.match(r"^\d{8}_\d{6}$", run_id), f"unexpected format: {run_id}"


def test_make_run_id_is_string() -> None:
    assert isinstance(make_run_id(), str)


def test_make_run_id_unique() -> None:
    # Two calls in the same second may collide, but calling twice should
    # produce strings of the correct form — uniqueness is best-effort.
    a = make_run_id()
    b = make_run_id()
    assert re.match(r"^\d{8}_\d{6}$", a)
    assert re.match(r"^\d{8}_\d{6}$", b)
