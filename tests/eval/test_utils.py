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

import pytest

from datarobot_genai.eval.utils import make_run_id
from datarobot_genai.eval.utils import resolve_archive_name

_RUN_ID_RE = r"^\d{8}_\d{6}_\d{6}$"


def test_make_run_id_format() -> None:
    run_id = make_run_id()
    assert re.match(_RUN_ID_RE, run_id), f"unexpected format: {run_id}"


def test_make_run_id_is_string() -> None:
    assert isinstance(make_run_id(), str)


def test_make_run_id_unique() -> None:
    # The run ID names the archived results file, so two runs starting in the
    # same second must not collide. Microsecond precision makes that practical.
    a = make_run_id()
    b = make_run_id()
    assert a != b
    assert re.match(_RUN_ID_RE, a)
    assert re.match(_RUN_ID_RE, b)


def test_make_run_id_sorts_chronologically() -> None:
    # Consumers listing the output directory order runs by filename.
    assert make_run_id() < make_run_id()


# ---------------------------------------------------------------------------
# resolve_archive_name
# ---------------------------------------------------------------------------


def test_archive_name_defaults_to_pipeline_and_run_id() -> None:
    name = resolve_archive_name(None, "answer_quality.yaml", "20260601_120000_000001")
    assert name == "answer_quality_20260601_120000_000001.json"


def test_archive_name_default_strips_pipeline_directory() -> None:
    # The pipeline argument is a path relative to user_pipelines/; only its stem
    # may reach the filename, so the archive cannot escape the output directory.
    name = resolve_archive_name(None, "nested/answer_quality.yaml", "20260601_120000_000001")
    assert name == "answer_quality_20260601_120000_000001.json"


def test_archive_name_explicit_override() -> None:
    assert resolve_archive_name("baseline.json", "p.yaml", "rid") == "baseline.json"


def test_archive_name_appends_json_suffix() -> None:
    assert resolve_archive_name("baseline", "p.yaml", "rid") == "baseline.json"


@pytest.mark.parametrize(
    "bad",
    ["../escape", "sub/dir/name", "/absolute/name", ".hidden"],
)
def test_archive_name_rejects_non_bare_filenames(bad: str) -> None:
    with pytest.raises(ValueError, match="bare filename"):
        resolve_archive_name(bad, "p.yaml", "rid")


@pytest.mark.parametrize("reserved", ["eval_results.json", "eval_status.json"])
def test_archive_name_rejects_reserved_names(reserved: str) -> None:
    with pytest.raises(ValueError, match="reserved"):
        resolve_archive_name(reserved, "p.yaml", "rid")


def test_archive_name_empty_override_falls_back_to_default() -> None:
    assert resolve_archive_name("", "p.yaml", "rid") == "p_rid.json"
