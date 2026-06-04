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
import json
from pathlib import Path

from datarobot_genai.eval.status import write_status


def test_write_status_creates_file(tmp_path: Path) -> None:
    write_status(
        "running", "20260601_120000", "test.yaml", "http://localhost/v1", tmp_path
    )
    assert (tmp_path / "eval_status.json").exists()


def test_write_status_fields(tmp_path: Path) -> None:
    write_status(
        "complete", "20260601_120000", "test.yaml", "http://localhost/v1", tmp_path
    )
    data = json.loads((tmp_path / "eval_status.json").read_text())

    assert data["status"] == "complete"
    assert data["run_id"] == "20260601_120000"
    assert data["pipeline"] == "test.yaml"
    assert data["agent_endpoint"] == "http://localhost/v1"
    assert data["error"] is None
    assert "updated_at" in data


def test_write_status_with_error(tmp_path: Path) -> None:
    write_status(
        "failed", "run-1", "p.yaml", "http://x", tmp_path, error="something broke"
    )
    data = json.loads((tmp_path / "eval_status.json").read_text())
    assert data["status"] == "failed"
    assert data["error"] == "something broke"


def test_write_status_creates_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "nested" / "output"
    write_status("running", "run-1", "p.yaml", "http://x", output_dir)
    assert (output_dir / "eval_status.json").exists()


def test_write_status_overwrites(tmp_path: Path) -> None:
    write_status("running", "run-1", "p.yaml", "http://x", tmp_path)
    write_status("complete", "run-1", "p.yaml", "http://x", tmp_path)
    data = json.loads((tmp_path / "eval_status.json").read_text())
    assert data["status"] == "complete"
