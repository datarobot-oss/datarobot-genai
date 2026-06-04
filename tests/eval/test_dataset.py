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
from typing import Any

import pytest

from datarobot_genai.eval.dataset import load_dataset
from datarobot_genai.eval.dataset import to_byob_jsonl

# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------


def test_load_dataset_json(tmp_path: Path, minimal_cases: list[dict[str, Any]]) -> None:
    p = tmp_path / "cases.json"
    p.write_text(json.dumps(minimal_cases))
    loaded = load_dataset(str(p))
    assert len(loaded) == len(minimal_cases)
    assert loaded[0]["id"] == "good-001"


def test_load_dataset_jsonl(tmp_path: Path, minimal_cases: list[dict[str, Any]]) -> None:
    p = tmp_path / "cases.jsonl"
    p.write_text("\n".join(json.dumps(c) for c in minimal_cases))
    loaded = load_dataset(str(p))
    assert len(loaded) == len(minimal_cases)
    assert loaded[1]["id"] == "bad-001"


def test_load_dataset_jsonl_ignores_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "cases.jsonl"
    p.write_text('{"id": "a"}\n\n{"id": "b"}\n')
    loaded = load_dataset(str(p))
    assert len(loaded) == 2


def test_load_dataset_json_non_list_raises(tmp_path: Path) -> None:
    p = tmp_path / "cases.json"
    p.write_text('{"id": "a"}')
    with pytest.raises(TypeError, match="expected a JSON array"):
        load_dataset(str(p))


# ---------------------------------------------------------------------------
# to_byob_jsonl
# ---------------------------------------------------------------------------


def test_to_byob_jsonl_writes_one_line_per_case(
    tmp_path: Path, minimal_cases: list[dict[str, Any]]
) -> None:
    out = tmp_path / "out.jsonl"
    to_byob_jsonl(minimal_cases, str(out))
    lines = [line for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) == len(minimal_cases)


def test_to_byob_jsonl_required_fields(tmp_path: Path, minimal_cases: list[dict[str, Any]]) -> None:
    out = tmp_path / "out.jsonl"
    to_byob_jsonl(minimal_cases, str(out))
    row = json.loads(out.read_text().splitlines()[0])
    assert set(row.keys()) == {
        "id",
        "input",
        "ideal_response",
        "expected_behavior",
        "notes",
        "source",
    }


def test_to_byob_jsonl_guarantees_input_and_id(tmp_path: Path) -> None:
    # Only `input` (defaulted) and `id` are guaranteed; optional fields are not
    # fabricated — they pass through if present, and are simply absent otherwise.
    cases = [{"id": "x"}]
    out = tmp_path / "out.jsonl"
    to_byob_jsonl(cases, str(out))
    row = json.loads(out.read_text())
    assert row["id"] == "x"
    assert row["input"] == ""
    assert "expected_behavior" not in row
    assert "notes" not in row


def test_to_byob_jsonl_passes_through_arbitrary_fields(tmp_path: Path) -> None:
    # Benchmark-specific fields (context, canary, constraints, …) must survive
    # verbatim so any benchmark can read them from sample.metadata.
    cases = [
        {
            "id": "x",
            "input": "hi",
            "context": "some passage",
            "canary": ["A", "B"],
            "constraints": {"max_words": 5},
            "match_mode": "contains",
        }
    ]
    out = tmp_path / "out.jsonl"
    to_byob_jsonl(cases, str(out))
    row = json.loads(out.read_text())
    assert row["context"] == "some passage"
    assert row["canary"] == ["A", "B"]
    assert row["constraints"] == {"max_words": 5}
    assert row["match_mode"] == "contains"


def test_to_byob_jsonl_preserves_null_ideal_response(tmp_path: Path) -> None:
    cases = [{"id": "x", "input": "hi", "ideal_response": None}]
    out = tmp_path / "out.jsonl"
    to_byob_jsonl(cases, str(out))
    row = json.loads(out.read_text())
    assert row["ideal_response"] is None
