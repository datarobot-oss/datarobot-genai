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

import pytest

from datarobot_genai.eval.converter import convert_csv_to_cases, save_cases


def _write_csv(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# convert_csv_to_cases
# ---------------------------------------------------------------------------


def test_convert_basic(tmp_path: Path) -> None:
    csv = _write_csv(
        tmp_path / "cases.csv",
        "id,source,input,notes\nq-001,collected,What is RAG?,Technical question\n",
    )
    cases = convert_csv_to_cases(csv)
    assert len(cases) == 1
    assert cases[0]["id"] == "q-001"
    assert cases[0]["source"] == "collected"
    assert cases[0]["input"] == "What is RAG?"
    assert cases[0]["notes"] == "Technical question"


def test_convert_multiple_rows(tmp_path: Path) -> None:
    csv = _write_csv(
        tmp_path / "cases.csv",
        "id,source,input,notes\n"
        "q-001,collected,First input,Note one\n"
        "q-002,synthetic,Second input,Note two\n",
    )
    cases = convert_csv_to_cases(csv)
    assert len(cases) == 2
    assert cases[1]["id"] == "q-002"


def test_convert_extra_columns_preserved(tmp_path: Path) -> None:
    csv = _write_csv(
        tmp_path / "cases.csv",
        "id,source,input,notes,expected_behavior,ideal_response\n"
        "q-001,collected,Hello,A note,good,The answer\n",
    )
    cases = convert_csv_to_cases(csv)
    assert cases[0]["expected_behavior"] == "good"
    assert cases[0]["ideal_response"] == "The answer"


def test_convert_empty_rows_skipped(tmp_path: Path) -> None:
    # csv.DictReader skips lines that are entirely blank
    csv = _write_csv(
        tmp_path / "cases.csv",
        "id,source,input,notes\nq-001,collected,Hello,A note\n\n",
    )
    cases = convert_csv_to_cases(csv)
    assert len(cases) == 1


def test_convert_quoted_fields(tmp_path: Path) -> None:
    csv = _write_csv(
        tmp_path / "cases.csv",
        "id,source,input,notes\n"
        'q-001,collected,"Input with, comma","Note with ""quotes"""\n',
    )
    cases = convert_csv_to_cases(csv)
    assert cases[0]["input"] == "Input with, comma"
    assert cases[0]["notes"] == 'Note with "quotes"'


def test_convert_missing_required_field_raises(tmp_path: Path) -> None:
    csv = _write_csv(
        tmp_path / "cases.csv",
        "id,source\nq-001,collected\n",  # missing 'input'
    )
    with pytest.raises(ValueError, match="missing required columns"):
        convert_csv_to_cases(csv)


def test_convert_missing_multiple_required_fields_raises(tmp_path: Path) -> None:
    csv = _write_csv(
        tmp_path / "cases.csv",
        "notes\nSome note\n",  # missing id, source, input
    )
    with pytest.raises(ValueError, match="missing required columns"):
        convert_csv_to_cases(csv)


def test_convert_no_notes_warns(tmp_path: Path) -> None:
    csv = _write_csv(
        tmp_path / "cases.csv",
        "id,source,input\nq-001,collected,Hello\n",
    )
    with pytest.warns(UserWarning, match="notes"):
        convert_csv_to_cases(csv)


def test_convert_no_header_raises(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "cases.csv", "")
    with pytest.raises(ValueError, match="no header row"):
        convert_csv_to_cases(csv)


def test_convert_header_only_returns_empty(tmp_path: Path) -> None:
    csv = _write_csv(
        tmp_path / "cases.csv",
        "id,source,input,notes\n",
    )
    cases = convert_csv_to_cases(csv)
    assert cases == []


# ---------------------------------------------------------------------------
# save_cases
# ---------------------------------------------------------------------------


def test_save_cases_writes_json(tmp_path: Path) -> None:
    cases = [
        {"id": "q-001", "source": "collected", "input": "Hello", "notes": "A note"}
    ]
    out = tmp_path / "output.json"
    save_cases(cases, out)
    written = json.loads(out.read_text())
    assert written == cases


def test_save_cases_creates_parent_dirs(tmp_path: Path) -> None:
    cases = [{"id": "q-001", "source": "collected", "input": "Hello"}]
    out = tmp_path / "nested" / "dir" / "output.json"
    save_cases(cases, out)
    assert out.exists()


def test_save_cases_overwrites(tmp_path: Path) -> None:
    out = tmp_path / "output.json"
    save_cases([{"id": "q-001", "source": "x", "input": "old"}], out)
    save_cases([{"id": "q-002", "source": "x", "input": "new"}], out)
    written = json.loads(out.read_text())
    assert len(written) == 1
    assert written[0]["id"] == "q-002"
