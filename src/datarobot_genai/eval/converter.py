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
import csv
import json
import warnings
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {"id", "source", "input"}


def convert_csv_to_cases(csv_path: Path) -> list[dict[str, Any]]:
    """Read a CSV dataset and return a list of case dicts.

    Required columns: id, source, input.
    All other columns are preserved as-is.
    Empty cells come through as empty strings (CSV has no null).
    """
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"{csv_path}: CSV has no header row")

        headers = set(reader.fieldnames)
        missing = REQUIRED_FIELDS - headers
        if missing:
            raise ValueError(f"{csv_path}: missing required columns: {sorted(missing)}")

        if "notes" not in headers:
            warnings.warn(
                f"{csv_path}: no 'notes' column found — notes are recommended for "
                "describing expected behavior per case",
                UserWarning,
                stacklevel=2,
            )

        return [dict(row) for row in reader]


def save_cases(cases: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
