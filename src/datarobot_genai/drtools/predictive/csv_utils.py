# Copyright 2025 DataRobot, Inc.
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

"""CSV and row-based data helpers for predictive tools (stdlib-only, no pandas)."""

import csv
import io
from datetime import datetime
from typing import Any


def read_csv_to_rows(
    csv_string: str | None = None, file_path: str | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """Read CSV into list of dicts and column names.

    Args:
        csv_string: CSV content as string, or None.
        file_path: Path to CSV file, or None.

    Returns:
        (rows, column_names). rows is list of dicts; column_names from header.

    Raises:
        ValueError: If neither csv_string nor file_path is provided.
    """
    if csv_string is not None:
        reader = csv.DictReader(io.StringIO(csv_string))
        rows = list(reader)
        columns = list(reader.fieldnames) if reader.fieldnames else []
    elif file_path is not None:
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            columns = list(reader.fieldnames) if reader.fieldnames else []
    else:
        raise ValueError("Must provide either file_path or csv_string.")
    return rows, columns


def is_numeric_column(values: list[Any]) -> bool:
    """Return True if all non-empty values can be cast to float."""
    for v in values:
        if v is None or v == "":
            continue
        try:
            float(v)
        except (TypeError, ValueError):
            return False
    return True


def column_values(rows: list[dict[str, Any]], col: str) -> list[Any]:
    """Extract one column's values from a list of row dicts."""
    return [r.get(col) for r in rows]


def all_null_or_empty(values: list[Any]) -> bool:
    """Return True if every value is None or empty string."""
    return all(v is None or v == "" for v in values)


def can_parse_datetime_column(values: list[Any]) -> bool:
    """Return True if all non-empty values parse as ISO datetime."""
    for v in values:
        if v is None or v == "":
            continue
        try:
            datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        except ValueError:
            return False
    return True
