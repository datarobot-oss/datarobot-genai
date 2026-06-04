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


def load_dataset(path: str) -> list[dict[str, Any]]:
    text = Path(path).read_text()
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    result: list[dict[str, Any]] = json.loads(text)
    return result


def to_byob_jsonl(dataset: list[dict[str, Any]], output_path: str) -> None:
    """Write the dataset to the JSONL the BYOB runner consumes.

    Every field of each case is passed through verbatim — NeMo BYOB places the
    whole row into ``sample.metadata``, so a benchmark can read any field it
    needs (``context``, ``canary``, ``constraints``, ``match_mode``,
    ``entity_types``, …) with no change required here. We only guarantee the two
    keys the runner itself relies on: ``input`` (the prompt template key,
    defaulted so a malformed row surfaces in the scorer rather than at render
    time) and ``id`` (required — a missing one is a genuine error).
    """
    lines = []
    for case in dataset:
        row = {"input": "", **case, "id": case["id"]}
        lines.append(json.dumps(row))
    Path(output_path).write_text("\n".join(lines))
