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
"""Instruction Following — does the response obey explicit constraints (judge-free)?

Checks structural constraints the prompt asked for: length limits, valid JSON,
required/forbidden phrases, regex shape. Deterministic, so it is reproducible and
needs no judge. Semantic constraints ("use a professional tone") are out of scope
here — use answer_quality for those.

Scoring (judge-free): fraction of specified constraints satisfied, in [0, 1].
With the 0.5 pass threshold, a response must satisfy at least half its
constraints to pass; the ``reason`` lists which ones failed.

Dataset fields:
    input        (required) the prompt sent to the agent
    constraints  (required) object with any of:
        max_words / min_words  (int)   word-count bounds
        max_chars              (int)   character-count upper bound
        must_be_json           (bool)  response must parse as JSON
        must_include           (str | list[str]) substrings that must appear
        must_exclude           (str | list[str]) substrings that must NOT appear
        regex                  (str)   pattern that must match somewhere

A case with no ``constraints`` is scored inconclusive, not failed.
"""

import json
import re
from typing import Any

from nemo_evaluator.contrib.byob import ScorerInput, benchmark, scorer


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _strip_code_fences(text: str) -> str:
    """Drop a leading ```json / ``` fence and trailing ``` if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _is_json(text: str) -> bool:
    try:
        json.loads(_strip_code_fences(text))
        return True
    except (ValueError, TypeError):
        return False


def evaluate_response(response: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Pure scoring logic — importable for unit tests, no judge, no I/O."""
    constraints = metadata.get("constraints") or {}
    if not constraints:
        return {"reason": "no constraints specified — cannot score"}

    resp = response or ""
    words = len(resp.split())
    checks: list[tuple[str, bool]] = []

    if "max_words" in constraints:
        limit = int(constraints["max_words"])
        checks.append((f"max_words<={limit} (got {words})", words <= limit))
    if "min_words" in constraints:
        limit = int(constraints["min_words"])
        checks.append((f"min_words>={limit} (got {words})", words >= limit))
    if "max_chars" in constraints:
        limit = int(constraints["max_chars"])
        checks.append((f"max_chars<={limit} (got {len(resp)})", len(resp) <= limit))
    if constraints.get("must_be_json"):
        checks.append(("must_be_json", _is_json(resp)))
    for needle in _as_list(constraints.get("must_include")):
        checks.append((f"must_include '{needle}'", needle.lower() in resp.lower()))
    for needle in _as_list(constraints.get("must_exclude")):
        checks.append((f"must_exclude '{needle}'", needle.lower() not in resp.lower()))
    if constraints.get("regex"):
        pattern = str(constraints["regex"])
        checks.append((f"regex /{pattern}/", re.search(pattern, resp) is not None))

    if not checks:
        return {"reason": "no recognized constraints — cannot score"}

    passed = sum(1 for _, ok in checks if ok)
    value = passed / len(checks)
    failed = [label for label, ok in checks if not ok]
    reason = (
        f"all {len(checks)} constraints satisfied"
        if not failed
        else f"{passed}/{len(checks)} satisfied; failed: " + "; ".join(failed)
    )
    return {"score": value, "instruction_following": value, "reason": reason}


@benchmark(  # type: ignore[untyped-decorator]
    name="instruction_following",
    dataset="cases.jsonl",
    prompt="{input}",
    endpoint_type="chat",
)
@scorer  # type: ignore[untyped-decorator]
def score(sample: ScorerInput) -> dict[str, Any]:
    return evaluate_response(sample.response, sample.metadata)
