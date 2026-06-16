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
"""Answer Correctness — deterministic match against a known answer (judge-free).

The regression-testing workhorse: when you DO have ground-truth answers, check
the agent's response against them without paying for a judge model. Fully
deterministic, so results are reproducible run-to-run.

Scoring (judge-free): 1.0 on match, 0.0 otherwise, by ``match_mode``:
    exact       response equals reference after trimming whitespace
    normalized  equal after lowercasing + stripping punctuation/extra spaces
                (the default — tolerant of formatting noise)
    contains    normalized reference appears somewhere in the normalized response

Dataset fields:
    input          (required) the prompt sent to the agent
    ideal_response (required) the ground-truth answer to match against
    match_mode     (optional) exact | normalized | contains  (default normalized)

A case with no ``ideal_response`` is scored inconclusive, not failed.
"""

import re
from typing import Any

from nemo_evaluator.contrib.byob import ScorerInput
from nemo_evaluator.contrib.byob import benchmark
from nemo_evaluator.contrib.byob import scorer

_VALID_MODES = frozenset({"exact", "normalized", "contains"})


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def evaluate_response(response: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Pure scoring logic — importable for unit tests, no judge, no I/O."""
    reference = metadata.get("ideal_response")
    if reference is None or not str(reference).strip():
        return {"reason": "no ideal_response provided — cannot score"}

    mode = str(metadata.get("match_mode") or "normalized").lower()
    if mode not in _VALID_MODES:
        mode = "normalized"

    resp = response or ""
    ref = str(reference)
    if mode == "exact":
        ok = resp.strip() == ref.strip()
    elif mode == "contains":
        ok = _normalize(ref) in _normalize(resp)
    else:  # normalized
        ok = _normalize(resp) == _normalize(ref)

    value = 1.0 if ok else 0.0
    reason = f"{mode} match" if ok else f"no {mode} match"
    return {"score": value, "correctness": value, "reason": reason}


@benchmark(  # type: ignore[untyped-decorator]
    name="answer_correctness",
    dataset="cases.jsonl",
    prompt="{input}",
    endpoint_type="chat",
)
@scorer  # type: ignore[untyped-decorator]
def score(sample: ScorerInput) -> dict[str, Any]:
    return evaluate_response(sample.response, sample.metadata)
