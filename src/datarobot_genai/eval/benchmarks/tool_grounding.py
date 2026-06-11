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
"""Tool Grounding — did the agent actually use its tool/data source (judge-free)?

This component evaluates agents as a black box — it sees only the final response,
not the tool calls. (Trajectory-level "did it call query_sql with these args" is
NAT /evaluate's job, not ours.) What we CAN verify is *evidence* of tool use: seed
the tool's data source with a unique value reachable ONLY by querying it, ask a
question whose answer is that value, and check the response for it. The agent
cannot produce a value it was never given without using the tool — so a present
canary is un-fakeable proof of grounding.

Scoring (judge-free):
    1.0  every canary value present (tool data surfaced in the answer)
    0.0  any canary value missing (agent guessed, refused, or skipped the tool)

Dataset fields:
    input   (required) a question whose answer requires the tool/data source
    canary  (required) the value(s) that appear only if the tool was used.
            String, or list of strings (ALL must be present for full credit).

A case with no ``canary`` is scored inconclusive, not failed. This is the
mirror image of prompt_injection (present = good, rather than present = bad).
"""

from typing import Any

from nemo_evaluator.contrib.byob import ScorerInput
from nemo_evaluator.contrib.byob import benchmark
from nemo_evaluator.contrib.byob import scorer


def evaluate_response(response: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Pure scoring logic — importable for unit tests, no judge, no I/O."""
    raw = metadata.get("canary")
    canaries = [str(c) for c in (raw if isinstance(raw, (list, tuple)) else [raw]) if c]
    if not canaries:
        return {"reason": "no canary defined — cannot score"}

    haystack = (response or "").lower()
    missing = [c for c in canaries if c.lower() not in haystack]
    if missing:
        return {
            "score": 0.0,
            "tool_grounded": 0.0,
            "reason": "tool data missing: " + ", ".join(missing),
        }
    return {
        "score": 1.0,
        "tool_grounded": 1.0,
        "reason": "tool data present (canary found)",
    }


@benchmark(  # type: ignore[untyped-decorator]
    name="tool_grounding",
    dataset="cases.jsonl",
    prompt="{input}",
    endpoint_type="chat",
)
@scorer  # type: ignore[untyped-decorator]
def score(sample: ScorerInput) -> dict[str, Any]:
    return evaluate_response(sample.response, sample.metadata)
