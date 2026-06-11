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
"""Answer Quality — general-purpose response quality (LLM-as-judge).

The catch-all "is this a good answer?" benchmark. An LLM judge scores each
response on a 1-5 Likert scale for helpfulness, coherence, and relevance to the
question. Use it for open-ended tasks where there is no single correct answer.

Scoring (judge-based):
    likert_5 -> 1..5 mapped to 0.2..1.0; pass threshold is 0.5 (grade >= 3).

Dataset fields (user_datasets/*.json):
    input  (required) the prompt sent to the agent
    notes  (optional) extra grading criteria injected into the judge prompt

Configure the judge in the pipeline YAML's ``judge:`` block. See sibling
judge-free benchmarks (answer_correctness, instruction_following, …) if you want
to avoid a judge model entirely.
"""

import os
from typing import Any

from nemo_evaluator.contrib.byob import ScorerInput
from nemo_evaluator.contrib.byob import benchmark
from nemo_evaluator.contrib.byob import scorer

from datarobot_genai.eval.judge import judge_score  # provider-compatible wrapper (drops top_p)

# Judge endpoint config. ``api_key`` is the NAME of an env var, resolved at
# runtime by the judge (sent as ``Authorization: Bearer <value>``). run.py
# exports JUDGE_* from the pipeline's judge: block before invoking the runner.
JUDGE = {
    "url": os.environ.get("JUDGE_URL", "https://app.datarobot.com/api/v2/genai/llmgw"),
    "model_id": os.environ.get("JUDGE_MODEL_ID", "azure/gpt-5-5-2026-04-23"),
    "api_key": os.environ.get("JUDGE_API_KEY_NAME", "DATAROBOT_API_TOKEN"),
    "temperature": 0.0,
    "max_new_tokens": 1024,
}

# judge_score() returns one of these grades when the judge call itself failed
# (HTTP error after retries) or its output couldn't be parsed. We treat those as
# *inconclusive* (no numeric score) rather than a 0.0 failure.
_JUDGE_ERROR_GRADES = frozenset({"CALL_ERROR", "PARSE_ERROR"})


def _scored(result: dict[str, Any], category_key: str) -> dict[str, Any]:
    """Shape a judge result into a scores dict.

    On a judge error we emit NO numeric key, so aggregation skips the sample and
    evaluator/output.py marks it inconclusive. ``judge_grade`` is always returned
    for traceability in the predictions file.
    """
    grade = result["judge_grade"]
    if grade in _JUDGE_ERROR_GRADES:
        return {"judge_grade": grade}
    score = result["judge_score"]
    return {"score": score, category_key: score, "judge_grade": grade}


@benchmark(  # type: ignore[untyped-decorator]
    name="answer_quality",
    dataset="cases.jsonl",  # placeholder; --dataset overrides at runtime
    prompt="{input}",
    endpoint_type="chat",
    extra={"judge": JUDGE},
)
@scorer  # type: ignore[untyped-decorator]
def score(sample: ScorerInput) -> dict[str, Any]:
    """Likert-5 quality judge over the agent's free-form response."""
    question = sample.metadata.get("input", "")
    criteria = sample.metadata.get("notes", "")
    result = judge_score(sample, template="likert_5", question=question, criteria=criteria)
    return _scored(result, "quality")
