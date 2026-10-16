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
"""Response Acceptability: how acceptable is the response given example answers (LLM-as-judge).

The judge-based twin of response_similarity, reading the same dataset. Instead
of counting shared words, a judge sees the example answers and rates how well
the agent's response matches them in substance. It catches paraphrases that the
lexical metrics miss, at the cost of a judge call per case.

The examples are illustrations, not an exhaustive list: a response can score 5
without matching any of them word for word.

Scoring (judge-based): the judge rates 1-5 on the rubric below, mapped to
0.2..1.0. With the default 0.5 pass threshold, a rating of 3 or more passes.
The ``reason`` repeats the rubric line for the rating the judge gave.

Dataset fields:
    input                   (required) the prompt sent to the agent
    acceptable_responses    (required) str | list[str] example good answers
    unacceptable_responses  (optional) str | list[str] example bad answers
    notes                   (optional) extra grading guidance for the judge

``metric`` is ignored here, so one dataset can drive both benchmarks. A case
with no ``acceptable_responses`` is scored inconclusive.
"""

import os
from typing import Any

from nemo_evaluator.contrib.byob import ScorerInput
from nemo_evaluator.contrib.byob import benchmark
from nemo_evaluator.contrib.byob import scorer

from datarobot_genai.eval.judge import judge_score  # provider-compatible wrapper (drops top_p)

JUDGE = {
    "url": os.environ.get("JUDGE_URL", "https://app.datarobot.com/api/v2/genai/llmgw"),
    "model_id": os.environ.get("JUDGE_MODEL_ID", "azure/gpt-5-5-2026-04-23"),
    "api_key": os.environ.get("JUDGE_API_KEY_NAME", "DATAROBOT_API_TOKEN"),
    "temperature": 0.0,
    "max_new_tokens": 1024,
}

_JUDGE_ERROR_GRADES = frozenset({"CALL_ERROR", "PARSE_ERROR"})

RUBRIC = {
    "5": "same substance as an acceptable example; only wording, length, or format differ",
    "4": "same substance with a minor omission or small inaccuracy",
    "3": "partly matches: some key points present, others missing or wrong",
    "2": "mostly misses the key point, or is closer to an unacceptable example",
    "1": "contradicts the examples or answers a different question",
}
_SCORE_MAPPING = {grade: int(grade) / 5 for grade in RUBRIC}

_TEMPLATE = (
    "You are an expert evaluation judge. Rate how acceptable the model's response "
    "is, given example answers that are known to be acceptable.\n\n"
    "Question: {question}\n"
    "Acceptable example answers:\n{reference}\n"
    "Model Response: {response}\n"
    "{criteria}\n\n"
    "The examples illustrate acceptable answers; they are not an exhaustive list. "
    "Judge substance, not wording. Use the following rubric:\n"
    + "\n".join(f"{grade} - {text}" for grade, text in sorted(RUBRIC.items(), reverse=True))
    + "\n\nThink step by step, then output your final rating on a new line in exactly "
    "this format:\nGRADE: 5\nwhere the number is your rating from 1 to 5."
)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)] if str(value).strip() else []


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def build_criteria(unacceptable: list[str], notes: str) -> str:
    """Build the optional criteria block: unacceptable examples and extra guidance."""
    parts = []
    if unacceptable:
        parts.append(f"Unacceptable example answers:\n{_bullets(unacceptable)}")
    if notes:
        parts.append(f"Additional guidance: {notes}")
    return "\n".join(parts)


def score_sample(sample: Any) -> dict[str, Any]:
    """Scoring logic behind ``score``, separate so tests can call it undecorated."""
    acceptable = _as_list(sample.metadata.get("acceptable_responses"))
    if not acceptable:
        return {"reason": "no acceptable_responses provided: cannot score"}
    result = judge_score(
        sample,
        template=_TEMPLATE,
        grade_pattern=r"GRADE:\s*([1-5])",
        score_mapping=_SCORE_MAPPING,
        question=sample.metadata.get("input", ""),
        reference=_bullets(acceptable),
        criteria=build_criteria(
            _as_list(sample.metadata.get("unacceptable_responses")),
            sample.metadata.get("notes", ""),
        ),
    )
    grade = result["judge_grade"]
    if grade in _JUDGE_ERROR_GRADES:
        return {"judge_grade": grade}
    value = result["judge_score"]
    return {
        "score": value,
        "acceptability": value,
        "judge_grade": grade,
        "reason": f"judge rated {grade}/5: {RUBRIC.get(grade, 'unrecognized grade')}",
    }


@benchmark(  # type: ignore[untyped-decorator]
    name="response_acceptability",
    dataset="cases.jsonl",
    prompt="{input}",
    endpoint_type="chat",
    extra={"judge": JUDGE},
)
@scorer  # type: ignore[untyped-decorator]
def score(sample: ScorerInput) -> dict[str, Any]:
    """Judge how acceptable the response is given the case's example answers."""
    return score_sample(sample)
