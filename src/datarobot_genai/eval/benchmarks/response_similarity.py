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
"""Response Similarity: lexical or embedding similarity to example answers (judge-free).

The fuzzy sibling of answer_correctness. Where answer_correctness asks "did the
agent say exactly this?", this asks "how close did the agent get to one of
these?". Each case lists one or more acceptable responses, and the score is the
response's similarity to the closest one.

Lexical metrics compare words, not meaning, so a paraphrase with no shared words
("30 days" vs "a month") scores low. ``embedding_cosine`` tolerates rewording but
measures topic, not facts. Use response_acceptability (the judge-based twin,
same dataset shape) when correctness matters.

Lexical metrics, all in [0, 1], computed on lowercased words with punctuation
and common stopwords removed:
    token_recall  fraction of the example's words that appear in the response
                  (the default; tolerant of verbose agents)
    token_f1      balances covering the example against adding extra words
                  (SQuAD-style); penalizes long responses to short examples
    rouge_l       longest run of words the two share in the same order
                  (ROUGE-L F-measure); rewards matching word order

Embedding metric, in [0, 1] (negative cosines are clamped to 0):
    embedding_cosine  cosine similarity of the two texts' embeddings, from a
                      local fastembed model (``benchmark.params.embedding_model``,
                      default BAAI/bge-small-en-v1.5, ~70MB downloaded on first
                      use to FASTEMBED_CACHE_PATH). With the default model,
                      unrelated text scores about 0.4 and a near-miss wrong
                      answer ("90 degrees" vs "100 degrees") can score as high as
                      a correct paraphrase, so raise run.pass_threshold well
                      above 0.5 and read it as "on topic", not "correct".

Scoring (judge-free): the score is the raw similarity to the closest acceptable
example, not a pass/fail. ``passed`` compares it against the pipeline's
``run.pass_threshold`` (default 0.5), which users tune to their data. The
``reason`` says in plain words what the number means and which example matched.

``unacceptable_responses`` do not change the score. When given, the reason also
reports the closest unacceptable example and flags the case when the response is
at least as close to it as to any acceptable one.

Pick the default metric in the pipeline YAML; a case can override it:
    benchmark:
      params:
        metric: token_recall
        embedding_model: BAAI/bge-small-en-v1.5   # embedding_cosine only

Dataset fields:
    input                   (required) the prompt sent to the agent
    acceptable_responses    (required) str | list[str] example good answers
    unacceptable_responses  (optional) str | list[str] example bad answers
    metric                  (optional) per-case override of the pipeline metric

A case with no ``acceptable_responses`` is scored inconclusive, not failed.
"""

import json
import os
import re
import threading
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np
from nemo_evaluator.contrib.byob import ScorerInput
from nemo_evaluator.contrib.byob import benchmark
from nemo_evaluator.contrib.byob import scorer

DEFAULT_METRIC = "token_recall"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_METRIC = "embedding_cosine"

# Pipeline `benchmark.params`, exported by runner.py as JSON.
PARAMS: dict[str, Any] = json.loads(os.environ.get("BENCHMARK_PARAMS") or "{}")

# Kept short on purpose: only words that carry almost no meaning, so dropping
# them stops "the"/"of" overlap from inflating scores without hiding content.
_STOPWORDS = frozenset(
    "a an the is are was were be been being am of to in on at for and or but "
    "it its this that these those with as by from i you we they he she".split()
)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)] if str(value).strip() else []


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, minus stopwords (unless that would leave nothing)."""
    words = re.findall(r"\w+", text.lower())
    content = [w for w in words if w not in _STOPWORDS]
    return content or words


def _token_recall(response: list[str], reference: list[str]) -> float:
    if not reference:
        return 0.0
    overlap = sum((Counter(response) & Counter(reference)).values())
    return overlap / len(reference)


def _f_measure(matched: int, response: list[str], reference: list[str]) -> float:
    if matched == 0:
        return 0.0
    precision = matched / len(response)
    recall = matched / len(reference)
    return 2 * precision * recall / (precision + recall)


def _token_f1(response: list[str], reference: list[str]) -> float:
    overlap = sum((Counter(response) & Counter(reference)).values())
    return _f_measure(overlap, response, reference)


def _lcs_length(a: list[str], b: list[str]) -> int:
    # Single-row DP keeps memory at O(len(b)) for long agent responses.
    row = [0] * (len(b) + 1)
    for x in a:
        diag = 0
        for j, y in enumerate(b, start=1):
            above = row[j]
            row[j] = diag + 1 if x == y else max(above, row[j - 1])
            diag = above
    return row[-1]


def _rouge_l(response: list[str], reference: list[str]) -> float:
    return _f_measure(_lcs_length(response, reference), response, reference)


_LEXICAL_METRICS: dict[str, Callable[[list[str], list[str]], float]] = {
    "token_recall": _token_recall,
    "token_f1": _token_f1,
    "rouge_l": _rouge_l,
}
METRICS = frozenset({*_LEXICAL_METRICS, EMBEDDING_METRIC})

# One-line plain-English meaning of each metric, shown in every case's reason.
_EXPLANATIONS = {
    "token_recall": "share of the example's words found in the response",
    "token_f1": "word overlap, penalizing words the example doesn't have",
    "rouge_l": "longest run of words shared in the same order",
    EMBEDDING_METRIC: "closeness in meaning; tracks topic, not factual accuracy",
}

# Loaded fastembed models by name. Imported lazily so lexical runs never load onnxruntime.
_EMBEDDERS: dict[str, Any] = {}
# NeMo scores cases on a thread pool; one lock guards the lazy load and each embed call.
_EMBED_LOCK = threading.Lock()


def _embed(texts: list[str], model_name: str) -> list[np.ndarray]:
    with _EMBED_LOCK:
        model = _EMBEDDERS.get(model_name)
        if model is None:
            from fastembed import TextEmbedding  # noqa: PLC0415

            model = _EMBEDDERS[model_name] = TextEmbedding(model_name)
        return list(model.embed(texts))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return max(0.0, float(np.dot(a, b)) / denom) if denom else 0.0


def _similarities(
    metric: str, response: str, references: list[str], model_name: str
) -> list[float]:
    """Similarity of ``response`` to each of ``references`` under ``metric``."""
    if metric == EMBEDDING_METRIC:
        if not response.strip():
            return [0.0] * len(references)
        response_vec, *reference_vecs = _embed([response, *references], model_name)
        return [_cosine(response_vec, vec) for vec in reference_vecs]
    tokens = _tokenize(response)
    return [_LEXICAL_METRICS[metric](tokens, _tokenize(ref)) for ref in references]


def _best_match(scores: list[float], references: list[str]) -> tuple[float, str]:
    """Return (best similarity, reference it came from) across ``references``."""
    return max(zip(scores, references), key=lambda pair: pair[0])


def _truncate(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def evaluate_response(
    response: str, metadata: dict[str, Any], params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Pure scoring logic, importable for unit tests, no judge, no I/O."""
    acceptable = _as_list(metadata.get("acceptable_responses"))
    if not acceptable:
        return {"reason": "no acceptable_responses provided: cannot score"}

    params = PARAMS if params is None else params
    metric = str(metadata.get("metric") or params.get("metric") or DEFAULT_METRIC).lower()
    if metric not in METRICS:
        return {
            "reason": f"unknown metric '{metric}' (expected one of: "
            f"{', '.join(sorted(METRICS))}): cannot score"
        }

    unacceptable = _as_list(metadata.get("unacceptable_responses"))
    model_name = str(params.get("embedding_model") or DEFAULT_EMBEDDING_MODEL)
    try:
        # One call for both lists, so embedding_cosine embeds the response once.
        scores = _similarities(metric, response or "", acceptable + unacceptable, model_name)
    except Exception as e:  # noqa: BLE001
        # Only the embedding path can raise (model download or load failure).
        return {"reason": f"{metric} could not run ({type(e).__name__}: {e}): cannot score"}

    best, best_ref = _best_match(scores[: len(acceptable)], acceptable)
    explanation = _EXPLANATIONS[metric]
    if metric == EMBEDDING_METRIC:
        explanation += f"; model {model_name}"
    reason = (
        f"{metric} {best:.2f} ({explanation}); "
        f"closest of {len(acceptable)} acceptable: '{_truncate(best_ref)}'"
    )

    if unacceptable:
        worst, worst_ref = _best_match(scores[len(acceptable) :], unacceptable)
        reason += f"; closest unacceptable {worst:.2f}: '{_truncate(worst_ref)}'"
        if worst >= best:
            reason += " (WARNING: at least as close as any acceptable example)"

    value = round(best, 4)
    return {"score": value, metric: value, "reason": reason}


@benchmark(  # type: ignore[untyped-decorator]
    name="response_similarity",
    dataset="cases.jsonl",
    prompt="{input}",
    endpoint_type="chat",
)
@scorer  # type: ignore[untyped-decorator]
def score(sample: ScorerInput) -> dict[str, Any]:
    return evaluate_response(sample.response, sample.metadata)
