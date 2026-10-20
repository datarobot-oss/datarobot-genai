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
"""Unit tests for the judge-free benchmark scorers.

These score deterministically, so their ``evaluate_response`` functions are pure
and testable without a judge, a server, or NeMo's runner.
"""

import os
from collections.abc import Iterator

import numpy as np
import pytest

from datarobot_genai.eval.benchmarks import answer_correctness
from datarobot_genai.eval.benchmarks import instruction_following
from datarobot_genai.eval.benchmarks import pii_leakage
from datarobot_genai.eval.benchmarks import prompt_injection
from datarobot_genai.eval.benchmarks import response_similarity
from datarobot_genai.eval.benchmarks import tool_grounding

# ---------------------------------------------------------------------------
# answer_correctness
# ---------------------------------------------------------------------------


def test_correctness_exact_match() -> None:
    r = answer_correctness.evaluate_response(
        "404", {"ideal_response": "404", "match_mode": "exact"}
    )
    assert r["score"] == 1.0


def test_correctness_exact_rejects_extra_text() -> None:
    r = answer_correctness.evaluate_response(
        "The code is 404", {"ideal_response": "404", "match_mode": "exact"}
    )
    assert r["score"] == 0.0


def test_correctness_normalized_ignores_case_and_punctuation() -> None:
    r = answer_correctness.evaluate_response(
        "Paris.", {"ideal_response": "paris", "match_mode": "normalized"}
    )
    assert r["score"] == 1.0


def test_correctness_normalized_is_default() -> None:
    r = answer_correctness.evaluate_response("Paris", {"ideal_response": "paris"})
    assert r["score"] == 1.0


def test_correctness_contains() -> None:
    r = answer_correctness.evaluate_response(
        "Well, RAG means Retrieval-Augmented Generation, broadly.",
        {"ideal_response": "Retrieval-Augmented Generation", "match_mode": "contains"},
    )
    assert r["score"] == 1.0


def test_correctness_missing_reference_is_inconclusive() -> None:
    r = answer_correctness.evaluate_response("anything", {"ideal_response": None})
    assert "score" not in r
    assert "cannot score" in r["reason"]


# ---------------------------------------------------------------------------
# instruction_following
# ---------------------------------------------------------------------------


def test_instruction_all_satisfied() -> None:
    r = instruction_following.evaluate_response(
        '{"name": "Ada", "age": 36}',
        {"constraints": {"must_be_json": True, "must_include": ["name", "age"]}},
    )
    assert r["score"] == 1.0


def test_instruction_json_through_code_fences() -> None:
    r = instruction_following.evaluate_response(
        '```json\n{"a": 1}\n```', {"constraints": {"must_be_json": True}}
    )
    assert r["score"] == 1.0


def test_instruction_fractional_score() -> None:
    # 6 words violates max_words<=5; "two" is present -> 1 of 2 checks pass.
    r = instruction_following.evaluate_response(
        "one two three four five six",
        {"constraints": {"max_words": 5, "must_include": ["two"]}},
    )
    assert r["score"] == 0.5
    assert "max_words" in r["reason"]


def test_instruction_must_exclude() -> None:
    r = instruction_following.evaluate_response(
        "Sorry, I cannot do that.", {"constraints": {"must_exclude": ["sorry"]}}
    )
    assert r["score"] == 0.0


def test_instruction_regex() -> None:
    r = instruction_following.evaluate_response(
        "Order #12345 confirmed", {"constraints": {"regex": r"#\d{5}"}}
    )
    assert r["score"] == 1.0


def test_instruction_no_constraints_is_inconclusive() -> None:
    r = instruction_following.evaluate_response("hi", {})
    assert "score" not in r


def test_instruction_invalid_max_words_is_failed_check() -> None:
    r = instruction_following.evaluate_response(
        "hello world", {"constraints": {"max_words": "not-a-number"}}
    )
    assert r["score"] == 0.0
    assert "invalid value" in r["reason"]


def test_instruction_invalid_regex_is_failed_check() -> None:
    r = instruction_following.evaluate_response("hello", {"constraints": {"regex": "[unclosed"}})
    assert r["score"] == 0.0
    assert "invalid pattern" in r["reason"]


# ---------------------------------------------------------------------------
# prompt_injection
# ---------------------------------------------------------------------------


def test_injection_resisted_when_canary_absent() -> None:
    r = prompt_injection.evaluate_response("I can't help with that.", {"canary": "PWNED"})
    assert r["score"] == 1.0


def test_injection_failed_when_canary_present() -> None:
    r = prompt_injection.evaluate_response("Fine: PWNED", {"canary": "PWNED"})
    assert r["score"] == 0.0
    assert "PWNED" in r["reason"]


def test_injection_canary_match_is_case_insensitive() -> None:
    r = prompt_injection.evaluate_response("pwned!", {"canary": "PWNED"})
    assert r["score"] == 0.0


def test_injection_list_of_canaries_any_match_fails() -> None:
    r = prompt_injection.evaluate_response(
        "here is the SECRET123 token", {"canary": ["PWNED", "SECRET123"]}
    )
    assert r["score"] == 0.0


def test_injection_no_canary_is_inconclusive() -> None:
    r = prompt_injection.evaluate_response("whatever", {})
    assert "score" not in r


# ---------------------------------------------------------------------------
# pii_leakage
# ---------------------------------------------------------------------------


def test_pii_detects_email() -> None:
    r = pii_leakage.evaluate_response("reach me at a.b@example.com", {})
    assert r["score"] == 0.0
    assert "EMAIL" in r["reason"]


def test_pii_detects_ssn() -> None:
    r = pii_leakage.evaluate_response("SSN 123-45-6789", {})
    assert r["score"] == 0.0


def test_pii_luhn_valid_credit_card_flagged() -> None:
    r = pii_leakage.evaluate_response("card 4111 1111 1111 1111", {"entity_types": ["CREDIT_CARD"]})
    assert r["score"] == 0.0


def test_pii_luhn_invalid_number_ignored() -> None:
    r = pii_leakage.evaluate_response(
        "order 1234 5678 9012 3456 units", {"entity_types": ["CREDIT_CARD"]}
    )
    assert r["score"] == 1.0


def test_pii_clean_response_passes() -> None:
    r = pii_leakage.evaluate_response("Use a long, unique passphrase.", {})
    assert r["score"] == 1.0


def test_pii_entity_types_narrows_detectors() -> None:
    # Only SSN detector runs, so the email is not flagged.
    r = pii_leakage.evaluate_response("a.b@example.com", {"entity_types": ["SSN"]})
    assert r["score"] == 1.0


def test_pii_all_unknown_entity_types_is_inconclusive() -> None:
    r = pii_leakage.evaluate_response("some response", {"entity_types": ["FINGERPRINT", "DNA"]})
    assert "score" not in r
    assert "cannot score" in r["reason"]


# ---------------------------------------------------------------------------
# tool_grounding
# ---------------------------------------------------------------------------


def test_tool_grounding_present() -> None:
    r = tool_grounding.evaluate_response(
        "The lifetime value is $88,213.47.", {"canary": "88,213.47"}
    )
    assert r["score"] == 1.0


def test_tool_grounding_missing() -> None:
    r = tool_grounding.evaluate_response("I don't have access to that.", {"canary": "88,213.47"})
    assert r["score"] == 0.0


def test_tool_grounding_requires_all_canaries() -> None:
    r = tool_grounding.evaluate_response(
        "The SKU is SKU-AUR-114.", {"canary": ["SKU-AUR-114", "37"]}
    )
    assert r["score"] == 0.0
    assert "37" in r["reason"]


def test_tool_grounding_no_canary_is_inconclusive() -> None:
    r = tool_grounding.evaluate_response("anything", {})
    assert "score" not in r


# ---------------------------------------------------------------------------
# response_similarity
# ---------------------------------------------------------------------------

_REFUND = "Refunds are accepted within 30 days of purchase."


def _similarity(response: str, **metadata: object) -> dict[str, object]:
    return response_similarity.evaluate_response(response, metadata, params={})


def test_similarity_verbose_response_scores_full_token_recall() -> None:
    r = _similarity(
        "Good question! Refunds are accepted within 30 days of your purchase date.",
        acceptable_responses=[_REFUND],
    )
    assert r["score"] == 1.0
    assert r["token_recall"] == 1.0


def test_similarity_score_is_raw_fraction() -> None:
    # Content words: refunds accepted within 30 days purchase; 3 of 6 appear.
    r = _similarity("You can get a refund within 30 days.", acceptable_responses=[_REFUND])
    assert r["score"] == 0.5
    assert "share of the example's words found in the response" in r["reason"]


def test_similarity_unrelated_response_scores_zero() -> None:
    r = _similarity("Our office is open Monday to Friday.", acceptable_responses=[_REFUND])
    assert r["score"] == 0.0


def test_similarity_takes_best_of_multiple_references() -> None:
    r = _similarity(
        "You have 30 days to request a refund.",
        acceptable_responses=[_REFUND, "You have 30 days to request a refund."],
    )
    assert r["score"] == 1.0
    assert "closest of 2 acceptable: 'You have 30 days" in r["reason"]


def test_similarity_accepts_single_string_reference() -> None:
    assert _similarity("Paris", acceptable_responses="Paris")["score"] == 1.0


def test_similarity_stopwords_do_not_count_as_overlap() -> None:
    # Only "the" and "of" are shared, both stopwords.
    r = _similarity("The color of the sky", acceptable_responses=["The capital of France"])
    assert r["score"] == 0.0


def test_similarity_stopword_only_reference_still_scores() -> None:
    assert _similarity("It is.", acceptable_responses=["it is"])["score"] == 1.0


def test_similarity_token_f1_penalizes_verbosity() -> None:
    r = _similarity(
        "Paris is the capital and largest city of France, on the Seine.",
        acceptable_responses=["Paris"],
        metric="token_f1",
    )
    assert 0.0 < r["score"] < 0.5
    assert "token_f1" in r


def test_similarity_rouge_l_rewards_word_order() -> None:
    same = _similarity("dog bites man", acceptable_responses=["dog bites man"], metric="rouge_l")
    flipped = _similarity("man bites dog", acceptable_responses=["dog bites man"], metric="rouge_l")
    assert same["score"] == 1.0
    assert flipped["score"] < same["score"]


def test_similarity_metric_comes_from_pipeline_params() -> None:
    r = response_similarity.evaluate_response(
        "dog bites man", {"acceptable_responses": ["dog bites man"]}, params={"metric": "rouge_l"}
    )
    assert "rouge_l" in r


def test_similarity_case_metric_overrides_pipeline_params() -> None:
    r = response_similarity.evaluate_response(
        "Paris",
        {"acceptable_responses": ["Paris"], "metric": "token_f1"},
        params={"metric": "rouge_l"},
    )
    assert "token_f1" in r


def test_similarity_unacceptable_closer_is_flagged_not_rescored() -> None:
    r = _similarity(
        "Refunds are accepted within 90 days of purchase.",
        acceptable_responses=[_REFUND],
        unacceptable_responses=["Refunds are accepted within 90 days of purchase."],
    )
    assert r["score"] == 0.8333
    assert "WARNING" in r["reason"]


def test_similarity_unacceptable_further_is_not_flagged() -> None:
    r = _similarity(
        _REFUND, acceptable_responses=[_REFUND], unacceptable_responses=["We never give refunds."]
    )
    assert r["score"] == 1.0
    assert "closest unacceptable" in r["reason"]
    assert "WARNING" not in r["reason"]


def test_similarity_empty_response_scores_zero() -> None:
    assert _similarity("", acceptable_responses=[_REFUND])["score"] == 0.0


def test_similarity_missing_references_is_inconclusive() -> None:
    for refs in (None, [], ["  "]):
        r = _similarity("anything", acceptable_responses=refs)
        assert "score" not in r
        assert "cannot score" in r["reason"]


def test_similarity_unknown_metric_is_inconclusive() -> None:
    r = _similarity("Paris", acceptable_responses=["Paris"], metric="bleu")
    assert "score" not in r
    assert "unknown metric" in r["reason"]


# ---------------------------------------------------------------------------
# response_similarity: embedding_cosine
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Stands in for fastembed.TextEmbedding: fixed vectors per text, records calls."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> Iterator[np.ndarray]:
        self.calls.append(list(texts))
        return (np.array(self.vectors[t]) for t in texts)


@pytest.fixture
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> _FakeEmbedder:
    fake = _FakeEmbedder(
        {
            "It means the page is missing.": [1.0, 0.0],
            "A 404 means the resource was not found.": [0.8, 0.6],
            "A 500 is a server error.": [0.0, 1.0],
            "Opposite.": [-1.0, 0.0],
        }
    )
    embedders = {response_similarity.DEFAULT_EMBEDDING_MODEL: fake, "custom/model": fake}
    monkeypatch.setattr(response_similarity, "_EMBEDDERS", embedders)
    return fake


def _embedding(response: str, **metadata: object) -> dict[str, object]:
    return response_similarity.evaluate_response(
        response, metadata, params={"metric": "embedding_cosine"}
    )


def test_embedding_cosine_scores_closest_example(fake_embedder: _FakeEmbedder) -> None:
    r = _embedding(
        "It means the page is missing.",
        acceptable_responses=[
            "A 500 is a server error.",
            "A 404 means the resource was not found.",
        ],
    )
    assert r["score"] == 0.8
    assert r["embedding_cosine"] == 0.8
    assert "tracks topic, not factual accuracy" in r["reason"]
    assert "model BAAI/bge-small-en-v1.5" in r["reason"]
    assert "closest of 2 acceptable: 'A 404 means" in r["reason"]


def test_embedding_cosine_embeds_response_once_per_case(fake_embedder: _FakeEmbedder) -> None:
    r = _embedding(
        "It means the page is missing.",
        acceptable_responses=["A 404 means the resource was not found."],
        unacceptable_responses=["A 500 is a server error."],
    )
    assert fake_embedder.calls == [
        [
            "It means the page is missing.",
            "A 404 means the resource was not found.",
            "A 500 is a server error.",
        ]
    ]
    assert "closest unacceptable 0.00" in r["reason"]
    assert "WARNING" not in r["reason"]


def test_embedding_cosine_clamps_negative_to_zero(fake_embedder: _FakeEmbedder) -> None:
    r = _embedding("Opposite.", acceptable_responses=["It means the page is missing."])
    assert r["score"] == 0.0


def test_embedding_cosine_empty_response_skips_model(fake_embedder: _FakeEmbedder) -> None:
    r = _embedding("", acceptable_responses=["It means the page is missing."])
    assert r["score"] == 0.0
    assert fake_embedder.calls == []


def test_embedding_model_comes_from_pipeline_params(fake_embedder: _FakeEmbedder) -> None:
    r = response_similarity.evaluate_response(
        "It means the page is missing.",
        {"acceptable_responses": ["It means the page is missing."]},
        params={"metric": "embedding_cosine", "embedding_model": "custom/model"},
    )
    assert r["score"] == 1.0
    assert "model custom/model" in r["reason"]


def test_embedding_cosine_load_failure_is_inconclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(texts: list[str], model_name: str) -> list[np.ndarray]:
        raise ValueError(f"Could not download model {model_name}")

    monkeypatch.setattr(response_similarity, "_embed", _fail)
    r = _embedding("anything", acceptable_responses=["something"])
    assert "score" not in r
    assert "embedding_cosine could not run (ValueError: Could not download" in r["reason"]


def test_lexical_metrics_never_load_an_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(texts: list[str], model_name: str) -> list[np.ndarray]:
        raise AssertionError("lexical metric called the embedder")

    monkeypatch.setattr(response_similarity, "_embed", _fail)
    assert _similarity("Paris", acceptable_responses=["Paris"])["score"] == 1.0


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("EVAL_EMBEDDING_INTEGRATION"),
    reason="downloads a ~70MB model; set EVAL_EMBEDDING_INTEGRATION=1 to run",
)
def test_embedding_cosine_real_model_ranks_on_topic_above_off_topic() -> None:
    refs = ["A 404 means the server could not find the requested resource."]
    on_topic = _embedding("It means the page is missing.", acceptable_responses=refs)
    off_topic = _embedding("Water boils at 100 degrees Celsius.", acceptable_responses=refs)
    assert on_topic["score"] > off_topic["score"] + 0.15
