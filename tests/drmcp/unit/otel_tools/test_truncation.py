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

"""Unit tests for the OTel truncation module and its constants.

Style: GIVEN preconditions / WHEN behavior under test / THEN expected outcomes.

No tools exist yet — these exercise the helpers in isolation against the two
generated trace populations, which is the point of doing the hard part first.
"""

import copy
import json
from typing import Any

from datarobot_genai.drmcputils.constants import LOG_LEVELS
from datarobot_genai.drtools.otel.constants import OTEL_ENTITY_TYPES
from datarobot_genai.drtools.otel.constants import OTEL_LOG_LEVELS
from datarobot_genai.drtools.otel.constants import EntityType
from datarobot_genai.drtools.otel.truncation import PAYLOAD_FIELDS_MAX
from datarobot_genai.drtools.otel.truncation import PAYLOAD_MIN_CHARS
from datarobot_genai.drtools.otel.truncation import STATUS_MESSAGE_MAX_CHARS
from datarobot_genai.drtools.otel.truncation import apply_char_budget
from datarobot_genai.drtools.otel.truncation import canonical_attributes
from datarobot_genai.drtools.otel.truncation import cap_payload_value
from datarobot_genai.drtools.otel.truncation import cap_value
from datarobot_genai.drtools.otel.truncation import summarize_spans
from tests.drmcp.unit.otel_tools.trace_factory import CHARS_PER_TOKEN
from tests.drmcp.unit.otel_tools.trace_factory import OVERSIZED_CANONICAL_PAYLOAD_CHARS
from tests.drmcp.unit.otel_tools.trace_factory import OVERSIZED_GIANT_ATTRIBUTE_CHARS
from tests.drmcp.unit.otel_tools.trace_factory import OVERSIZED_SPAN_COUNT
from tests.drmcp.unit.otel_tools.trace_factory import OVERSIZED_TOTAL_PAYLOAD_CHARS
from tests.drmcp.unit.otel_tools.trace_factory import SMALL_TRACE_MAX_TOKENS

# Pinned defaults from the plan. 60,000 chars is ~20k tokens at the measured 3.1
# chars per token. Both per-field caps govern payload projections only — §2.2's
# view="summary" has no field to cap, and §2.2 marks max_field_chars/max_total_chars
# 'view="payloads" only'.
MAX_TOTAL_CHARS_DEFAULT = 60_000  # otel_trace_get(max_total_chars), §2.2
MAX_FIELD_CHARS_TRACE_VIEW = 2_000  # otel_trace_get(max_field_chars), §2.2
MAX_FIELD_CHARS_SPAN_VIEW = 8_000  # otel_span_payload_get(max_field_chars), §2.3

# Measured cost of the summary projection: 4,936 tokens over 37 spans on the
# 1.56M-token trace, i.e. ~133 tokens a span, flat and independent of payload
# size. The summary projection must stay under that rate.
SUMMARY_TOKENS_PER_SPAN_CEILING = 133


# ------------------------------------------------------------------ #
# constants                                                          #
# ------------------------------------------------------------------ #


def test_otel_log_levels_accept_warning_and_critical_unlike_the_workload_tuple() -> None:
    # GIVEN the /otel logs endpoint accepts six minimum levels
    # WHEN OTEL_LOG_LEVELS is compared with drmcputils' four-level workload tuple
    # THEN it is a strict superset, and 'warning'/'critical' are the difference
    assert OTEL_LOG_LEVELS == ("debug", "info", "warn", "warning", "error", "critical")
    assert set(OTEL_LOG_LEVELS) > set(LOG_LEVELS)
    assert set(OTEL_LOG_LEVELS) - set(LOG_LEVELS) == {"warning", "critical"}
    # Reusing LOG_LEVELS here would silently reject two levels the API accepts.
    assert "warning" not in LOG_LEVELS
    assert "critical" not in LOG_LEVELS


def test_otel_entity_types_stay_snake_case_on_the_wire() -> None:
    # GIVEN entityType is a ChoiceField with initial_camelization=False
    # WHEN the tuple is inspected
    # THEN every value is snake_case, and it is exactly the EntityType Literal
    assert OTEL_ENTITY_TYPES == (
        "deployment",
        "use_case",
        "experiment_container",
        "custom_application",
        "workload",
        "execution_environment",
        "custom_job",
        "artifact",
    )
    assert set(OTEL_ENTITY_TYPES) == set(EntityType.__args__)
    assert not any(character.isupper() for value in OTEL_ENTITY_TYPES for character in value)


# ------------------------------------------------------------------ #
# summarize_spans                                                    #
# ------------------------------------------------------------------ #


def test_summarize_spans_projects_structure_and_drops_every_payload(
    oversized_agent_trace: dict[str, Any],
) -> None:
    # GIVEN the oversized trace, whose spans carry 1,022,000 chars of payload
    # WHEN the spans are summarized
    summaries, _ = summarize_spans(oversized_agent_trace["spans"])

    # THEN every span keeps its structural fields and no payload survives
    assert len(summaries) == OVERSIZED_SPAN_COUNT
    for summary in summaries:
        assert set(summary) >= {
            "span_id",
            "parent_span_id",
            "name",
            "status_code",
            "status_message",
            "kind",
            "service_name",
            "duration",
            "start_time",
            "payload_chars",
            "payload_fields",
        }
        assert "prompt" not in summary
        assert "completion" not in summary
        assert "attributes" not in summary
    # THEN the ERROR span's status message is preserved — it is why you drilled in
    error_summaries = [s for s in summaries if s["status_code"] == "ERROR"]
    assert len(error_summaries) == 1
    assert "RateLimitError" in error_summaries[0]["status_message"]


def test_summarize_spans_payload_chars_sum_to_the_fixtures_real_total(
    oversized_agent_trace: dict[str, Any],
) -> None:
    # GIVEN a fixture whose payload plan declares 1,022,000 chars
    # WHEN the spans are summarized
    summaries, stats = summarize_spans(oversized_agent_trace["spans"])

    # THEN the accounting is correct, not merely present
    assert stats["total_payload_chars"] == OVERSIZED_TOTAL_PAYLOAD_CHARS
    assert sum(s["payload_chars"] for s in summaries) == OVERSIZED_TOTAL_PAYLOAD_CHARS
    # THEN the fixture really is the oversized population (>= 200k tokens)
    assert OVERSIZED_TOTAL_PAYLOAD_CHARS / CHARS_PER_TOKEN >= 200_000


def test_summarize_spans_payload_fields_point_at_where_the_mass_is(
    oversized_agent_trace: dict[str, Any],
) -> None:
    # GIVEN one tool span whose 160,000-char output is written by four
    # instrumentations, and a second span with no payload at all
    summaries, _ = summarize_spans(oversized_agent_trace["spans"])
    giant = max(summaries, key=lambda summary: summary["payload_chars"])
    payload_free = [s for s in summaries if s["name"] == "guardrail.check"]

    # WHEN / THEN the heaviest span names all four carriers, largest first
    assert giant["name"] == "tool.execute"
    assert giant["payload_chars"] == 4 * OVERSIZED_GIANT_ATTRIBUTE_CHARS
    assert set(giant["payload_fields"]) == {
        "completion",
        "gen_ai.task.output",
        "input.value",
        "traceloop.entity.output",
    }
    # THEN small metadata attributes never masquerade as payload
    assert "gen_ai.tool.name" not in giant["payload_fields"]
    # THEN a payload-free span reports zero rather than omitting the accounting
    assert payload_free[0]["payload_chars"] == 0
    assert payload_free[0]["payload_fields"] == []


def test_summary_projection_of_the_oversized_trace_stays_under_the_measured_ceiling(
    oversized_agent_trace: dict[str, Any],
) -> None:
    # GIVEN the oversized trace: 1,022,000 payload chars, ~330k tokens
    # WHEN the summary projection is serialized as it would be returned
    summaries, stats = summarize_spans(oversized_agent_trace["spans"])
    serialized_chars = len(json.dumps(summaries))
    ceiling = OVERSIZED_SPAN_COUNT * SUMMARY_TOKENS_PER_SPAN_CEILING * CHARS_PER_TOKEN

    # THEN it fits the measured ~133 tokens-a-span rate and is a rounding error
    # against the payload it describes. This is the regression guard for the whole
    # summary-by-default design.
    assert serialized_chars < ceiling, (
        f"summary projection grew to {serialized_chars} chars, over the "
        f"{ceiling:.0f}-char ceiling implied by the measured 133 tokens a span"
    )
    assert serialized_chars < stats["total_payload_chars"] / 100


def test_summary_projection_size_does_not_grow_with_payload_size(
    oversized_agent_trace: dict[str, Any],
) -> None:
    # GIVEN the same trace with every payload string inflated tenfold
    baseline_summaries, baseline_stats = summarize_spans(oversized_agent_trace["spans"])
    inflated_summaries, inflated_stats = summarize_spans(
        _inflate_payloads(copy.deepcopy(oversized_agent_trace)["spans"], 10)
    )

    # WHEN the two projections are compared
    baseline_chars = len(json.dumps(baseline_summaries))
    inflated_chars = len(json.dumps(inflated_summaries))

    # THEN the payload grew by ~9.2M chars and the summary by less than 64 — the
    # digits of payload_chars. Span-count-bounded, not payload-bounded.
    assert inflated_stats["total_payload_chars"] == baseline_stats["total_payload_chars"] * 10
    assert inflated_chars - baseline_chars < 64
    assert [s["payload_fields"] for s in inflated_summaries] == [
        s["payload_fields"] for s in baseline_summaries
    ]


def test_summarize_spans_caps_payload_fields_and_reports_the_omission() -> None:
    # GIVEN a span carrying more large attributes than the payload_fields cap
    span = {
        "span_id": "aaaa0000bbbb1111",
        "name": "llm.chat",
        "attributes": {f"vendor.blob.{index:02d}": "x" * (1_000 + index) for index in range(20)},
    }

    # WHEN the span is summarized
    summaries, stats = summarize_spans([span])

    # THEN the list is capped, ordered largest first, and the omission is reported
    assert len(summaries[0]["payload_fields"]) == PAYLOAD_FIELDS_MAX
    assert summaries[0]["payload_fields"][0] == "vendor.blob.19"
    assert summaries[0]["payload_fields_omitted"] == 20 - PAYLOAD_FIELDS_MAX
    # THEN payload_chars still counts every field, capped list or not
    assert summaries[0]["payload_chars"] == sum(1_000 + index for index in range(20))
    assert stats["total_payload_chars"] == summaries[0]["payload_chars"]


def test_summarize_spans_counts_a_response_field_and_a_same_named_attribute() -> None:
    # GIVEN a span whose attributes carry their own key named 'prompt' beside the
    # response field of the same name. SpanViewValidator.attributes is a dynamic
    # dict, so the collision is reachable on the wire.
    span = {
        "span_id": "aaaa0000bbbb1111",
        "name": "llm.chat",
        "prompt": "A" * 300,
        "attributes": {"prompt": "B" * 500},
    }

    # WHEN the span is summarized
    summaries, stats = summarize_spans([span])

    # THEN both fields are counted — 800 chars, not the 500 an overwrite reports.
    # payload_chars is what an agent uses to pick a span, so it must not undercount.
    assert summaries[0]["payload_chars"] == 800
    assert stats["total_payload_chars"] == 800
    assert summaries[0]["payload_fields"] == ["prompt"]


def test_summarize_spans_counts_container_payloads_by_their_serialized_size() -> None:
    # GIVEN a span whose only payload is a JSON-decoded message list — the wire
    # shape OTLP array attributes and instrumentations actually produce
    messages = [{"role": "user", "content": "x" * 100_000}]
    span = {
        "span_id": "aaaa0000bbbb1111",
        "name": "llm.chat",
        "attributes": {"gen_ai.prompt": messages, "gen_ai.usage.input_tokens": 38_211},
    }

    # WHEN the span is summarized
    summaries, stats = summarize_spans([span])

    # THEN the list is counted at its serialized size, not reported as 0 payload
    # chars — 'where the mass is' must include non-str payload carriers
    expected = len(json.dumps(messages, default=str, ensure_ascii=False))
    assert summaries[0]["payload_chars"] == expected
    assert summaries[0]["payload_fields"] == ["gen_ai.prompt"]
    assert stats["total_payload_chars"] == expected
    # THEN scalar metadata still never masquerades as payload
    assert "gen_ai.usage.input_tokens" not in summaries[0]["payload_fields"]


def test_summarize_spans_caps_a_traceback_in_status_message(
    oversized_agent_trace: dict[str, Any],
) -> None:
    """status_message is the one structural field that can blow the summary view.

    ``SpanViewValidator.status_message`` is a StringField with no server-side
    max_length, and §2.2 puts ``max_total_chars`` on view="payloads" only — so
    nothing else stands behind the summary projection. Uncapped, one 48,000-char
    traceback grew this fixture's projection from 4,094 to 64,077 chars, 13x the
    §7 ceiling and past the whole 60,000-char budget.
    """
    # GIVEN every span carrying a 48,000-char traceback
    baseline = len(json.dumps(summarize_spans(oversized_agent_trace["spans"])[0]))
    traceback = 'Traceback (most recent call last):\n  File "agent.py", line 41\n' * 800
    assert len(traceback) > 48_000
    spans = copy.deepcopy(oversized_agent_trace["spans"])
    for span in spans:
        span["status_message"] = traceback

    # WHEN the spans are summarized
    summaries, _ = summarize_spans(spans)

    # THEN each message is windowed and the loss is marked, never silent
    remainder = len(traceback) - STATUS_MESSAGE_MAX_CHARS
    for summary in summaries:
        assert summary["status_message"].startswith(traceback[:STATUS_MESSAGE_MAX_CHARS])
        assert summary["status_message"].endswith(f"…[truncated, {remainder} more chars]")

    # THEN the projection grew by the cap times the span count, not by the input:
    # O(span_count), which is the property §2.2's flat ~133 tok/span rate rests on
    inflated = len(json.dumps(summaries))
    assert inflated - baseline < OVERSIZED_SPAN_COUNT * (STATUS_MESSAGE_MAX_CHARS + 64), (
        f"summary grew {inflated - baseline} chars on {len(traceback) * OVERSIZED_SPAN_COUNT} "
        "chars of status_message — the summary view must stay bounded by span count"
    )


def test_summarize_spans_does_not_mutate_the_spans_it_reads(
    oversized_agent_trace: dict[str, Any],
) -> None:
    # GIVEN the oversized trace and a snapshot of it
    before = copy.deepcopy(oversized_agent_trace["spans"])

    # WHEN the spans are summarized
    summarize_spans(oversized_agent_trace["spans"])

    # THEN the caller's spans are untouched — the projection is a copy
    assert oversized_agent_trace["spans"] == before


# ------------------------------------------------------------------ #
# canonical_attributes                                               #
# ------------------------------------------------------------------ #


def test_canonical_attributes_reports_every_dropped_field(
    oversized_agent_trace: dict[str, Any],
) -> None:
    # GIVEN every span of the oversized trace, reduced both the way a tool will
    # call it (rank-1 response fields merged in) and the way its §3 signature
    # alone allows (bare attributes, no merge)
    for span in oversized_agent_trace["spans"]:
        for payload in (_merged_payload(span), dict(span["attributes"])):
            # WHEN its payload fields are reduced to the canonical set
            kept, dropped = canonical_attributes(payload)

            # THEN nothing vanishes silently: every input name is either kept or
            # reported in exactly one drop bucket
            reported = set(kept) | set(dropped["duplicate"]) | set(dropped["semconv"])
            assert reported == set(payload)
            assert not set(dropped["duplicate"]) & set(dropped["semconv"])
            assert not set(dropped["duplicate"]) & set(kept)
            assert not set(dropped["semconv"]) & set(kept)

            # THEN every name in 'duplicate' really is a duplicate: §2.3 surfaces
            # that bucket as "you already have this text", so a name may only land
            # there when a byte-identical copy survives in kept
            surviving = {value for value in kept.values() if isinstance(value, str)}
            for name in dropped["duplicate"]:
                assert payload[name] in surviving, (
                    f"{name} was reported as a duplicate but no copy of its text "
                    "survives in the response"
                )


def test_canonical_attributes_drops_the_traceloop_gen_ai_twins_as_duplicates(
    oversized_agent_trace: dict[str, Any],
) -> None:
    # GIVEN the llm.chat span whose prompt and completion are each written by
    # four instrumentations byte for byte
    span = oversized_agent_trace["spans"][2]
    payload = _merged_payload(span)
    assert payload["gen_ai.task.input"] == payload["traceloop.entity.input"] == payload["prompt"]

    # WHEN the payload is reduced
    kept, dropped = canonical_attributes(payload)

    # THEN the twins are dropped as duplicates and named, so an agent looking for
    # traceloop.entity.output by name is told where it went
    assert dropped["duplicate"] == [
        "gen_ai.completion.0.content",
        "gen_ai.task.input",
        "gen_ai.task.output",
        "input.value",
        "traceloop.entity.input",
        "traceloop.entity.output",
    ]
    # THEN the text itself survives on the rank-1 response fields
    assert kept["prompt"] == payload["prompt"]
    assert kept["completion"] == payload["completion"]


def test_canonical_attributes_separates_derived_families_from_duplicates(
    oversized_agent_trace: dict[str, Any],
) -> None:
    # GIVEN a span carrying derived families whose text appears nowhere else
    span = oversized_agent_trace["spans"][2]

    # WHEN the payload is reduced
    _, dropped = canonical_attributes(_merged_payload(span))

    # THEN they land in the semconv bucket, not the duplicate one — the caller
    # reports which and why
    assert dropped["semconv"] == ["input.value_obj", "nat.metadata"]


def test_canonical_attributes_never_drops_the_rank_one_response_fields() -> None:
    # GIVEN a span where prompt and completion happen to hold the same text
    echoed = "e" * (PAYLOAD_MIN_CHARS * 3)
    payload = {"prompt": echoed, "completion": echoed, "traceloop.entity.output": echoed}

    # WHEN the payload is reduced
    kept, dropped = canonical_attributes(payload)

    # THEN both response fields survive and only the attribute twin is dropped
    assert kept == {"prompt": echoed, "completion": echoed}
    assert dropped == {"duplicate": ["traceloop.entity.output"], "semconv": []}


def test_canonical_attributes_keeps_the_highest_precedence_carrier() -> None:
    # GIVEN the same text under gen_ai, openinference, traceloop and an unknown
    # family, with no response field to win the tie
    shared = "s" * (PAYLOAD_MIN_CHARS * 5)
    payload = {
        "vendor.custom.output": shared,
        "traceloop.llm.response": shared,
        "output.value": shared,
        "gen_ai.output.messages": shared,
    }

    # WHEN the payload is reduced
    kept, dropped = canonical_attributes(payload)

    # THEN gen_ai wins per the §3 precedence order, whatever the input order was
    assert list(kept) == ["gen_ai.output.messages"]
    assert dropped["duplicate"] == [
        "output.value",
        "traceloop.llm.response",
        "vendor.custom.output",
    ]


def test_canonical_attributes_ranks_openinference_over_traceloop_over_nat() -> None:
    # GIVEN the same text under §3's ranks 3, 4 and 5, with no higher family
    # present to decide the tie for them
    shared = "p" * (PAYLOAD_MIN_CHARS * 4)

    # WHEN all three compete
    kept, dropped = canonical_attributes(
        {"nat.output": shared, "traceloop.llm.response": shared, "output.value": shared}
    )

    # THEN openinference (rank 3) wins, whatever order the attributes arrived in
    assert list(kept) == ["output.value"]
    assert dropped["duplicate"] == ["nat.output", "traceloop.llm.response"]

    # WHEN openinference is absent, THEN traceloop (rank 4) still outranks nat (5)
    kept, dropped = canonical_attributes({"nat.output": shared, "traceloop.llm.response": shared})
    assert list(kept) == ["traceloop.llm.response"]
    assert dropped["duplicate"] == ["nat.output"]

    # WHEN a family no table names competes, THEN it loses to every known one
    kept, _ = canonical_attributes({"vendor.blob.output": shared, "nat.output": shared})
    assert list(kept) == ["nat.output"]


def test_canonical_attributes_reports_a_wholly_dropped_group_as_semconv_not_duplicate() -> None:
    """A drop is a 'duplicate' only if a copy of its text really survives.

    §3's field breakdown of the worst trace measures ``completion`` at 23% against
    a 20% ``gen_ai.task.output`` / ``traceloop.entity.output`` pair — byte-identical
    to each other but, at a different size, not to ``completion``. Both are on §3's
    unconditional drop list, so no field carries that text out. Reporting it as
    ``dropped_as_duplicate`` would tell the agent it already has text the response
    does not contain; §2.3's contract is the opposite — it names twins of a field
    that *is* returned.
    """
    # GIVEN a Traceloop-plus-gen_ai span whose completion differs from its derived
    # output pair, the combination that leaves no non-derived output carrier
    completion = "c" * 4_700
    orphaned = "d" * 5_000
    payload = {
        "completion": completion,
        "gen_ai.task.output": orphaned,
        "traceloop.entity.output": orphaned,
    }

    # WHEN the payload is reduced
    kept, dropped = canonical_attributes(payload)

    # THEN the pair is reported under semconv, and every name in 'duplicate' is
    # backed by text the response still contains
    assert kept == {"completion": completion}
    assert dropped == {
        "duplicate": [],
        "semconv": ["gen_ai.task.output", "traceloop.entity.output"],
    }
    assert all(payload[name] in set(kept.values()) for name in dropped["duplicate"])


def test_canonical_attributes_reports_a_short_byte_identical_twin_as_a_duplicate() -> None:
    # GIVEN a derived twin below the dedup threshold, so it never enters the
    # byte-identity map at all
    short = "s" * (PAYLOAD_MIN_CHARS - 50)
    payload = {"prompt": short, "traceloop.entity.input": short}

    # WHEN the payload is reduced
    kept, dropped = canonical_attributes(payload)

    # THEN it is still reported as a duplicate, because its text does survive on
    # 'prompt'. The bucket describes what happened to the text, not which code path
    # decided the drop — §2.3 surfaces the two buckets to the agent as distinct
    # reasons.
    assert kept == {"prompt": short}
    assert dropped == {"duplicate": ["traceloop.entity.input"], "semconv": []}


def test_canonical_attributes_does_not_dedup_short_metadata_values() -> None:
    # GIVEN two unrelated metadata attributes that merely share a short value
    payload = {"gen_ai.system": "openai", "vendor.llm.provider": "openai", "http.status": "200"}

    # WHEN the payload is reduced
    kept, dropped = canonical_attributes(payload)

    # THEN nothing is dropped: dedup applies to payload text, not to metadata
    assert kept == payload
    assert dropped == {"duplicate": [], "semconv": []}


def test_canonical_attributes_passes_through_non_string_values() -> None:
    # GIVEN attribute values that are not strings, two of them equal
    payload = {"gen_ai.usage.input_tokens": 0, "gen_ai.usage.output_tokens": 0, "ok": True}

    # WHEN the payload is reduced
    kept, dropped = canonical_attributes(payload)

    # THEN they are kept untouched — byte-identity dedup is for text only
    assert kept == payload
    assert dropped == {"duplicate": [], "semconv": []}


def test_canonical_attributes_dedups_byte_identical_container_twins() -> None:
    # GIVEN the same message list written by two instrumentations
    messages = [{"role": "user", "content": "m" * (PAYLOAD_MIN_CHARS * 2)}]
    payload = {
        "gen_ai.input.messages": list(messages),
        "traceloop.entity.input": list(messages),
    }

    # WHEN the payload is reduced
    kept, dropped = canonical_attributes(payload)

    # THEN the higher-precedence carrier survives and the twin is a duplicate —
    # container twins dedup over their serialized JSON just like string twins
    assert list(kept) == ["gen_ai.input.messages"]
    assert dropped == {"duplicate": ["traceloop.entity.input"], "semconv": []}


def test_canonical_attributes_does_not_mutate_its_input(
    oversized_agent_trace: dict[str, Any],
) -> None:
    # GIVEN a span payload and a snapshot of it
    payload = _merged_payload(oversized_agent_trace["spans"][2])
    before = dict(payload)

    # WHEN the payload is reduced
    canonical_attributes(payload)

    # THEN the caller's mapping is untouched
    assert payload == before


# ------------------------------------------------------------------ #
# cap_value                                                          #
# ------------------------------------------------------------------ #


def test_cap_value_returns_the_whole_value_untouched_when_it_fits() -> None:
    # GIVEN a value shorter than the cap
    value = "short completion"

    # WHEN it is capped
    window, info = cap_value(value, MAX_FIELD_CHARS_SPAN_VIEW)

    # THEN it comes back whole with no truncation record at all
    assert window == value
    assert info is None


def test_cap_value_windows_are_contiguous_and_next_offset_lands_on_the_boundary(
    oversized_agent_trace: dict[str, Any],
) -> None:
    # GIVEN the single 160,000-char attribute value (~52k tokens on its own)
    value = oversized_agent_trace["spans"][3]["attributes"]["traceloop.entity.output"]
    assert len(value) == OVERSIZED_GIANT_ATTRIBUTE_CHARS
    assert len(value) / CHARS_PER_TOKEN > 50_000

    # WHEN it is paged one field window at a time, following next_offset
    windows: list[str] = []
    offsets: list[int] = []
    offset: int | None = 0
    while offset is not None:
        offsets.append(offset)
        window, info = cap_value(value, MAX_FIELD_CHARS_SPAN_VIEW, offset)
        windows.append(window)
        assert info is not None
        assert info["total_chars"] == OVERSIZED_GIANT_ATTRIBUTE_CHARS
        assert info["returned_chars"] == len(window)
        offset = info["next_offset"]

    # THEN the windows are contiguous, complete, and next_offset is None exactly
    # once — at the end of the value
    assert "".join(windows) == value
    assert offsets == list(range(0, OVERSIZED_GIANT_ATTRIBUTE_CHARS, MAX_FIELD_CHARS_SPAN_VIEW))
    assert len(windows) == OVERSIZED_GIANT_ATTRIBUTE_CHARS // MAX_FIELD_CHARS_SPAN_VIEW
    assert all(len(window) == MAX_FIELD_CHARS_SPAN_VIEW for window in windows)


def test_cap_value_reports_the_final_partial_window_as_the_end() -> None:
    # GIVEN a value that does not divide evenly by the cap
    value = "x" * 25

    # WHEN the last window is requested
    window, info = cap_value(value, 10, 20)

    # THEN it is short, and next_offset says there is nothing left
    assert window == "x" * 5
    assert info == {"returned_chars": 5, "total_chars": 25, "next_offset": None}


def test_cap_value_offset_past_the_end_returns_an_empty_window() -> None:
    # GIVEN an offset beyond the value
    # WHEN the value is capped
    window, info = cap_value("x" * 10, 10, 500)

    # THEN the caller gets nothing plus an explicit record, not an exception
    assert window == ""
    assert info == {"returned_chars": 0, "total_chars": 10, "next_offset": None}


def test_cap_value_offset_past_the_end_of_an_empty_value_still_reports() -> None:
    # GIVEN an empty field and an out-of-range field_offset
    # WHEN the value is capped
    window, info = cap_value("", MAX_FIELD_CHARS_SPAN_VIEW, 500)

    # THEN the agent is told its offset found nothing, rather than getting a bare
    # empty string with no entry in truncation.fields, which reads as an
    # untruncated field
    assert window == ""
    assert info == {"returned_chars": 0, "total_chars": 0, "next_offset": None}
    # THEN offset 0 on the same empty value is genuinely 'whole value returned'
    assert cap_value("", MAX_FIELD_CHARS_SPAN_VIEW) == ("", None)


def test_cap_value_non_positive_max_chars_disables_the_cap() -> None:
    # GIVEN a long value and a disabled cap
    value = "y" * 5_000

    # WHEN it is capped with 0 and with a negative bound
    whole, info = cap_value(value, 0)
    tail, tail_info = cap_value(value, -1, 4_000)

    # THEN the whole remainder comes back
    assert whole == value
    assert info is None
    assert tail == value[4_000:]
    assert tail_info == {"returned_chars": 1_000, "total_chars": 5_000, "next_offset": None}


def test_cap_value_negative_offset_clamps_to_the_start() -> None:
    # GIVEN a negative offset, which slicing would otherwise read from the end
    # WHEN the value is capped
    window, info = cap_value("abcdef", 3, -4)

    # THEN it reads from the start
    assert window == "abc"
    assert info == {"returned_chars": 3, "total_chars": 6, "next_offset": 3}


def test_cap_value_counts_characters_not_bytes() -> None:
    # GIVEN a multibyte value: byte-slicing it would split a code point
    value = "café—naïve—" * 10

    # WHEN it is capped at four characters
    window, info = cap_value(value, 4)

    # THEN exactly four characters come back and the totals are in characters
    assert window == "café"
    assert info is not None
    assert info["total_chars"] == len(value)
    assert len(value.encode("utf-8")) > len(value)


# ------------------------------------------------------------------ #
# cap_payload_value                                                  #
# ------------------------------------------------------------------ #


def test_cap_payload_value_passes_a_small_container_through_natively() -> None:
    # GIVEN a container attribute whose JSON fits the cap
    value = [{"role": "user", "content": "hi"}]

    # WHEN it is capped
    window, info = cap_payload_value(value, MAX_FIELD_CHARS_SPAN_VIEW)

    # THEN it comes back as the container itself, with no truncation record
    assert window == value
    assert info is None


def test_cap_payload_value_windows_an_oversized_list_over_its_json() -> None:
    # GIVEN a gen_ai.prompt-shaped list whose content dwarfs the cap — OTLP allows
    # array attributes, and 'cap only str' used to return this whole
    value = [{"role": "user", "content": "x" * 100_000}]
    serialized = json.dumps(value, default=str, ensure_ascii=False)

    # WHEN it is capped
    window, info = cap_payload_value(value, 2_000)

    # THEN the window is the serialized JSON's first 2,000 chars and the record
    # marks it as serialized, with next_offset for continuation
    assert window == serialized[:2_000]
    assert info == {
        "returned_chars": 2_000,
        "total_chars": len(serialized),
        "next_offset": 2_000,
        "serialized": True,
    }


def test_cap_payload_value_continues_a_container_from_an_offset() -> None:
    # GIVEN the same oversized container and the first call's next_offset
    value = [{"role": "user", "content": "x" * 100_000}]
    serialized = json.dumps(value, default=str, ensure_ascii=False)

    # WHEN the next window is requested
    window, info = cap_payload_value(value, 2_000, 2_000)

    # THEN the window continues contiguously over the serialized text
    assert window == serialized[2_000:4_000]
    assert info is not None
    assert info["serialized"] is True
    assert info["next_offset"] == 4_000


def test_cap_payload_value_matches_cap_value_for_strings() -> None:
    # GIVEN a plain string value
    value = "y" * 5_000

    # WHEN both helpers cap it
    # THEN they agree exactly — strings take the cap_value path unchanged
    assert cap_payload_value(value, 2_000) == cap_value(value, 2_000)
    assert cap_payload_value(value, 2_000, 2_000) == cap_value(value, 2_000, 2_000)


def test_cap_payload_value_leaves_scalars_untouched() -> None:
    # GIVEN metadata-sized scalar values
    # WHEN they are capped
    # THEN they pass through with no record — numbers, bools and None are
    # metadata-sized by construction
    for scalar in (200, 0.5, True, None):
        assert cap_payload_value(scalar, 10) == (scalar, None)


# ------------------------------------------------------------------ #
# apply_char_budget                                                  #
# ------------------------------------------------------------------ #


def test_apply_char_budget_never_exceeds_the_budget_and_reports_spans_dropped(
    oversized_agent_trace: dict[str, Any],
) -> None:
    # GIVEN the oversized trace projected to payload views with an 8,000-char
    # per-field cap: 80,187 chars, over the 60,000-char ceiling
    items = [
        _payload_view(span, MAX_FIELD_CHARS_SPAN_VIEW) for span in oversized_agent_trace["spans"]
    ]
    assert len(json.dumps(items, ensure_ascii=False)) > MAX_TOTAL_CHARS_DEFAULT

    # WHEN the budget is applied
    emitted, stats = apply_char_budget(items, MAX_TOTAL_CHARS_DEFAULT)

    # THEN the hard stop holds and the loss is reported, never silent
    assert len(json.dumps(emitted, ensure_ascii=False)) <= MAX_TOTAL_CHARS_DEFAULT
    assert stats["chars_used"] <= MAX_TOTAL_CHARS_DEFAULT
    assert stats["spans_dropped"] > 0
    assert stats["spans_returned"] + stats["spans_dropped"] == OVERSIZED_SPAN_COUNT
    assert stats["max_total_chars"] == MAX_TOTAL_CHARS_DEFAULT
    # THEN emission kept the caller's order
    assert [item["span_id"] for item in emitted] == [
        item["span_id"] for item in items[: len(emitted)]
    ]


def test_apply_char_budget_reports_zero_dropped_when_everything_fits(
    oversized_agent_trace: dict[str, Any],
) -> None:
    # GIVEN the same trace under the summary view's 2,000-char per-field cap
    items = [
        _payload_view(span, MAX_FIELD_CHARS_TRACE_VIEW) for span in oversized_agent_trace["spans"]
    ]

    # WHEN the budget is applied
    emitted, stats = apply_char_budget(items, MAX_TOTAL_CHARS_DEFAULT)

    # THEN nothing is dropped and spans_dropped is still reported
    assert len(emitted) == OVERSIZED_SPAN_COUNT
    assert stats["spans_dropped"] == 0
    assert "spans_dropped" in stats


def test_apply_char_budget_stops_at_the_first_item_that_does_not_fit() -> None:
    # GIVEN three items where the second is too large for the remaining budget
    items = [{"n": 0, "text": "a" * 10}, {"n": 1, "text": "b" * 500}, {"n": 2, "text": "c" * 10}]

    # WHEN a budget that only the first item fits is applied
    emitted, stats = apply_char_budget(items, 60)

    # THEN emission stops rather than skipping ahead to the smaller third item,
    # so the order the caller chose is the order the agent sees
    assert emitted == [items[0]]
    assert stats["spans_returned"] == 1
    assert stats["spans_dropped"] == 2


def test_apply_char_budget_returns_nothing_when_the_first_item_alone_overflows() -> None:
    # GIVEN a single item larger than the whole budget
    items = [{"text": "z" * 1_000}]

    # WHEN the budget is applied
    emitted, stats = apply_char_budget(items, 100)

    # THEN the budget still holds — a too-small budget yields an empty page and a
    # report, not an over-budget response
    assert emitted == []
    assert stats == {
        "spans_returned": 0,
        "spans_dropped": 1,
        "chars_used": 0,
        "max_total_chars": 100,
    }


def test_apply_char_budget_non_positive_budget_disables_the_cap() -> None:
    # GIVEN a budget of zero
    items = [{"text": "z" * 1_000}, {"text": "y" * 1_000}]

    # WHEN the budget is applied
    emitted, stats = apply_char_budget(items, 0)

    # THEN everything is emitted and the accounting says so
    assert emitted == items
    assert stats["spans_returned"] == 2
    assert stats["spans_dropped"] == 0


def test_apply_char_budget_chars_used_matches_the_serialized_page() -> None:
    # GIVEN items that all fit
    items = [{"n": index, "text": "a" * 100} for index in range(5)]

    # WHEN the budget is applied
    emitted, stats = apply_char_budget(items, MAX_TOTAL_CHARS_DEFAULT)

    # THEN chars_used is what actually reaches the model, not an undercount
    assert stats["chars_used"] == len(json.dumps(emitted, ensure_ascii=False))


def test_apply_char_budget_charges_non_ascii_payload_its_character_count() -> None:
    # GIVEN two pages with identical character counts, one ASCII and one CJK
    ascii_items = [{"text": "a" * 2_000} for _ in range(OVERSIZED_SPAN_COUNT)]
    cjk_items = [{"text": "あ" * 2_000} for _ in range(OVERSIZED_SPAN_COUNT)]

    # WHEN the same budget is applied to both
    ascii_emitted, ascii_stats = apply_char_budget(ascii_items, MAX_TOTAL_CHARS_DEFAULT)
    cjk_emitted, cjk_stats = apply_char_budget(cjk_items, MAX_TOTAL_CHARS_DEFAULT)

    # THEN the budget's unit is characters (§1's 3.1 chars/token conversion, §2.4's
    # "chars, not bytes"), so the language of a trace does not decide how many
    # spans come back. Measuring escape-expanded JSON charged CJK ~6x and returned
    # 4 spans of 12 where ASCII got all 12.
    assert len(ascii_emitted) == OVERSIZED_SPAN_COUNT
    assert len(cjk_emitted) == len(ascii_emitted)
    assert cjk_stats["chars_used"] == ascii_stats["chars_used"]
    assert cjk_stats["chars_used"] == len(json.dumps(cjk_emitted, ensure_ascii=False))


# ------------------------------------------------------------------ #
# the §3 regression guard                                            #
# ------------------------------------------------------------------ #


def test_dedup_alone_still_blows_the_budget_so_weakening_the_cap_is_a_regression(
    oversized_agent_trace: dict[str, Any],
) -> None:
    """Perfect dedup is not a substitute for the hard character cap.

    On the worst measured trace, 423,300 tokens of genuinely unique text remain
    after perfect dedup — still 2x a context window. Any change to this module
    that weakens ``apply_char_budget`` in favour of smarter dedup is a regression.
    """
    # GIVEN the oversized trace, reduced by perfect dedup and canonical semconv
    # selection with no cap applied at all
    unique_chars = 0
    for span in oversized_agent_trace["spans"]:
        kept, _ = canonical_attributes(_merged_payload(span))
        unique_chars += sum(
            len(value)
            for value in kept.values()
            if isinstance(value, str) and len(value) >= PAYLOAD_MIN_CHARS
        )

    # WHEN the remainder is measured against otel_trace_get's ceiling
    overshoot = unique_chars / MAX_TOTAL_CHARS_DEFAULT

    # THEN dedup removed most of the payload and is STILL nowhere near enough
    assert unique_chars == OVERSIZED_CANONICAL_PAYLOAD_CHARS
    assert unique_chars < OVERSIZED_TOTAL_PAYLOAD_CHARS * 0.4
    assert overshoot > 5, (
        f"Dedup left {unique_chars} chars ({unique_chars / CHARS_PER_TOKEN:.0f} tokens) "
        f"of genuinely unique text — {overshoot:.1f}x the {MAX_TOTAL_CHARS_DEFAULT}-char "
        "budget. Dedup does not bound output; only the hard character cap does. "
        "WEAKENING apply_char_budget IN FAVOUR OF SMARTER DEDUP IS A REGRESSION."
    )

    # THEN the cap, and only the cap, brings the response inside the budget
    items = [
        _payload_view(span, MAX_FIELD_CHARS_SPAN_VIEW) for span in oversized_agent_trace["spans"]
    ]
    emitted, stats = apply_char_budget(items, MAX_TOTAL_CHARS_DEFAULT)
    assert len(json.dumps(emitted, ensure_ascii=False)) <= MAX_TOTAL_CHARS_DEFAULT
    assert stats["spans_dropped"] > 0


# ------------------------------------------------------------------ #
# the small, easy population                                         #
# ------------------------------------------------------------------ #


def test_small_trace_survives_the_pipeline_untouched(small_trace: dict[str, Any]) -> None:
    # GIVEN the small non-agentic trace: no prompts, no completions, ~450 tokens
    assert len(json.dumps(small_trace)) / CHARS_PER_TOKEN < SMALL_TRACE_MAX_TOKENS

    # WHEN it goes through the summary projection and the budget
    summaries, stats = summarize_spans(small_trace["spans"])
    items = [_payload_view(span, MAX_FIELD_CHARS_TRACE_VIEW) for span in small_trace["spans"]]
    emitted, budget_stats = apply_char_budget(items, MAX_TOTAL_CHARS_DEFAULT)

    # THEN nothing is truncated, dropped or mangled: the easy case stays easy
    assert stats["total_payload_chars"] == 0
    assert all(summary["payload_chars"] == 0 for summary in summaries)
    assert all(summary["payload_fields"] == [] for summary in summaries)
    assert budget_stats["spans_dropped"] == 0
    assert len(emitted) == len(small_trace["spans"])
    for item in emitted:
        assert item["truncation"] == {
            "fields": {},
            "dropped_as_duplicate": [],
            "dropped_semconv": [],
        }
    # THEN the span tree is intact, root first
    assert [summary["name"] for summary in summaries] == ["GET /health", "db.query", "cache.get"]
    assert summaries[0]["parent_span_id"] is None
    assert all(summary["parent_span_id"] == summaries[0]["span_id"] for summary in summaries[1:])


# ------------------------------------------------------------------ #
# helpers                                                            #
# ------------------------------------------------------------------ #


def _merged_payload(span: dict[str, Any]) -> dict[str, Any]:
    """Merge a span's rank-1 response fields with its attributes, response first."""
    merged = {name: span[name] for name in ("prompt", "completion") if name in span}
    merged.update(span.get("attributes") or {})
    return merged


def _payload_view(span: dict[str, Any], max_field_chars: int) -> dict[str, Any]:
    """Compose the helpers the way otel_trace_get(view='payloads') will (§2.2).

    Lives here rather than in the module under test because no tool exists yet;
    it proves the four functions compose into the projection the plan specifies.
    """
    kept, dropped = canonical_attributes(_merged_payload(span))
    attributes: dict[str, Any] = {}
    fields: dict[str, Any] = {}
    for name, value in kept.items():
        if not isinstance(value, str):
            attributes[name] = value
            continue
        window, info = cap_value(value, max_field_chars)
        attributes[name] = window
        if info is not None:
            fields[name] = info
    return {
        "span_id": span["span_id"],
        "name": span["name"],
        "attributes": attributes,
        "truncation": {
            "fields": fields,
            "dropped_as_duplicate": dropped["duplicate"],
            "dropped_semconv": dropped["semconv"],
        },
    }


def _inflate_payloads(spans: list[dict[str, Any]], factor: int) -> list[dict[str, Any]]:
    """Multiply every payload-sized string in place, leaving metadata alone."""
    for span in spans:
        for field in ("prompt", "completion"):
            if isinstance(span.get(field), str) and len(span[field]) >= PAYLOAD_MIN_CHARS:
                span[field] = span[field] * factor
        span["attributes"] = {
            name: value * factor
            if isinstance(value, str) and len(value) >= PAYLOAD_MIN_CHARS
            else value
            for name, value in span["attributes"].items()
        }
    return spans
