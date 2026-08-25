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

"""Bounded projections of OTel trace payloads.

Pure helpers: no HTTP, no tool registration. drmcp's loader globs this module and
finds nothing to bind.

Why this is not drtools/panels/truncate.py: ``truncate_for_llm`` is
structure-preserving and generic (uniform per-string cap, per-array/-object caps,
depth collapse). It has no notion of a *total* budget, no notion of which of two
byte-identical attributes to keep, and its ``max_array_items=5`` would silently
drop 32 of a 37-span trace's spans. Wrong shape for this problem.

The number that sets the design: on the worst measured trace (4.76 MB, 37 spans,
1,560,241 tokens) even *perfect* dedup leaves 423,300 tokens of genuinely unique
text — still 2x a context window. Dedup does not solve this. Only
summary-by-default plus a hard character cap does. Weakening
``apply_char_budget`` in favour of smarter dedup is a regression.
"""

import json
from fnmatch import fnmatchcase
from typing import Any

from datarobot_genai.drtools.otel.constants import DERIVED_ATTRIBUTE_NAMES
from datarobot_genai.drtools.otel.constants import DERIVED_ATTRIBUTE_PATTERNS
from datarobot_genai.drtools.otel.constants import RESPONSE_PAYLOAD_FIELDS
from datarobot_genai.drtools.otel.constants import SEMCONV_PRECEDENCE
from datarobot_genai.drtools.otel.constants import UNRANKED_SEMCONV_PRECEDENCE

# A string field at least this long is treated as payload text: counted in
# ``payload_chars``, listed in ``payload_fields``, and eligible for byte-identity
# dedup. Shorter values (model names, token counts, ids) are metadata, and
# deduplicating those would drop unrelated attributes that merely share a value.
PAYLOAD_MIN_CHARS = 200

# Cap on the ``payload_fields`` list, so a span carrying hundreds of large
# attributes cannot make the summary grow with the payload. ``payload_chars``
# still counts every field; the omitted count is reported.
PAYLOAD_FIELDS_MAX = 12

# Cap on ``status_message`` in the summary projection. It is the one structural
# field that carries arbitrary text: ``SpanViewValidator.status_message`` is a
# StringField with no server-side max_length, so an exception traceback arrives
# verbatim. Nothing else backstops the summary view — §2.2 puts
# ``max_total_chars`` on view="payloads" only — and uncapped, a single
# 48,000-char traceback grows the projection 13x past the measured ~133 tok/span
# rate. Capped, the summary is O(span_count) rather than O(input size). 500 chars
# holds an exception type, its message and the first frame.
STATUS_MESSAGE_MAX_CHARS = 500

# Structural fields kept by ``summarize_spans``, in emission order.
SPAN_SUMMARY_FIELDS: tuple[str, ...] = (
    "span_id",
    "parent_span_id",
    "name",
    "status_code",
    "status_message",
    "kind",
    "service_name",
    "duration",
    "start_time",
)

# Chars each emitted item costs beyond its own JSON: json.dumps' default ", "
# separator, or the enclosing "[]" for the last item. Charging it makes
# ``chars_used`` equal to len(json.dumps(returned)) rather than an undercount, so
# the budget is a real bound on what reaches the model.
_ITEM_OVERHEAD_CHARS = 2


def summarize_spans(spans: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Span tree projection with per-span payload accounting.

    Returns (summaries, stats). Each summary carries the structural fields plus
    ``payload_chars`` and ``payload_fields``; ``stats`` carries the trace-wide
    ``total_payload_chars``. ~133 tokens per span, independent of payload size.

    ``payload_chars`` and ``payload_fields`` are the load-bearing detail: they
    tell the agent *where the mass is* for ~15 extra tokens a span, which turns
    drill-down into a deliberate, targeted second call instead of a blind one.
    Accounting is pre-dedup and pre-cap — it reports the raw payload as it exists
    on the trace, including byte-identical twins, because that is what an agent
    needs to see to choose a span. ``payload_fields`` is ordered largest first
    and capped at ``PAYLOAD_FIELDS_MAX``, adding ``payload_fields_omitted`` when
    the cap bites. ``status_message`` is capped at ``STATUS_MESSAGE_MAX_CHARS``
    with a ``…[truncated, N more chars]`` marker — it is the only structural field
    that can carry an unbounded traceback, and the summary view has no total
    budget behind it.
    """
    summaries: list[dict[str, Any]] = []
    total_payload_chars = 0

    for span in spans:
        summary: dict[str, Any] = {field: span.get(field) for field in SPAN_SUMMARY_FIELDS}
        summary["status_message"] = _capped_status_message(span.get("status_message"))
        sizes = _payload_sizes(span)
        payload_chars = sum(sizes.values())
        names = list(sizes)

        summary["payload_chars"] = payload_chars
        summary["payload_fields"] = names[:PAYLOAD_FIELDS_MAX]
        if len(names) > PAYLOAD_FIELDS_MAX:
            summary["payload_fields_omitted"] = len(names) - PAYLOAD_FIELDS_MAX

        total_payload_chars += payload_chars
        summaries.append(summary)

    return summaries, {"total_payload_chars": total_payload_chars}


def canonical_attributes(attributes: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Drop byte-identical twins and non-canonical semconv families.

    Returns (kept, dropped) where ``dropped`` distinguishes ``duplicate`` from
    ``semconv`` so callers can report which and why. Measured reduction: 36% from
    dedup, 42% from semconv selection, on the worst trace.

    Callers may include the span's own ``prompt`` and ``completion`` in
    ``attributes``: those are precedence rank 1, are never dropped, and win every
    byte-identity tie.

    A name lands in ``duplicate`` only when a byte-identical copy of its value
    actually survives in ``kept``, and in ``semconv`` when its text survives
    nowhere. §3 drops the derived families unconditionally, so the second case is
    reachable: a span instrumented only by Traceloop and gen_ai whose
    ``prompt``/``completion`` the platform's ``enrich_trace`` left empty — or
    filled with different text — has no surviving carrier for its output. Saying
    ``semconv`` there is the honest report; telling an agent "you already have
    this" about text the response does not contain is not. Either way the name is
    surfaced, so §2.3's ``fields`` parameter can fetch the field back by exact
    name. Both lists are sorted; no drop is silent.
    """
    names = list(attributes)
    derived = {name: _is_derived(name) for name in names}
    ranks = {name: _semconv_rank(name) for name in names}
    order = {name: index for index, name in enumerate(names)}

    twins: dict[str, list[str]] = {}
    for name in names:
        value = attributes[name]
        if isinstance(value, str) and len(value) >= PAYLOAD_MIN_CHARS:
            twins.setdefault(value, []).append(name)

    # Within a group of byte-identical values the keeper is the highest-precedence
    # member that is not itself a derived family: electing a derived member and
    # then dropping it would wipe out a group that had a viable carrier.
    keepers: set[str] = set()
    duplicated: set[str] = set()
    for members in twins.values():
        if len(members) < 2:
            continue
        duplicated.update(members)
        candidates = [name for name in members if not derived[name]]
        if candidates:
            keepers.add(min(candidates, key=lambda name: (ranks[name], order[name])))

    kept: dict[str, Any] = {}
    dropped_names: list[str] = []
    for name in names:
        if name in RESPONSE_PAYLOAD_FIELDS:
            kept[name] = attributes[name]
        elif derived[name] or (name in duplicated and name not in keepers):
            dropped_names.append(name)
        else:
            kept[name] = attributes[name]

    # Bucket by what actually survived, not by why the drop was decided. The
    # surviving set is length-unbounded on purpose: a short byte-identical twin is
    # still a duplicate even though it never entered the dedup map above.
    surviving = {value for value in kept.values() if isinstance(value, str)}
    dropped: dict[str, list[str]] = {"duplicate": [], "semconv": []}
    for name in dropped_names:
        value = attributes[name]
        bucket = "duplicate" if isinstance(value, str) and value in surviving else "semconv"
        dropped[bucket].append(name)

    dropped["duplicate"].sort()
    dropped["semconv"].sort()
    return kept, dropped


def cap_value(value: str, max_chars: int, offset: int = 0) -> tuple[str, dict[str, Any] | None]:
    """Window one string; second element records returned/total/next_offset.

    Windows, not dumps: one measured attribute value was 54,799 tokens and the
    largest was 740,000 chars, so "return the whole field" is not an option and
    ``next_offset`` is how an agent continues. ``next_offset`` is None at the end
    of the value.

    Both bounds are in characters, not bytes — we are slicing decoded ``str``,
    and byte-slicing UTF-8 breaks multibyte sequences. ``max_chars <= 0`` disables
    the cap; a negative ``offset`` clamps to 0 and an ``offset`` past the end
    returns an empty window. The second element is None only when the whole value
    is returned from offset 0 — the test is the *requested* offset, not the clamped
    one, so an out-of-range offset against an empty value still gets a record
    rather than looking like an untruncated field.
    """
    total_chars = len(value)
    start = max(0, min(offset, total_chars))
    end = total_chars if max_chars <= 0 else min(start + max_chars, total_chars)
    window = value[start:end]

    if offset <= 0 and end == total_chars:
        return window, None

    return window, {
        "returned_chars": len(window),
        "total_chars": total_chars,
        "next_offset": end if end < total_chars else None,
    }


def apply_char_budget(
    items: list[dict[str, Any]], max_total_chars: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Emit items in order until the budget is spent; report what was dropped.

    This is the hard stop, and the only thing that actually bounds output. Dedup
    (36%) and canonical-semconv selection (42%) are worth doing *before* the cap,
    never instead of it: strategy E — canonical semconv only, every value capped —
    still measured 64,641 tokens on the worst trace, which is why the character
    budget is not advisory.

    Cost is an item's serialized JSON length plus the array overhead it adds, so
    ``chars_used`` equals ``len(json.dumps(returned, ensure_ascii=False))`` and
    never undercounts what reaches the model. ``ensure_ascii=False`` is load-bearing,
    not cosmetic: the budget's unit is characters (§1's 3.1 chars/token conversion,
    §2.4's "chars, not bytes") and MCP does not escape non-ASCII on the wire, so
    measuring escape-expanded JSON charged a CJK payload ~6x its size and made the
    number of spans returned depend on the language of the trace. Emission stops at
    the first item that does not fit and every item from there on is dropped, which
    keeps the order the caller chose. ``max_total_chars <= 0`` disables the budget.
    Stat keys are span-named because paging a trace's spans is the caller this
    exists for.
    """
    emitted: list[dict[str, Any]] = []
    chars_used = 0

    for item in items:
        cost = len(json.dumps(item, default=str, ensure_ascii=False)) + _ITEM_OVERHEAD_CHARS
        if max_total_chars > 0 and chars_used + cost > max_total_chars:
            break
        emitted.append(item)
        chars_used += cost

    return emitted, {
        "spans_returned": len(emitted),
        "spans_dropped": len(items) - len(emitted),
        "chars_used": chars_used,
        "max_total_chars": max_total_chars,
    }


def _capped_status_message(value: Any) -> Any:
    """Cap the one structural field that can carry an arbitrary-length traceback."""
    if not isinstance(value, str):
        return value
    window, info = cap_value(value, STATUS_MESSAGE_MAX_CHARS)
    if info is None:
        return window
    return f"{window}…[truncated, {info['total_chars'] - info['returned_chars']} more chars]"


def _payload_sizes(span: dict[str, Any]) -> dict[str, int]:
    """Map a span's payload-sized string fields to their char counts, largest first.

    ``attributes`` is a dynamic dict on the wire, so it can carry a key literally
    named ``prompt`` or ``completion`` beside the span's own response field. Sizes
    accumulate under the shared name instead of overwriting, so ``payload_chars``
    counts both and never undercounts the payload actually on the span.
    """
    sizes: dict[str, int] = {}

    def add(name: str, value: Any) -> None:
        if isinstance(value, str) and len(value) >= PAYLOAD_MIN_CHARS:
            sizes[name] = sizes.get(name, 0) + len(value)

    for name in RESPONSE_PAYLOAD_FIELDS:
        add(name, span.get(name))

    attributes = span.get("attributes")
    if isinstance(attributes, dict):
        for name, value in attributes.items():
            add(str(name), value)

    return dict(sorted(sizes.items(), key=lambda item: (-item[1], item[0])))


def _semconv_rank(name: str) -> int:
    """Rank an attribute name by semconv family; lower wins a byte-identity tie."""
    for rank, family in enumerate(SEMCONV_PRECEDENCE):
        if name in family.exact or (family.prefixes and name.startswith(family.prefixes)):
            return rank
    return UNRANKED_SEMCONV_PRECEDENCE


def _is_derived(name: str) -> bool:
    """Report whether an attribute is a derived or duplicated family, dropped outright."""
    if name in DERIVED_ATTRIBUTE_NAMES:
        return True
    return any(fnmatchcase(name, pattern) for pattern in DERIVED_ATTRIBUTE_PATTERNS)
