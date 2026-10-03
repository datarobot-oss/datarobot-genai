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

"""Constants shared by the OTel query tools.

Registers no tools; drmcp's loader globs this module but finds nothing to bind.
"""

from typing import Literal
from typing import NamedTuple
from typing import get_args

# Entity types the /otel routes accept. These are snake_case on the wire and in
# the path: drflask's camelize() only rewrites underscores in *keys*, and the
# ChoiceField behind entityType sets initial_camelization=False. So 'use_case'
# stays 'use_case' and must never be sent as 'useCase'.
EntityType = Literal[
    "deployment",
    "use_case",
    "experiment_container",
    "custom_application",
    "workload",
    "execution_environment",
    "custom_job",
    "artifact",
]

OTEL_ENTITY_TYPES: tuple[str, ...] = get_args(EntityType)

# Minimum log levels accepted by GET /otel/{entityType}/{entityId}/logs/.
#
# Deliberately NOT drmcputils.constants.LOG_LEVELS, which is only
# ("debug", "info", "warn", "error"). Reusing that tuple would reject 'warning'
# and 'critical', which this API accepts. 'level' is a *minimum*, not an exact
# match.
OTEL_LOG_LEVELS: tuple[str, ...] = (
    "debug",
    "info",
    "warn",
    "warning",
    "error",
    "critical",
)


class SemconvFamily(NamedTuple):
    """One semantic-convention family, matched by exact name or by prefix."""

    name: str
    exact: frozenset[str] = frozenset()
    prefixes: tuple[str, ...] = ()


# Precedence rank 1: the response's own normalized fields. The platform's
# enrich_trace() copies the best available prompt-/completion-like attribute into
# these, so they are guaranteed to carry the text of any derived attribute
# dropped below. Never dropped, and they win every byte-identity tie.
RESPONSE_PAYLOAD_FIELDS: tuple[str, ...] = ("prompt", "completion")

# Which family wins when several carry the same text, highest precedence first.
# Traceloop, NAT, OpenInference and gen_ai semconv each independently write the
# same prompt/completion text; on the worst measured trace 64.5% of the payload
# was byte-identical duplication.
SEMCONV_PRECEDENCE: tuple[SemconvFamily, ...] = (
    SemconvFamily("response", exact=frozenset(RESPONSE_PAYLOAD_FIELDS)),
    SemconvFamily("gen_ai", prefixes=("gen_ai.",)),
    SemconvFamily("openinference", exact=frozenset({"input.value", "output.value"})),
    SemconvFamily("traceloop", prefixes=("traceloop.",)),
    SemconvFamily("nat", prefixes=("nat.",)),
)

# Rank assigned to attributes belonging to no known family. Lowest precedence, so
# an unknown attribute loses a byte-identity tie against every known family.
UNRANKED_SEMCONV_PRECEDENCE: int = len(SEMCONV_PRECEDENCE)

# Attributes dropped outright as derived or duplicated (plan §3's drop list).
# Usually lossless: the platform's enrich_trace() copies the best available
# prompt-/completion-like attribute into RESPONSE_PAYLOAD_FIELDS, so the text
# survives there. Not lossless *unconditionally* — §3's own field breakdown of the
# worst trace measures 'completion' at 23% against a 20% gen_ai.task.output /
# traceloop.entity.output pair, i.e. text that is byte-identical to its twin but
# not to 'completion'. When a whole byte-identical group is dropped,
# canonical_attributes reports it as 'semconv' rather than claiming a surviving
# duplicate, and an agent can fetch the field back by exact name (§2.3 'fields').
# Every drop is reported.
DERIVED_ATTRIBUTE_NAMES: frozenset[str] = frozenset(
    {
        "gen_ai.task.input",
        "gen_ai.task.output",
        "traceloop.entity.input",
        "traceloop.entity.output",
        "nat.metadata",
    }
)

# Same list, for the entries that need a glob. Matched with fnmatch.fnmatchcase.
DERIVED_ATTRIBUTE_PATTERNS: tuple[str, ...] = (
    "gen_ai.completion.*.content",
    "*.value_obj",
)
