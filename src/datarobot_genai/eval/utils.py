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
from datetime import datetime
from pathlib import Path

# Fixed filenames in the output directory. They are "latest" pointers rewritten
# by every run, so a per-run archive may never claim one of these names.
RESERVED_OUTPUT_NAMES = frozenset({"eval_results.json", "eval_status.json"})


def make_run_id() -> str:
    """Timestamp identifying a single run.

    Microsecond precision, not seconds: the run ID names both the raw artifact
    directory and (by default) the archived results file, so two runs starting
    in the same second would otherwise overwrite each other's output. The format
    stays lexically sortable, which is what consumers listing the output
    directory rely on to order runs by name.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def resolve_archive_name(output_name: str | None, pipeline: str, run_id: str) -> str:
    """Filename for a run's archived results copy, relative to the output dir.

    Defaults to ``<pipeline stem>_<run_id>.json`` so successive runs of the same
    pipeline are self-describing, sort chronologically, and never overwrite one
    another. An explicit ``output_name`` must be a bare filename: the archive
    always lands in the output directory, and a path would let it escape.
    """
    if output_name:
        name = output_name if output_name.endswith(".json") else f"{output_name}.json"
        if Path(name).name != name or name.startswith("."):
            raise ValueError(
                f"Output name must be a bare filename with no directory separators "
                f"and no leading dot, got {output_name!r}"
            )
    else:
        name = f"{Path(pipeline).stem}_{run_id}.json"

    if name in RESERVED_OUTPUT_NAMES:
        reserved = ", ".join(sorted(RESERVED_OUTPUT_NAMES))
        raise ValueError(f"Output name may not be one of the reserved names: {reserved}")
    return name
