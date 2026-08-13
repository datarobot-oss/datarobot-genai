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

"""Wire contract between the caller and the container runner.

The runner that executes user code *inside* the sandbox image is owned by
``datarobot/datarobot-user-models`` (``public_dropin_environments/
dr_mcp_execute_sandbox_minimal/runner.py``, PR
datarobot/datarobot-user-models#2137). That runner and this module are the
two ends of a small protocol used by the container-backed sandbox
(:class:`DataRobotWorkloadSandbox`):

- The runner emits its return value as a final stdout line prefixed with
  :data:`RESULT_MARKER`; :func:`parse_result_marker` strips and decodes it.
- The runner exits :data:`SANDBOX_TIMEOUT_EXIT_CODE` when its in-process
  wall-clock cap fires; the caller maps that to ``SandboxTimeout``.

Only these constants + the parser are shared, so we keep them here rather than
carrying a hand-synced duplicate of the whole runner in this repo (the runner
body would inevitably drift from the image copy).
"""

import json
from typing import Any

# Final stdout line emitted by the container runner: ``<marker><json>``.
RESULT_MARKER = "__DR_SANDBOX_RESULT__:"

# Exit code the runner uses when its in-process SIGALRM cap fires before the
# caller / workload-api timeout. Surfaced by the caller as ``SandboxTimeout``.
SANDBOX_TIMEOUT_EXIT_CODE = 124


def _marker_line_index(lines: list[str]) -> int:
    """Index of the last line that *is* a result marker, or ``-1`` if none.

    Scans from the end so trailing diagnostic lines don't hide the marker, and
    matches a line that *starts with* :data:`RESULT_MARKER` (not the marker as a
    substring anywhere) so incidental text can't be mistaken for a real marker.
    """
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith(RESULT_MARKER):
            return i
    return -1


def has_result_marker(stdout: str) -> bool:
    """Whether ``stdout`` contains a result-marker line (see :func:`parse_result_marker`)."""
    return _marker_line_index(stdout.splitlines()) != -1


def parse_result_marker(stdout: str) -> tuple[str, Any]:
    """Split the result marker off ``stdout``.

    Returns ``(clean_stdout, return_value)``: the last line that starts with
    :data:`RESULT_MARKER` is removed from the returned stdout and its JSON
    payload is decoded as the return value (``None`` on decode failure). When no
    marker line is present, stdout is returned unchanged with a ``None`` value.
    """
    lines = stdout.splitlines()
    idx = _marker_line_index(lines)
    if idx == -1:
        return stdout, None
    encoded = lines[idx][len(RESULT_MARKER) :]
    try:
        value = json.loads(encoded)
    except json.JSONDecodeError:
        value = None
    return "\n".join(lines[:idx] + lines[idx + 1 :]), value
