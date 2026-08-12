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

"""Artifact image-build status semantics shared by the workload tools.

Builds progress PENDING -> IN_PROGRESS -> BUILT -> COMPLETED (or FAILED).
BUILT and COMPLETED are sequential, not equivalent: BUILT means the image was
built locally but has NOT been pushed to the registry yet, so it is not
deployable — scheduling a workload on it fails with
``422 runtime_image_uri ... None``. Only COMPLETED is a green light. Some
flows (e.g. Code-to-Workload) report lowercase/dash-cased status variants
(``pending`` / ``in-progress`` / ``completed``), so statuses are normalized
before interpretation.
"""

from typing import Any
from typing import NoReturn

from datarobot.errors import ClientError

from datarobot_genai.drmcputils.client_exceptions import raise_tool_error_for_client_error
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind

WAIT_FOR_COMPLETED_NOTE = (
    "Builds run asynchronously: poll artifact_get_build(artifact_id=..., build_id=...) "
    "and wait for status COMPLETED before scheduling a workload. BUILT means the "
    "image was built but not yet pushed to the registry — deploying then fails "
    "with `422 runtime_image_uri ... None`."
)

_RUNTIME_IMAGE_URI_HINT = (
    "Hint: a 422 mentioning `runtime_image_uri ... None` usually means the "
    "artifact's image build is still BUILT (built but not yet pushed to the "
    "registry) rather than COMPLETED, so it is not deployable yet. Poll "
    "artifact_get_build(artifact_id=..., build_id=...) until status is COMPLETED, "
    "then retry."
)

_GUIDANCE_BY_STATUS: dict[str, str] = {
    "FAILED": (
        "Build FAILED. Call artifact_get_build(artifact_id=..., build_id=..., "
        "include_logs=True) to see error details."
    ),
    "COMPLETED": (
        "Build COMPLETED — image built AND pushed to the registry. It is now "
        "deployable; the artifact's imageUri is populated."
    ),
    "BUILT": (
        "Build status is BUILT — the image was built locally but has NOT been "
        "pushed to the registry yet, so it is NOT deployable. Keep polling: only "
        "COMPLETED is a green light. Scheduling a workload now would fail with "
        "`422 runtime_image_uri ... None`. The gap to COMPLETED can be seconds "
        "to minutes for large images."
    ),
    "PENDING": "Build in progress. Check again later.",
    "IN_PROGRESS": "Build in progress. Check again later.",
}


def normalize_build_status(status: str) -> str:
    """Uppercase and underscore a build status (``in-progress`` -> ``IN_PROGRESS``)."""
    return (status or "").upper().replace("-", "_")


def annotate_build_deployability(build: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *build* with ``deployable`` and ``status_guidance`` attached.

    Unrecognized statuses are returned unchanged rather than guessed at.
    """
    norm = normalize_build_status(str(build.get("status") or ""))
    guidance = _GUIDANCE_BY_STATUS.get(norm)
    if guidance is None:
        return build
    annotated = dict(build)
    annotated["deployable"] = norm == "COMPLETED"
    annotated["status_guidance"] = guidance
    return annotated


def raise_tool_error_for_workload_client_error(exc: ClientError) -> NoReturn:
    """Map SDK errors like the shared helper, adding the BUILT-not-pushed 422 hint."""
    sc = getattr(exc, "status_code", None)
    if sc == 422 and "runtime_image_uri" in str(exc):
        raise ToolError(
            f"DataRobot API error ({sc}): {exc}\n{_RUNTIME_IMAGE_URI_HINT}",
            kind=ToolErrorKind.UPSTREAM,
        ) from exc
    raise_tool_error_for_client_error(exc)
