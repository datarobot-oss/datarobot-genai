# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import subprocess
import sys


def test_unify_litellm_logging_dedupes_even_when_imported_after() -> None:
    """LiteLLM logs must not be duplicated.

    LiteLLM attaches its own stderr handler to the ``LiteLLM`` logger at import
    while leaving ``propagate=True``, so each line is emitted twice. After
    ``unify_litellm_logging()`` the logger must have no own handler and still
    ``propagate=True`` (so records flow through the unified root handler once) —
    and a later ``import litellm`` must not re-add a handler.
    """
    # Fresh interpreter: call unify_litellm_logging() (it imports litellm itself),
    # then import litellm again to prove no handler is re-added.
    script = (
        "import logging;"
        "from datarobot_genai.dragent.frontends.logging import unify_litellm_logging;"
        "unify_litellm_logging();"
        "import litellm;"  # noqa: F401 -- must not re-add LiteLLM's handler
        "lg = logging.getLogger('LiteLLM');"
        "print(f'HANDLERS={len(lg.handlers)} PROPAGATE={lg.propagate}')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    out = result.stdout.strip().splitlines()[-1]
    assert out == "HANDLERS=0 PROPAGATE=True", (
        f"LiteLLM logger should have no own handler and propagate to root; got: {out}"
    )


def test_register_import_wires_litellm_dedup() -> None:
    """The dedup must run via the real entry point, not only the helper.

    Importing the dragent frontend (the ``nat.front_ends`` entry point) runs
    ``logging_handler_setup()`` at module load, which must call
    ``unify_litellm_logging()`` — so a subsequent ``import litellm`` leaves the
    ``LiteLLM`` logger with no own handler and ``propagate=True``. This guards the
    wiring (logging_handler_setup -> unify_litellm_logging), which a direct-call
    test would miss if the call were removed.
    """
    script = (
        "import logging;"
        "import datarobot_genai.dragent.frontends.register;"  # noqa: F401 -- module-load runs logging_handler_setup()
        "import litellm;"  # noqa: F401
        "lg = logging.getLogger('LiteLLM');"
        "print(f'HANDLERS={len(lg.handlers)} PROPAGATE={lg.propagate}')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180, check=False
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    out = result.stdout.strip().splitlines()[-1]
    assert out == "HANDLERS=0 PROPAGATE=True", (
        f"importing the dragent frontend should dedupe LiteLLM logging; got: {out}"
    )
