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


def test_setup_logging_dedupes_litellm_even_when_imported_after() -> None:
    """LiteLLM logs must not be duplicated.

    LiteLLM attaches its own stderr handler to the ``LiteLLM`` logger at import
    time while leaving ``propagate=True``, so each line is emitted twice (once by
    LiteLLM's handler, once by the root handler). ``setup_logging`` must leave the
    logger with no own handler and ``propagate=True`` even when LiteLLM is imported
    AFTER ``setup_logging`` runs (the real dragent serve ordering).
    """
    # Run in a fresh interpreter so we control import order: setup_logging FIRST,
    # then `import litellm` (which is when LiteLLM adds its handler).
    script = (
        "import logging;"
        "from datarobot_genai.core.utils.logging import setup_logging;"
        "setup_logging();"
        "import litellm;"  # noqa: F401 -- adds LiteLLM's own handler at import time
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
