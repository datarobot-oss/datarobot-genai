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

"""Minimal subprocess runner mirroring ``datarobot-user-models``/``run_agent.py``.

Used by ``test_run_agent_inline.py`` to exercise ``execute_dragent_inline``
end-to-end as if it were running inside the agent custom-model environment:
chat completion params come in as JSON on the command line and the resulting
``ChatCompletion`` or list of ``ChatCompletionChunk`` is written to disk as
JSON for the test to read back.

CLI shape intentionally matches the relevant subset of ``run_agent.py``:
``--chat_completion``, ``--custom_model_dir``, ``--output_path``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from datarobot_genai.dragent import execute_dragent_inline

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chat_completion",
        required=True,
        help="OpenAI ChatCompletion create params as a JSON string.",
    )
    parser.add_argument(
        "--custom_model_dir",
        required=True,
        help="Directory holding the agent's workflow.yaml (and code).",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="File to write the result JSON to (matches run_agent.py's --output_path).",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = _parse_args()
    chat_completion = json.loads(args.chat_completion)
    custom_model_dir = Path(args.custom_model_dir)
    output_path = Path(args.output_path)

    logger.info("Executing dragent inline from %s", custom_model_dir)
    result = execute_dragent_inline(
        chat_completion=chat_completion,
        custom_model_dir=custom_model_dir,
    )

    if isinstance(result, list):
        payload = [chunk.model_dump(mode="json") for chunk in result]
    else:
        payload = result.model_dump(mode="json")

    output_path.write_text(json.dumps(payload))
    logger.info("Wrote result to %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
