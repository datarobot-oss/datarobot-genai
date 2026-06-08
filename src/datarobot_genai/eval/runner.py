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
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_byob(
    cfg: dict[str, Any],
    endpoint: str,
    dataset_jsonl: str,
    output_dir: str,
    repo_root: Path,
) -> None:
    benchmark = cfg["benchmark"]
    target = cfg["target"]
    judge = cfg.get("judge") or {}
    run = cfg.get("run", {})

    module_path = str((repo_root / benchmark["module"]).absolute())

    env = {**os.environ}
    # Judge config consumed by judge-based benchmarks/*.py at import time. A
    # judge-free pipeline omits the `judge` section, so we export nothing and the
    # benchmark never calls judge_score().
    if judge:
        env["JUDGE_URL"] = str(judge["url"])
        env["JUDGE_MODEL_ID"] = str(judge["model_id"])
        env["JUDGE_API_KEY_NAME"] = str(judge.get("api_key_name", "DATAROBOT_API_TOKEN"))
    else:
        # Explicitly clear any inherited JUDGE_* vars so a judge-free pipeline
        # is not accidentally activated by a pre-existing shell environment.
        for _key in ("JUDGE_URL", "JUDGE_MODEL_ID", "JUDGE_API_KEY_NAME"):
            env.pop(_key, None)

    cmd = [
        sys.executable,
        "-m",
        "nemo_evaluator.contrib.byob.runner",
        "--benchmark-module",
        module_path,
        "--benchmark-name",
        str(benchmark["name"]),
        "--dataset",
        dataset_jsonl,
        "--model-type",
        str(target.get("model_type", "chat")),
        "--model-url",
        endpoint,
        "--model-id",
        str(target.get("model_id", "datarobot-agent")),
        "--output-dir",
        output_dir,
        "--save-predictions",
        "--parallelism",
        str(run.get("parallelism", 4)),
        "--max-tokens",
        str(run.get("max_tokens", 1024)),
        "--temperature",
        str(run.get("temperature", 0.0)),
        "--timeout-per-sample",
        str(run.get("timeout_per_sample", 180)),
    ]

    # Only pass the target API key name if that env var is actually set. A local
    # DRUM agent needs no auth; the runner errors if the name is given but unset.
    target_key_name: str | None = target.get("api_key_name")
    if target_key_name and os.environ.get(target_key_name):
        cmd += ["--api-key-name", str(target_key_name)]

    result = subprocess.run(cmd, env=env, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"BYOB runner exited with code {result.returncode}")
