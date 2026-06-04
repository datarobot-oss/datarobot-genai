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
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


def health_check(endpoint_url: str) -> str | None:
    """Return None if the server responds at all, else an error string.

    A DRUM agentic-workflow server exposes /chat/completions but not
    necessarily /v1/models, so any HTTP response (even 4xx/405) means the
    server is up and reachable. Only a connection-level failure is fatal.
    """
    base = endpoint_url.rstrip("/")
    try:
        req = Request(base, headers={"Accept": "application/json"})
        urlopen(req, timeout=10)
        return None
    except HTTPError:
        # Server responded (e.g. 404/405 on the root path) — it's reachable.
        return None
    except URLError as e:
        return f"Endpoint not reachable at {base}: {e.reason}"
    except Exception as e:  # noqa: BLE001
        return f"Endpoint check failed: {e}"


def load_pipeline(pipeline_path: Path) -> dict[str, Any]:
    cfg: Any = yaml.safe_load(pipeline_path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"Pipeline {pipeline_path} did not parse to a mapping")
    # `judge` is optional: judge-free benchmarks (PII, prompt-injection, exact
    # match, …) score deterministically and omit the section entirely. The UI
    # keys off its presence/absence to show whether a run needs a judge model.
    for key in ("benchmark", "target"):
        if key not in cfg:
            raise ValueError(
                f"Pipeline {pipeline_path} missing required section: {key}"
            )
    return cfg


def validate_inputs(
    endpoint: str,
    pipeline: str,
    dataset: str,
    pipelines_dir: Path,
    repo_root: Path,
) -> list[str]:
    errors: list[str] = []

    error = health_check(endpoint)
    if error:
        errors.append(f"Health check failed — {error}")

    pipeline_path = pipelines_dir / pipeline
    if not pipeline_path.exists():
        available = [f.name for f in pipelines_dir.glob("*.yaml")]
        errors.append(
            f"Pipeline '{pipeline}' not found in user_pipelines/. Available: {available or 'none'}"
        )
    else:
        try:
            cfg = load_pipeline(pipeline_path)
            module = repo_root / cfg["benchmark"]["module"]
            if not module.exists():
                errors.append(f"Benchmark module not found: {module}")
        except (ValueError, KeyError) as e:
            errors.append(f"Pipeline '{pipeline}' invalid: {e}")

    if not Path(dataset).exists():
        errors.append(f"Dataset not found: {dataset}")

    return errors
