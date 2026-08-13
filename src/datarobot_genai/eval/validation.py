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
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request
from urllib.request import urlopen

import yaml

from datarobot_genai.eval.runner import resolve_benchmark_module


def preflight_judge(judge_cfg: dict[str, Any]) -> None:
    """Ping the judge endpoint with a minimal chat-completions call.

    Raises RuntimeError on any non-200 response or network failure so callers
    can bail before running the full benchmark — catches missing / invalid /
    expired tokens, wrong model_id, and gateway outages that would otherwise
    surface only as a cascade of per-case CALL_ERROR results.
    """
    url = judge_cfg["url"].rstrip("/") + "/chat/completions"
    model_id = judge_cfg["model_id"]
    api_key_name = judge_cfg["api_key_name"]
    token = os.environ.get(api_key_name)
    if not token:
        raise RuntimeError(
            f"Judge preflight failed: env var {api_key_name} is not set "
            f"(judge url={judge_cfg['url']}, model={model_id})"
        )

    # Reasoning models (o-series, etc.) spend tokens on an internal reasoning
    # pass before emitting output, so a tight max_tokens budget yields an HTTP
    # 400 even when auth and model are fine — a false negative. Give the ping
    # enough headroom to finish reasoning and emit at least one output token.
    payload = json.dumps(
        {
            "model": model_id,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 512,
            "temperature": 0,
        }
    ).encode()
    req = Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        urlopen(req, timeout=15)
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(
            f"Judge preflight failed: HTTP {e.code} from {url} "
            f"(model={model_id}, token env={api_key_name}). Response: {body}"
        ) from e
    except URLError as e:
        raise RuntimeError(
            f"Judge preflight failed: cannot reach {url} (model={model_id}): {e.reason}"
        ) from e


def health_check(endpoint_url: str) -> str | None:
    """Return None if the server responds at all, else an error string.

    Prefer a dedicated /health probe: dragent/DRUM-fronted agents expose a
    /health route (returning 200) at the host root, so probing it avoids the
    404 log noise that pinging a bare /v1-style base URL generates on the
    agent side. The endpoint's path (e.g. /v1) is stripped so /health resolves
    against the host:port.

    If /health isn't available (some deployed / production endpoints don't
    expose it), fall back to pinging the literal endpoint and treating any HTTP
    response (even 4xx/405) as "reachable" — a DRUM agentic-workflow server
    exposes /chat/completions but not necessarily /v1/models. Only a
    connection-level failure is fatal.
    """
    parts = urlsplit(endpoint_url)
    if parts.scheme and parts.netloc:
        health_url = f"{parts.scheme}://{parts.netloc}/health"
        try:
            req = Request(health_url, headers={"Accept": "application/json"})
            urlopen(req, timeout=10)
            return None
        except (HTTPError, URLError, OSError):
            # No /health route, or it errored — fall back to the base ping,
            # which either confirms reachability or reports the failure.
            pass

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
            raise ValueError(f"Pipeline {pipeline_path} missing required section: {key}")
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
            module_str = str(cfg["benchmark"]["module"])
            # Shared resolver with the runner so validation and execution agree
            # on where a benchmark lives (local file or installed package). It
            # raises if the module resolves to neither; we collect that as an
            # error rather than propagating.
            try:
                resolve_benchmark_module(module_str, repo_root)
            except ImportError:
                errors.append(f"Benchmark module not found: {repo_root / module_str}")
        except (ValueError, KeyError) as e:
            errors.append(f"Pipeline '{pipeline}' invalid: {e}")

    if not Path(dataset).exists():
        errors.append(f"Dataset not found: {dataset}")

    return errors
