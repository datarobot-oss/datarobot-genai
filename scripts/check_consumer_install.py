# Copyright 2025 DataRobot, Inc. and its affiliates.
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

"""Consumer install check for datarobot-genai.

uv's `[tool.uv]` override / exclude / constraint lists are read ONLY from the
workspace-root `pyproject.toml`. They are never published in a wheel's
`Requires-Dist` and never inherited by consumers, so a green CI in this repo is
not evidence that `pip install datarobot-genai[extra]` works for a downstream
consumer. This script builds throwaway single-extra consumer projects and runs
`uv lock` on each, twice: once with NO consumer config (the true naive
experience) and once with the shipped `uv_consumer_config.toml` block applied
(the documented setup).

Extras are checked individually, not combined, because that's how they're
actually consumed (`pip install datarobot-genai[langgraph]`, not every extra at
once -- only our pre-built Docker image installs everything together, and it
carries the overrides).

This is a resolution smoke test, not a CVE scanner: known-vulnerability
scanning and ticketing is handled separately (Trivy / GameWarden).

Example
-------
    uv run python scripts/check_consumer_install.py
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIPPED_CONFIG = (
    REPO_ROOT / "src" / "datarobot_genai" / "_packaging" / "uv_consumer_config.toml"
)

# Every extra defined in setup.py's `extras_require`.
ALL_EXTRAS = [
    "core",
    "crewai",
    "langgraph",
    "llamaindex",
    "nat",
    "auth",
    "eval",
    "drmcpbase",
    "drmcputils",
    "drmcp",
    "drtools",
    "dragent",
]

# Extras where a naive (no [tool.uv] config) install is EXPECTED to fail, because
# some *other* package in the graph declares a version cap that only a uv
# override/exclude can force past -- not something a setup.py floor bump can fix.
# This is exactly why uv_consumer_config.toml exists.
#   crewai: nvidia-nat-crewai==1.7.0 caps litellm<1.85.0; genai needs litellm>=1.91.1.
#   eval:   nemo-evaluator-launcher -> leptonai pins httpx==0.27.2; genai needs
#           litellm>=1.91.1, which needs httpx>=0.28.0.
# If an entry here starts resolving naively, the override/exclude behind it may no
# longer be needed. If an extra NOT here starts failing naively, that's a real
# regression.
EXPECTED_NAIVE_FAILURES = {"crewai", "eval"}

CONSUMER_PYPROJECT = """\
[project]
name = "clean-room-consumer"
version = "0.0.0"
requires-python = ">=3.11,<3.14"
dependencies = ["datarobot-genai[{extra}]"]

[tool.uv]
package = false
{uv_block}
[tool.uv.sources]
datarobot-genai = {{ path = "{wheel}" }}
"""


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a command, capturing combined output as text."""
    return subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def build_local_wheel(dest: Path) -> Path:
    """Build the current tree into ``dest`` and return the wheel path."""
    print("Building local wheel ...")
    result = run(["uv", "build", "--wheel", "--out-dir", str(dest)], cwd=REPO_ROOT)
    if result.returncode != 0:
        print(result.stdout)
        raise SystemExit("uv build failed")
    wheels = sorted(dest.glob("datarobot_genai-*.whl"))
    if not wheels:
        raise SystemExit("no wheel produced by uv build")
    return wheels[-1]


def write_consumer(project_dir: Path, extra: str, *, apply_config: bool, wheel: Path) -> None:
    """Materialize a single-extra consumer project directory."""
    uv_block = SHIPPED_CONFIG.read_text(encoding="utf-8") if apply_config else ""
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "pyproject.toml").write_text(
        CONSUMER_PYPROJECT.format(extra=extra, uv_block=uv_block, wheel=wheel.as_posix()),
        encoding="utf-8",
    )


def try_lock(project_dir: Path) -> tuple[bool, str]:
    """Attempt to resolve a consumer project; return (ok, output)."""
    result = run(["uv", "lock"], cwd=project_dir)
    return result.returncode == 0, result.stdout


def tail(output: str, n: int = 8) -> str:
    """Return the last n lines of output, indented for report display."""
    lines = "\n".join(output.strip().splitlines()[-n:])
    return "    " + lines.replace("\n", "\n    ")


def main() -> int:
    """Run the per-extra consumer install check and print a report."""
    workdir = Path(tempfile.mkdtemp(prefix="consumer-install-check-"))
    try:
        wheel = build_local_wheel(workdir / "dist")
        print("\n=== Consumer install check ===")
        print(f"target : datarobot-genai -- local build ({wheel.name})\n")

        col = f"{'extra':<12}{'naive':<20}{'configured':<12}"
        print(col)
        print("-" * len(col))

        problems: list[str] = []
        for extra in ALL_EXTRAS:
            naive_dir = workdir / f"naive_{extra}"
            write_consumer(naive_dir, extra, apply_config=False, wheel=wheel)
            naive_ok, naive_out = try_lock(naive_dir)

            conf_dir = workdir / f"configured_{extra}"
            write_consumer(conf_dir, extra, apply_config=True, wheel=wheel)
            conf_ok, conf_out = try_lock(conf_dir)

            expected_failure = extra in EXPECTED_NAIVE_FAILURES
            naive_label = "RESOLVED" if naive_ok else "FAILED"
            if expected_failure and not naive_ok:
                naive_label += " (expected)"
            print(f"{extra:<12}{naive_label:<20}{'RESOLVED' if conf_ok else 'FAILED':<12}")

            if naive_ok and expected_failure:
                problems.append(
                    f"{extra}: naive install unexpectedly RESOLVED -- its "
                    "EXPECTED_NAIVE_FAILURES entry may be stale; consider removing it."
                )
            if not naive_ok and not expected_failure:
                problems.append(
                    f"{extra}: naive install FAILED unexpectedly -- a uv override/exclude "
                    "may now be required for this extra.\n" + tail(naive_out)
                )
            if not conf_ok:
                problems.append(
                    f"{extra}: CONFIGURED install FAILED -- the shipped "
                    "uv_consumer_config.toml is incomplete!\n" + tail(conf_out)
                )

        print()
        if problems:
            print("=== problems ===\n")
            print("\n\n".join(problems))
            print()
            return 1

        print("All extras: naive results match expectations, configured resolves cleanly.")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
