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

"""Clean-room consumer audit for datarobot-genai.

A package's own CI is green because it applies its `[tool.uv]` override / exclude /
constraint lists at the workspace root. Those lists are NOT published and NOT
inherited by consumers, so a green CI is not evidence that a downstream consumer
is safe. This script removes that safety net on purpose and reports what a naive
consumer actually gets.

For a chosen extras set it builds two throwaway consumer projects and runs
`uv lock` on each:

  * NAIVE       -- depends on datarobot-genai[extras] with NO [tool.uv] config.
                   If this fails to lock, the overrides are load-bearing: a naive
                   consumer literally cannot install the package.
  * CONFIGURED  -- same, plus the shipped uv_consumer_config.toml block applied at
                   the consumer root (what the README tells consumers to do).

With --audit it exports the CONFIGURED lock and runs pip-audit over it, so you can
see whether the documented consumer setup is actually CVE-clean.

Examples
--------
    # Audit a local build of the current tree (default extras: crewai,dragent,auth)
    uv run python scripts/clean_room_audit.py

    # Audit a released version straight from PyPI, with a CVE scan
    uv run python scripts/clean_room_audit.py --version 0.23.22 --audit

    # Audit the full framework matrix
    uv run python scripts/clean_room_audit.py --extras crewai,langgraph,llamaindex,nat,dragent,drmcp,drtools,auth
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIPPED_CONFIG = (
    REPO_ROOT / "src" / "datarobot_genai" / "_packaging" / "uv_consumer_config.toml"
)
DEFAULT_EXTRAS = "crewai,dragent,auth"

# Minimal consumer pyproject. `{uv_block}` is either empty (naive) or the shipped
# override/exclude/constraint block (configured). `{source}` optionally pins the
# datarobot-genai requirement to a locally built wheel.
CONSUMER_PYPROJECT = """\
[project]
name = "clean-room-consumer"
version = "0.0.0"
requires-python = ">=3.11,<3.14"
dependencies = ["datarobot-genai[{extras}]{version_spec}"]

[tool.uv]
package = false
{uv_block}
{source}
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


def write_consumer(
    project_dir: Path,
    extras: str,
    *,
    apply_config: bool,
    version_spec: str,
    wheel: Path | None,
) -> None:
    """Materialize a consumer project directory."""
    uv_block = ""
    if apply_config:
        # The shipped file is top-level TOML arrays; drop it straight under [tool.uv].
        uv_block = SHIPPED_CONFIG.read_text(encoding="utf-8")
    source = ""
    if wheel is not None:
        source = f'[tool.uv.sources]\ndatarobot-genai = {{ path = "{wheel.as_posix()}" }}\n'
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "pyproject.toml").write_text(
        CONSUMER_PYPROJECT.format(
            extras=extras,
            version_spec=version_spec,
            uv_block=uv_block,
            source=source,
        ),
        encoding="utf-8",
    )


def try_lock(project_dir: Path) -> tuple[bool, str]:
    """Attempt to resolve a consumer project; return (ok, output)."""
    result = run(["uv", "lock"], cwd=project_dir)
    return result.returncode == 0, result.stdout


def audit_configured(project_dir: Path) -> tuple[bool, str]:
    """Export the lock and run pip-audit over it; return (clean, output)."""
    export = run(
        ["uv", "export", "--frozen", "--no-emit-project", "--format", "requirements-txt"],
        cwd=project_dir,
    )
    if export.returncode != 0:
        return False, "uv export failed:\n" + export.stdout
    reqs = project_dir / "audit-requirements.txt"
    reqs.write_text(export.stdout, encoding="utf-8")
    audit = run(
        ["uvx", "pip-audit", "--requirement", str(reqs), "--progress-spinner", "off"],
        cwd=project_dir,
    )
    return audit.returncode == 0, audit.stdout


def main() -> int:
    """Run the clean-room audit and print a report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extras",
        default=DEFAULT_EXTRAS,
        help=f"comma-separated extras to audit (default: {DEFAULT_EXTRAS})",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="audit this released version from PyPI instead of a local build",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="run pip-audit over the CONFIGURED lock",
    )
    args = parser.parse_args()

    extras = args.extras
    version_spec = f"=={args.version}" if args.version else ""

    workdir = Path(tempfile.mkdtemp(prefix="clean-room-audit-"))
    try:
        wheel: Path | None = None
        if not args.version:
            wheel = build_local_wheel(workdir / "dist")

        target = f"version {args.version} from PyPI" if args.version else f"local build ({wheel.name})"
        print(f"\n=== Clean-room consumer audit ===")
        print(f"target : datarobot-genai[{extras}] -- {target}\n")

        report: list[str] = []

        # 1) NAIVE consumer (no [tool.uv] config) -- the true downstream experience.
        naive_dir = workdir / "naive"
        write_consumer(
            naive_dir, extras, apply_config=False, version_spec=version_spec, wheel=wheel
        )
        naive_ok, naive_out = try_lock(naive_dir)
        if naive_ok:
            report.append("NAIVE consumer      : RESOLVED (overrides are not load-bearing for resolution)")
        else:
            tail = "\n".join(naive_out.strip().splitlines()[-8:])
            report.append(
                "NAIVE consumer      : FAILED to resolve -- overrides ARE required by consumers.\n"
                + "    " + tail.replace("\n", "\n    ")
            )

        # 2) CONFIGURED consumer (shipped block applied) -- the documented setup.
        conf_dir = workdir / "configured"
        write_consumer(
            conf_dir, extras, apply_config=True, version_spec=version_spec, wheel=wheel
        )
        conf_ok, conf_out = try_lock(conf_dir)
        if conf_ok:
            report.append("CONFIGURED consumer : RESOLVED (shipped uv_consumer_config.toml works)")
        else:
            tail = "\n".join(conf_out.strip().splitlines()[-12:])
            report.append(
                "CONFIGURED consumer : FAILED to resolve -- the shipped config is incomplete!\n"
                + "    " + tail.replace("\n", "\n    ")
            )

        audit_clean = None
        if args.audit and conf_ok:
            audit_clean, audit_out = audit_configured(conf_dir)
            status = "no known vulnerabilities" if audit_clean else "VULNERABILITIES FOUND"
            report.append(f"CONFIGURED CVE scan : {status}")
            report.append("    " + audit_out.strip().replace("\n", "\n    "))
        elif args.audit and not conf_ok:
            report.append("CONFIGURED CVE scan : skipped (configured lock failed)")

        print("\n".join(report))
        print()

        # Exit non-zero only when the documented consumer path is broken: the
        # configured lock must resolve, and (if requested) be CVE-clean.
        if not conf_ok:
            return 1
        if args.audit and audit_clean is False:
            return 2
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
