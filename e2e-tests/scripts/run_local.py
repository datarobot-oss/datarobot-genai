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

"""Sequential local runner for e2e cases.

Drives the same combinations that GitHub Actions would: parses a cases file
via ``cases.py``, then for each combination runs the full lifecycle locally —
``uv sync --group dragent-<agent>``, start the agent server, wait for
``/health``, run pytest, stop the server.

Combinations execute one at a time on the fixed agent port (8080); for fast
iteration use ``--no-install`` and/or ``--no-server`` to skip steps you've
already done in another shell.

This script does NOT load ``.env`` itself — invoke it via ``task cases-run``
(or any other task), which already wires up the project's ``dotenv:`` chain.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

from cases import Combination
from cases import expand
from cases import load_cases
from cases import resolve_case_file

E2E_ROOT = Path(__file__).resolve().parent.parent
HEALTH_URL = "http://localhost:8080/health"
HEALTH_TIMEOUT_S = 60
HEALTH_POLL_S = 1.0
SERVER_GRACE_S = 5.0


class ComboResult(NamedTuple):
    name: str
    status: str  # PASS | FAIL | TIMEOUT | SKIPPED
    detail: str = ""


def _parse_overrides(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(
                f"override {item!r} must be in KEY=VALUE form"
            )
        k, v = item.split("=", 1)
        if not k:
            raise SystemExit(f"override {item!r} has empty key")
        out[k] = v
    return out


def _build_env(combo: Combination) -> dict[str, str]:
    """Merge inherited environment with the case env + matrix combo overlay.

    The ``dotenv:`` chain in ``e2e-tests/Taskfile.yaml`` is responsible for
    loading ``.env*`` files — by the time this script runs, those values are
    already in ``os.environ``. The case ``env`` and matrix combination then
    overlay on top, so case-specific knobs always win.
    """
    merged = dict(os.environ)
    merged.update(combo.env)
    return merged


def _print_plan(combos: list[Combination]) -> None:
    print(f"# {len(combos)} combination(s) to run", file=sys.stderr)
    for combo in combos:
        print(f"\n# === {combo.name} ===")
        for k, v in sorted(combo.env.items()):
            print(f"export {k}={shlex.quote(v)}")
        print(f"# (in another shell) task dragent:run-{combo.agent}")
        pytest_cmd = [
            "uv", "run", "pytest",
            *combo.tests, *combo.pytest_args,
        ]
        print(" ".join(shlex.quote(p) for p in pytest_cmd))


def _uv_sync(agent: str) -> None:
    print(f"\n>>> uv sync --group dragent-{agent}", file=sys.stderr)
    subprocess.run(
        ["uv", "sync", "--group", f"dragent-{agent}"],
        cwd=str(E2E_ROOT),
        check=True,
    )


def _wait_for_health() -> bool:
    """Poll the agent ``/health`` endpoint until 200 or timeout."""
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=2) as resp:  # noqa: S310
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(HEALTH_POLL_S)
    return False


def _start_server(combo: Combination, env: dict[str, str]) -> subprocess.Popen[bytes]:
    """Launch the agent server in its own process group."""
    print(f">>> task dragent:run-{combo.agent}", file=sys.stderr)
    return subprocess.Popen(
        ["task", f"dragent:run-{combo.agent}"],
        cwd=str(E2E_ROOT),
        env=env,
        # Own process group so SIGTERM hits `task` and the agent it spawned.
        start_new_session=True,
    )


def _stop_server(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    pgid = os.getpgid(proc.pid)
    print(f">>> stopping agent (pgid={pgid})", file=sys.stderr)
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=SERVER_GRACE_S)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=SERVER_GRACE_S)


def _apply_override(combo: Combination) -> Path | None:
    """Copy the combo's WORKFLOW_FILE from dragent/overrides into the agent dir.

    Shared overlays live once in ``dragent/overrides``; copying the one in use
    next to the agent's ``workflow.yaml`` lets its relative ``base:`` resolve.
    Skips when the agent already has its own file. Returns the copy to remove.
    """
    workflow_file = combo.env.get("WORKFLOW_FILE", "workflow.yaml")
    src = E2E_ROOT / "dragent" / "overrides" / workflow_file
    dst = E2E_ROOT / "dragent" / combo.agent / workflow_file
    if not src.exists() or dst.exists():
        return None
    shutil.copyfile(src, dst)
    return dst


def _run_pytest(combo: Combination, env: dict[str, str]) -> int:
    results_dir = E2E_ROOT / "test_results"
    results_dir.mkdir(exist_ok=True)
    junit_path = results_dir / f"{combo.name}.xml"
    cmd = [
        "uv", "run", "pytest",
        *combo.tests,
        *combo.pytest_args,
        "--junitxml", str(junit_path),
    ]
    print(f">>> {' '.join(shlex.quote(p) for p in cmd)}", file=sys.stderr)
    return subprocess.run(cmd, cwd=str(E2E_ROOT), env=env, check=False).returncode


def _run_one(
    combo: Combination,
    *,
    no_server: bool,
) -> ComboResult:
    env = _build_env(combo)

    # Materialize the workflow overlay in use next to the agent's workflow.yaml
    # so its relative ``base: workflow.yaml`` resolves; removed again below.
    override_copy = _apply_override(combo)

    server: subprocess.Popen[bytes] | None = None
    try:
        if not no_server:
            server = _start_server(combo, env)
            if not _wait_for_health():
                return ComboResult(
                    combo.name,
                    "TIMEOUT",
                    f"agent did not become healthy within {HEALTH_TIMEOUT_S}s",
                )

        rc = _run_pytest(combo, env)
        return ComboResult(
            combo.name,
            "PASS" if rc == 0 else "FAIL",
            f"pytest exit={rc}" if rc else "",
        )
    finally:
        if server is not None:
            _stop_server(server)
        if override_copy is not None:
            override_copy.unlink(missing_ok=True)


def _print_summary(results: list[ComboResult]) -> None:
    print("\n=== summary ===", file=sys.stderr)
    width = max((len(r.name) for r in results), default=0)
    for r in results:
        line = f"  {r.name.ljust(width)}  {r.status}"
        if r.detail:
            line += f"  ({r.detail})"
        print(line, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_local.py",
        description="Run e2e case combinations locally, one at a time.",
    )
    parser.add_argument("case_file",
                        help="Case YAML to read. A bare file name (e.g. "
                             "'pr-tests.yaml') is resolved against "
                             "e2e-tests/cases/; anything containing a slash "
                             "is used as-is.")
    parser.add_argument("--case", default=None,
                        help="Limit to a single case by name.")
    parser.add_argument("--no-install", action="store_true",
                        help="Skip `uv sync` between combos.")
    parser.add_argument("--no-server", action="store_true",
                        help="Don't start the agent; assume one is running on "
                             "localhost:8080.")
    parser.add_argument("--print", dest="dry_run", action="store_true",
                        help="Print the plan, env exports, and pytest "
                             "command for each combo, then exit 0.")
    parser.add_argument("--keep-going", action="store_true",
                        help="Don't bail on first failure when running "
                             "multiple combos.")
    parser.add_argument("overrides", nargs="*",
                        help="KEY=VALUE filters that pin matrix dimensions "
                             "(e.g. AGENT=nat).")
    # parse_intermixed_args handles mixed-order: positionals interleaved
    # with options. Plain parse_args bails on `--case X AGENT=nat`.
    args = parser.parse_intermixed_args(argv)

    overrides = _parse_overrides(args.overrides)

    try:
        case_path = resolve_case_file(args.case_file)
        cases = load_cases(case_path)
        combos = expand(cases, case_name=args.case, overrides=overrides)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not combos:
        print("No combinations matched the given filters.", file=sys.stderr)
        return 1

    if args.dry_run:
        _print_plan(combos)
        return 0

    results: list[ComboResult] = []
    last_agent: str | None = None
    interrupted = False

    def _on_signal(signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        print(f"\n>>> received signal {signum}, tearing down...",
              file=sys.stderr)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    for combo in combos:
        if interrupted:
            results.append(ComboResult(combo.name, "SKIPPED", "interrupted"))
            continue

        if not args.no_install and combo.agent != last_agent:
            try:
                _uv_sync(combo.agent)
            except subprocess.CalledProcessError as exc:
                results.append(ComboResult(
                    combo.name, "FAIL", f"uv sync failed (rc={exc.returncode})",
                ))
                if not args.keep_going:
                    break
                continue
            last_agent = combo.agent

        try:
            result = _run_one(combo, no_server=args.no_server)
        except KeyboardInterrupt:
            results.append(ComboResult(combo.name, "SKIPPED", "interrupted"))
            interrupted = True
            continue

        results.append(result)
        if result.status != "PASS" and not args.keep_going:
            break

    _print_summary(results)

    failed = [r for r in results if r.status not in ("PASS", "SKIPPED")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
