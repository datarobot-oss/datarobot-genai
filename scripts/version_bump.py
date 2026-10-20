#!/usr/bin/env python3
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
"""Bump the datarobot-genai version, and rebase a branch without version/changelog conflicts.

`rebase` rewrites every branch commit to drop its version bump and new CHANGELOG
sections, rebases onto the base, then re-applies one bump and the saved entries
on top of the base's version.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

PYPROJECT = "pyproject.toml"
CHANGELOG = "CHANGELOG.md"
LOCKS = ("uv.lock", "docs/uv.lock", "e2e-tests/uv.lock")

PYPROJECT_VERSION = re.compile(r'(?m)^(version = )"([^"]+)"')
LOCK_VERSION = re.compile(r'(name = "datarobot-genai"\nversion = )"([^"]+)"')
SECTION_HEADER = re.compile(r"(?m)^## (\S+)")


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def repo_root() -> Path:
    return Path(git("rev-parse", "--show-toplevel"))


def state_dir() -> Path:
    return Path(git("rev-parse", "--absolute-git-dir")) / "version-rebase"


def read_version(pyproject_text: str) -> str:
    match = PYPROJECT_VERSION.search(pyproject_text)
    if not match:
        sys.exit(f"No version found in {PYPROJECT}")
    return match.group(2)


def bumped(version: str, part: str) -> str:
    major, minor, patch = (int(n) for n in version.split("."))
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def set_version(root: Path, version: str, strict: bool = True) -> bool:
    """Write version to pyproject.toml and the datarobot-genai entry of every lock."""
    changed = False
    for name, pattern in [(PYPROJECT, PYPROJECT_VERSION)] + [
        (lock, LOCK_VERSION) for lock in LOCKS
    ]:
        path = root / name
        if not path.exists():
            if strict:
                sys.exit(f"{name} not found")
            continue
        text = path.read_text()
        new_text, count = pattern.subn(lambda m: f'{m.group(1)}"{version}"', text, count=1)
        if count == 0:
            if strict:
                sys.exit(f"No datarobot-genai version found in {name}")
            continue
        if new_text != text:
            path.write_text(new_text)
            changed = True
    return changed


def split_sections(changelog: str) -> tuple[str, list[tuple[str, str]]]:
    """Split a changelog into its preamble and (version, section text) pairs."""
    matches = list(SECTION_HEADER.finditer(changelog))
    if not matches:
        return changelog, []
    ends = [m.start() for m in matches[1:]] + [len(changelog)]
    sections = [(m.group(1), changelog[m.start() : end]) for m, end in zip(matches, ends)]
    return changelog[: matches[0].start()], sections


def section_body(section: str) -> str:
    return section.split("\n", 1)[1].strip("\n") if "\n" in section else ""


def cmd_bump(args: argparse.Namespace) -> None:
    root = repo_root()
    old = read_version((root / PYPROJECT).read_text())
    new = bumped(old, args.part)
    set_version(root, new)
    print(f"Bumped datarobot-genai {old} -> {new}")


def cmd_rebase(args: argparse.Namespace) -> None:
    root = repo_root()
    state = state_dir()
    if state.exists():
        sys.exit(
            "A version rebase is already in progress. "
            "Run `task rebase-continue` or `task rebase-abort`."
        )
    if git("status", "--porcelain"):
        sys.exit("Working tree is not clean. Commit or stash first.")
    branch = git("symbolic-ref", "--short", "-q", "HEAD", check=False)
    if not branch:
        sys.exit("Not on a branch.")

    if args.base.startswith("origin/"):
        git("fetch", "origin", args.base.removeprefix("origin/"))
    merge_base = git("merge-base", "HEAD", args.base)
    base_version = read_version(git("show", f"{merge_base}:{PYPROJECT}"))
    head_version = read_version((root / PYPROJECT).read_text())

    if head_version == base_version:
        print(f"Branch does not bump the version ({head_version}); doing a plain rebase.")
        sys.exit(subprocess.call(["git", "rebase", args.base]))

    base_parts = base_version.split(".")[:2]
    part = "minor" if head_version.split(".")[:2] != base_parts else "patch"
    base_sections = {v for v, _ in split_sections(git("show", f"{merge_base}:{CHANGELOG}"))[1]}
    _, head_sections = split_sections((root / CHANGELOG).read_text())
    entries = [section_body(s) for v, s in head_sections if v not in base_sections]

    state.mkdir()
    # Rebase checks out old commits, which may not contain this script, so run a copy.
    script = state / "version_bump.py"
    shutil.copy(__file__, script)
    (state / "state.json").write_text(
        json.dumps(
            {
                "base": args.base,
                "base_version": base_version,
                "base_sections": sorted(base_sections),
                "part": part,
                "entries": entries,
                "orig_head": git("rev-parse", "HEAD"),
                "branch": branch,
            },
            indent=2,
        )
    )
    print(
        f"Saved {len(entries)} CHANGELOG section(s) and a {part} bump. "
        f"Original HEAD: {git('rev-parse', 'HEAD')}",
        flush=True,
    )

    strip = f"{sys.executable} {script} strip"
    if (
        subprocess.call(["git", "rebase", "--quiet", "--force-rebase", merge_base, "--exec", strip])
        != 0
    ):
        sys.exit(
            "Stripping the version bump failed. Run `task rebase-abort` to restore the branch."
        )

    if subprocess.call(["git", "rebase", "--no-keep-empty", args.base]) != 0:
        sys.exit(
            "\nThe rebase stopped on a real conflict. Resolve it and `git add` the files, "
            "then run `task rebase-continue` (not `git rebase --continue`).\n"
            "`task rebase-abort` restores the branch as it was."
        )
    finish(root, state)


def cmd_strip(_: argparse.Namespace) -> None:
    """Rebase --exec hook: drop the version bump and new CHANGELOG sections from HEAD."""
    root = repo_root()
    saved = json.loads((state_dir() / "state.json").read_text())
    changed = set_version(root, saved["base_version"], strict=False)

    changelog = root / CHANGELOG
    if changelog.exists():
        text = changelog.read_text()
        preamble, sections = split_sections(text)
        kept = preamble + "".join(s for v, s in sections if v in saved["base_sections"])
        if kept != text:
            changelog.write_text(kept)
            changed = True

    if changed:
        git("add", "--", *[p for p in (PYPROJECT, CHANGELOG, *LOCKS) if (root / p).exists()])
        git("commit", "--amend", "--no-edit", "--allow-empty", "--no-verify")


def finish(root: Path, state: Path) -> None:
    saved = json.loads((state / "state.json").read_text())
    current = read_version((root / PYPROJECT).read_text())
    new = bumped(current, saved["part"])
    set_version(root, new)

    changelog = root / CHANGELOG
    preamble, sections = split_sections(changelog.read_text())
    body = "\n".join(e for e in saved["entries"] if e)
    entry = f"## {new} - {datetime.date.today().isoformat()}\n{body}\n\n"
    changelog.write_text(preamble + entry + "".join(s for _, s in sections))

    git("add", "--", PYPROJECT, CHANGELOG, *LOCKS)
    git("commit", "--no-verify", "-m", f"Bump version to {new}")
    shutil.rmtree(state)
    print(f"Rebased onto {saved['base']} and re-bumped datarobot-genai {current} -> {new}.")
    print(f"Check CHANGELOG.md, then force-push. Previous HEAD was {saved['orig_head']}.")


def cmd_continue(_: argparse.Namespace) -> None:
    state = state_dir()
    if not state.exists():
        sys.exit("No version rebase in progress.")
    git_dir = Path(git("rev-parse", "--absolute-git-dir"))
    if (git_dir / "rebase-merge").exists() or (git_dir / "rebase-apply").exists():
        env_call = ["git", "-c", "core.editor=true", "rebase", "--continue"]
        if subprocess.call(env_call) != 0:
            sys.exit("\nThe rebase stopped again. Resolve it, then run `task rebase-continue`.")
    finish(repo_root(), state)


def cmd_abort(_: argparse.Namespace) -> None:
    state = state_dir()
    if not state.exists():
        sys.exit("No version rebase in progress.")
    saved = json.loads((state / "state.json").read_text())
    git_dir = Path(git("rev-parse", "--absolute-git-dir"))
    if (git_dir / "rebase-merge").exists() or (git_dir / "rebase-apply").exists():
        git("rebase", "--abort")
    git("checkout", "--quiet", saved["branch"])
    git("reset", "--hard", "--quiet", saved["orig_head"])
    shutil.rmtree(state)
    print(f"Restored {saved['branch']} to {saved['orig_head']}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)

    bump = sub.add_parser("bump", help="Bump the version in pyproject.toml and every uv.lock")
    bump.add_argument("part", choices=["patch", "minor"])
    bump.set_defaults(func=cmd_bump)

    rebase = sub.add_parser("rebase", help="Rebase without version and CHANGELOG conflicts")
    rebase.add_argument("--base", default="origin/main")
    rebase.set_defaults(func=cmd_rebase)

    sub.add_parser("strip", help="Internal: rebase --exec hook").set_defaults(func=cmd_strip)
    sub.add_parser("continue", help="Continue after resolving a conflict").set_defaults(
        func=cmd_continue
    )
    sub.add_parser("abort", help="Restore the branch as it was").set_defaults(func=cmd_abort)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
