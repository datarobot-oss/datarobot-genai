#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib


PYPROJECT = Path("pyproject.toml")


@dataclass
class Version:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, s: str) -> Version:
        m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", s)
        if not m:
            raise ValueError(f"unsupported version format: {s}")
        return cls(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    def bump(self, part: str) -> Version:
        if part == "patch":
            return Version(self.major, self.minor, self.patch + 1)
        if part == "minor":
            return Version(self.major, self.minor + 1, 0)
        if part == "major":
            return Version(self.major + 1, 0, 0)
        raise ValueError(f"unknown part: {part}")

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.major}.{self.minor}.{self.patch}"


def read_version() -> str:
    with PYPROJECT.open("rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


def write_version(new_version: str) -> None:
    text = PYPROJECT.read_text()
    new_text = re.sub(
        r"(?m)^version\s*=\s*\"[^\"]*\"",
        f"version = \"{new_version}\"",
        text,
        count=1,
    )
    PYPROJECT.write_text(new_text)


def pypi_has_version(package: str, version: str, repository: str) -> bool:
    index = "https://pypi.org/pypi" if repository == "pypi" else "https://test.pypi.org/pypi"
    url = f"{index}/{package}/json"
    try:
        out = subprocess.check_output(["python", "-c", f"import json,urllib.request as r;print(json.dumps(r.urlopen('{url}').read().decode()))"], text=True)
        data = json.loads(json.loads(out))
        return version in data.get("releases", {})
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Ensure and optionally bump version vs (Test)PyPI")
    parser.add_argument("--package", default="datarobot-genai")
    parser.add_argument("--repo", choices=["pypi", "testpypi"], default="pypi")
    parser.add_argument("--bump", choices=["patch", "minor", "major"], default="patch")
    parser.add_argument("--apply", action="store_true", help="write back bumped version")
    args = parser.parse_args()

    current = read_version()
    base = current.split(".")
    if any(part.startswith("dev") for part in base):
        print(f"Current version appears to be a dev/prerelease: {current}")
        return 0

    exists = pypi_has_version(args.package, current, args.repo)
    if not exists:
        print(f"OK: {args.repo} does not have {current}")
        return 0

    # version exists, bump
    v = Version.parse(current).bump(args.bump)
    print(f"Bumping {current} -> {v}")
    if args.apply:
        write_version(str(v))
        print("pyproject.toml updated")
    else:
        print("Run with --apply to write the new version")
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
