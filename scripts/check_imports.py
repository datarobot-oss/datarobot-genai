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

"""
Custom import checker.

Enforces the following rules:
1. drtools cannot import from: core, drmcp, drmcpbase, fastmcp
2. drmcp can only import from drtools, drmcp, and drmcpbase subpackages
3. drmcpbase can only import from drmcpbase (must not import drtools, drmcp, or core)
"""

import ast
import sys
from pathlib import Path
import tomllib
from typing import Any


def _is_under_module(module_name: str, parent: str) -> bool:
    """Return True if module_name is parent or a submodule of parent."""
    return module_name == parent or module_name.startswith(f"{parent}.")


def load_config(base_dir: Path) -> dict[str, Any]:
    """Load configuration from pyproject.toml."""
    pyproject_path = base_dir / "pyproject.toml"
    if pyproject_path.exists():
        try:
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
                config = data.get("subpackages", {}).get("import-restrictions", {})
                return {
                    "drtools_forbidden": config["drtools_forbidden"],
                    "drtools_allowed_subpackages": config["drtools_allowed_subpackages"],
                    "drmcp_allowed_subpackages": config["drmcp_allowed_subpackages"],
                    "drmcpbase_allowed_subpackages": config["drmcpbase_allowed_subpackages"],
                    "drmcpbase_forbidden": config["drmcpbase_forbidden"],
                    "drmcputils_allowed_subpackages": config["drmcputils_allowed_subpackages"],
                    "drmcputils_forbidden": config["drmcputils_forbidden"],
                }
        except Exception as e:
            print(f"Warning: Failed to load pyproject.toml: {e}")
            return {}

    return {}


class ImportChecker(ast.NodeVisitor):
    """AST visitor to check import statements."""

    def __init__(self, filepath: Path, config: dict[str, Any]):
        self.filepath = filepath
        self.errors: list[tuple[int, str]] = []
        parts = filepath.parts
        self.is_drmcputils = "drmcputils" in parts
        self.is_drtools = "drtools" in parts
        self.is_drmcp = "drmcp" in parts and "drmcpbase" not in parts and not self.is_drmcputils
        self.is_drmcpbase = "drmcpbase" in parts
        self.config = config

    def visit_Import(self, node: ast.Import) -> None:
        """Check regular import statements."""
        for alias in node.names:
            self._check_import(node.lineno, alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check from ... import statements."""
        if node.module:
            self._check_import(node.lineno, node.module)
        self.generic_visit(node)

    def _check_forbidden(
        self,
        lineno: int,
        module_name: str,
        forbidden_imports: list[str],
        package_label: str,
    ) -> None:
        for forbidden in forbidden_imports:
            if _is_under_module(module_name, forbidden):
                self.errors.append(
                    (
                        lineno,
                        f"{package_label} cannot import from '{forbidden}' (found: {module_name})",
                    )
                )

    def _check_allowed_local(
        self,
        lineno: int,
        module_name: str,
        allowed_local: list[str],
        package_label: str,
    ) -> None:
        if not module_name.startswith("datarobot_genai."):
            return
        parts = module_name.split(".")
        if len(parts) < 2:
            return
        subpackage = parts[1]
        if subpackage in allowed_local:
            return
        if (
            package_label in ("drtools", "drmcputils")
            and subpackage == "core"
            and module_name.startswith("datarobot_genai.core.utils.auth")
        ):
            return
        self.errors.append(
            (
                lineno,
                f"{package_label} can only import from {allowed_local} subpackages, "
                f"not '{subpackage}' (found: {module_name})",
            )
        )

    def _check_import(self, lineno: int, module_name: str) -> None:
        """Check if an import is allowed based on the rules."""
        if self.is_drmcputils:
            self._check_forbidden(
                lineno, module_name, self.config["drmcputils_forbidden"], "drmcputils"
            )
            self._check_allowed_local(
                lineno,
                module_name,
                self.config["drmcputils_allowed_subpackages"],
                "drmcputils",
            )

        elif self.is_drtools:
            self._check_forbidden(
                lineno, module_name, self.config["drtools_forbidden"], "drtools"
            )
            self._check_allowed_local(
                lineno,
                module_name,
                self.config["drtools_allowed_subpackages"],
                "drtools",
            )

        elif self.is_drmcp:
            self._check_allowed_local(
                lineno,
                module_name,
                self.config["drmcp_allowed_subpackages"],
                "drmcp",
            )

        elif self.is_drmcpbase:
            self._check_forbidden(
                lineno, module_name, self.config["drmcpbase_forbidden"], "drmcpbase"
            )
            self._check_allowed_local(
                lineno,
                module_name,
                self.config["drmcpbase_allowed_subpackages"],
                "drmcpbase",
            )


def check_file(filepath: Path, config: dict[str, Any]) -> list[tuple[int, str]]:
    """Check a single Python file for import violations."""
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content, filename=str(filepath))
        checker = ImportChecker(filepath, config)
        checker.visit(tree)
        return checker.errors
    except Exception as e:
        return [(0, f"Error parsing file: {e}")]


def main():
    """Run the import checker over the configured subpackages."""
    base_dir = Path(__file__).parent.parent
    src_dir = base_dir / "src" / "datarobot_genai"

    config = load_config(base_dir)

    all_errors = []

    for subpackage in ("drmcputils", "drtools", "drmcp", "drmcpbase"):
        package_dir = src_dir / subpackage
        if package_dir.exists():
            for py_file in package_dir.rglob("*.py"):
                errors = check_file(py_file, config)
                if errors:
                    all_errors.append((py_file, errors))

    if all_errors:
        print("❌ Import violations found:\n")
        for filepath, errors in all_errors:
            rel_path = filepath.relative_to(base_dir)
            print(f"  {rel_path}:")
            for lineno, error in errors:
                print(f"    Line {lineno}: {error}")
        print(f"\n❌ Total files with violations: {len(all_errors)}")
        sys.exit(1)
    else:
        print("✅ All imports are compliant with the rules")
        sys.exit(0)


if __name__ == "__main__":
    main()
