#!/usr/bin/env python3
"""
Custom import checker.

Enforces the following rules:
1. drtools cannot import from: core, drmcp, fastmcp
2. drmcp can only import from drtools subpackage, no other local subpackages
"""

import ast
import sys
from pathlib import Path
import tomllib
from typing import Any
from typing import List
from typing import Tuple

def load_config(base_dir: Path) -> dict[str, Any]:
    """Load configuration from pyproject.toml."""
    pyproject_path = base_dir / "pyproject.toml"
    if pyproject_path.exists():
        try:
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
                config = data.get("subpackages", {}).get("import-restrictions", {})
                # Transform the config to match what the script expects
                return {
                    "drtools_forbidden": config["drtools_forbidden"],
                    "drtools_allowed_subpackages": config["drtools_allowed_subpackages"],
                    "drmcp_allowed_subpackages": config["drmcp_allowed_subpackages"],
                }
        except Exception as e:
            print(f"Warning: Failed to load pyproject.toml: {e}")
            return {}

    return {}


class ImportChecker(ast.NodeVisitor):
    """AST visitor to check import statements."""

    def __init__(self, filepath: Path, config: dict[str, Any]):
        self.filepath = filepath
        self.errors: List[Tuple[int, str]] = []
        self.is_drtools = "drtools" in filepath.parts
        self.is_drmcp = "drmcp" in filepath.parts
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

    def _check_import(self, lineno: int, module_name: str) -> None:
        """Check if an import is allowed based on the rules."""
        if self.is_drtools:
            # Rules for drtools

            # Check if the import is forbidden
            forbidden_imports = self.config["drtools_forbidden"]
            for forbidden in forbidden_imports:
                if module_name.startswith(forbidden):
                    # Special case for auth import, we need to fix that in the future
                    if forbidden == "fastmcp" and (module_name.startswith("fastmcp.server.dependencies") or module_name.startswith("fastmcp.server.middleware")):
                        continue
                    else:
                        self.errors.append(
                            (lineno, f"drtools cannot import from '{forbidden}' (found: {module_name})")
                        )
            
            # Rules for drmcp
            allowed_local = self.config["drtools_allowed_subpackages"]
            if module_name.startswith("datarobot_genai."):
                # Extract the subpackage
                parts = module_name.split(".")
                if len(parts) >= 2:
                    subpackage = parts[1]
                    # Only allowed subpackages
                    if subpackage not in allowed_local:
                        if subpackage == "core" and module_name.startswith("datarobot_genai.core.utils.auth"):
                            pass
                        else:
                            self.errors.append(
                                (
                                    lineno,
                                    f"drtools can only import from {allowed_local} subpackages, not '{subpackage}' (found: {module_name})",
                                )
                            )

        elif self.is_drmcp:
            # Rules for drmcp
            allowed_local = self.config["drmcp_allowed_subpackages"]
            if module_name.startswith("datarobot_genai."):
                # Extract the subpackage
                parts = module_name.split(".")
                if len(parts) >= 2:
                    subpackage = parts[1]
                    # Only allowed subpackages
                    if subpackage not in allowed_local:
                        self.errors.append(
                            (
                                lineno,
                                f"drmcp can only import from {allowed_local} subpackages, not '{subpackage}' (found: {module_name})",
                            )
                        )


def check_file(filepath: Path, config: dict[str, Any]) -> List[Tuple[int, str]]:
    """Check a single Python file for import violations."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content, filename=str(filepath))
        checker = ImportChecker(filepath, config)
        checker.visit(tree)
        return checker.errors
    except Exception as e:
        return [(0, f"Error parsing file: {e}")]


def main():
    """Main entry point."""
    # Get all Python files in drtools and drmcp
    base_dir = Path(__file__).parent.parent
    src_dir = base_dir / "src" / "datarobot_genai"

    # Load configuration
    config = load_config(base_dir)

    all_errors = []

    # Check drtools files
    drtools_dir = src_dir / "drtools"
    if drtools_dir.exists():
        for py_file in drtools_dir.rglob("*.py"):
            errors = check_file(py_file, config)
            if errors:
                all_errors.append((py_file, errors))

    # Check drmcp files
    drmcp_dir = src_dir / "drmcp"
    if drmcp_dir.exists():
        for py_file in drmcp_dir.rglob("*.py"):
            errors = check_file(py_file, config)
            if errors:
                all_errors.append((py_file, errors))

    # Report results
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
