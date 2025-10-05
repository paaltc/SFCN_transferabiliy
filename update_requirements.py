#!/usr/bin/env python3
"""Generate a requirements.txt from imports in the repository.

Usage:
  python update_requirements.py --write      # write requirements.txt (unpinned)
  python update_requirements.py --write --resolve  # write with pinned versions (best-effort)
  python update_requirements.py --install    # install packages found

This script heuristically extracts top-level package names from import statements
in .py files, filters out standard-library modules and local packages, and
produces a requirements.txt file. It can also attempt to pin versions by
querying the currently installed distributions.

Limitations:
- Mapping import names to pip package names is not always 1:1 (e.g. "sklearn" -> "scikit-learn").
- Version pinning requires the package to already be installed in the environment.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
import pkgutil
#!/usr/bin/env python3
"""Minimal, clean implementation to extract imports and write requirements.txt.

This file intentionally small to avoid previous corruption and is safe to run.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Set


ROOT = Path(__file__).resolve().parent


def collect_imports(root: Path) -> Set[str]:
    imports: Set[str] = set()
    for p in root.rglob("*.py"):
        if any(part.startswith(".") for part in p.parts):
            continue
        try:
            src = p.read_text(encoding="utf8")
        except Exception:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    imports.add(n.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    imports.add(node.module.split(".")[0])
    return imports


def write_requirements(path: Path, names: Set[str]) -> None:
    names = sorted(names)
    path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf8")


def main():
    root = ROOT
    imports = collect_imports(root)
    # Very small stdlib filter to exclude common standard modules (not exhaustive)
    stdlib = {
        "os",
        "sys",
        "re",
        "math",
        "json",
        "typing",
        "pathlib",
        "itertools",
        "collections",
        "datetime",
        "subprocess",
    }
    filtered = {n for n in imports if n not in stdlib and not n.startswith("_")}
    out = ROOT / "requirements.txt"
    write_requirements(out, filtered)
    print(f"Wrote {out} with {len(filtered)} packages")


if __name__ == "__main__":
    main()
    
