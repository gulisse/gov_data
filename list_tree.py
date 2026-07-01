#!/usr/bin/env python3
"""
list_tree.py — walk all subfolders from a root and list their contents.

Usage:
    python list_tree.py                 # current directory
    python list_tree.py /path/to/root   # given directory
"""

import os
import sys

SKIP = {".git", "node_modules", "__pycache__", ".venv", "venv",
        ".mypy_cache", ".pytest_cache", ".ipynb_checkpoints"}


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.0f}{u}" if u == "B" else f"{n:.1f}{u}"
        n /= 1024


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        print(f"Not a directory: {root}")
        return 2

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP)
        depth = dirpath[len(root):].count(os.sep)
        indent = "    " * depth
        label = os.path.basename(dirpath) or dirpath
        print(f"{indent}{label}/")
        for fn in sorted(filenames):
            fp = os.path.join(dirpath, fn)
            try:
                size = human(os.path.getsize(fp))
            except OSError:
                size = "?"
            print(f"{indent}    {fn}  ({size})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
