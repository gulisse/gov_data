#!/usr/bin/env python3
"""
locate_project_files.py  —  build a file-location manifest for the CH project.

WHY
    The git-linked Claude project deliberately excludes the large data files, and
    the folder tree was restructured. A fresh chat therefore cannot see where the
    big files live, nor be sure which copy of a duplicated script is current.
    This script runs LOCALLY (where every file exists), locates each known file
    across the tree, picks the newest copy as canonical, flags duplicates and
    missing files, and writes:

        file_manifest.json   machine-readable  {filename -> canonical path + meta}
        file_manifest.md     human/prompt-readable report

    Add the outputs to the project explorer (outside git) and point the project
    instructions at file_manifest.json so scripts reference the correct paths.

USAGE
    python locate_project_files.py                 # scan current dir
    python locate_project_files.py --root /path/to/repo_root
    python locate_project_files.py --root . --out-json file_manifest.json \
                                   --out-md file_manifest.md
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

# ── Directories never worth scanning ───────────────────────────────
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
             ".mypy_cache", ".pytest_cache", ".ipynb_checkpoints"}

# ── Files the scripts reference. role = why it matters; large_local = it is
#    excluded from git and only exists on the local machine. ──────────
#    Add or remove entries here as the pipeline grows.
TARGETS: dict[str, dict] = {
    # scripts (should be single-copy; duplicates are the known hazard)
    "bt_stp1b_ch_match.py":        {"role": "matcher (contains normalise())", "large_local": False},
    "check_labels_v2.py":          {"role": "v1->v2 scorer",                  "large_local": False},
    "test_normalise.py":           {"role": "regression harness",             "large_local": False},
    # ground-truth / inputs
    "ch_label_sample_v1.csv":      {"role": "hand labels — DO NOT overwrite",  "large_local": False},
    "distinct_suppliers_clean.csv":{"role": "supplier list fed to matcher",    "large_local": False},
    # large, git-excluded, local-only
    "ch_slim.parquet":             {"role": "slim CH cache (matcher reads)",   "large_local": True},
    "Companies_House_companies_list.csv": {"role": "full CH list (grep checks)","large_local": True},
    # v1 outputs (needed for the delta summary)
    "ch_exact.csv":                {"role": "v1 exact output (delta)",         "large_local": False},
    "ch_candidates.csv":           {"role": "v1 candidates output (delta)",    "large_local": False},
    # v2 outputs (regenerated each run)
    "ch_exact_v2.csv":             {"role": "v2 exact output",                 "large_local": False},
    "ch_candidates_v2.csv":        {"role": "v2 candidates output",            "large_local": False},
    # scoring output
    "label_v1_vs_v2_check.csv":    {"role": "final scoring output",            "large_local": False},
}


def has_bom(path: str) -> bool | None:
    """True if a file starts with the UTF-8 BOM (utf-8-sig). None on read error."""
    try:
        with open(path, "rb") as fh:
            return fh.read(3) == b"\xef\xbb\xbf"
    except OSError:
        return None


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def scan(root: str) -> dict[str, list[dict]]:
    """Return {basename -> [ {rel, abs, size, mtime}, ... ]} for target files only."""
    wanted = set(TARGETS)
    found: dict[str, list[dict]] = {name: [] for name in wanted}
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn in wanted:
                ap = os.path.join(dirpath, fn)
                try:
                    st = os.stat(ap)
                except OSError:
                    continue
                rel = os.path.relpath(ap, root).replace(os.sep, "/")
                found[fn].append({
                    "rel": rel,
                    "abs": ap,
                    "size": st.st_size,
                    "mtime": st.st_mtime,
                })
    return found


def build_manifest(root: str, found: dict[str, list[dict]]) -> dict:
    root = os.path.abspath(root)
    files: dict[str, dict] = {}
    path_map: dict[str, str] = {}
    warnings: list[str] = []

    for name, meta in TARGETS.items():
        hits = sorted(found[name], key=lambda h: h["mtime"], reverse=True)
        entry: dict = {"role": meta["role"], "large_local": meta["large_local"]}

        if not hits:
            entry["status"] = "MISSING"
            entry["canonical_path"] = None
            entry["all_paths"] = []
            files[name] = entry
            sev = "expected local-only file" if meta["large_local"] else "referenced by scripts"
            warnings.append(f"MISSING: {name} ({sev}) — not found under root.")
            continue

        canonical = hits[0]
        entry["status"] = "DUPLICATE" if len(hits) > 1 else "OK"
        entry["canonical_path"] = canonical["rel"]
        entry["canonical_abs"] = canonical["abs"]
        entry["size"] = canonical["size"]
        entry["size_h"] = human_size(canonical["size"])
        entry["modified"] = datetime.fromtimestamp(
            canonical["mtime"], tz=timezone.utc).isoformat(timespec="seconds")
        entry["all_paths"] = [h["rel"] for h in hits]

        if name.lower().endswith(".csv"):
            entry["has_bom"] = has_bom(canonical["abs"])
            if entry["has_bom"] is False:
                warnings.append(
                    f"ENCODING: {name} at {canonical['rel']} has no UTF-8 BOM "
                    f"(project requires utf-8-sig on CSV outputs).")

        if len(hits) > 1:
            others = "; ".join(
                f"{h['rel']} ({human_size(h['size'])}, "
                f"{datetime.fromtimestamp(h['mtime'], tz=timezone.utc):%Y-%m-%d %H:%M})"
                for h in hits)
            warnings.append(
                f"DUPLICATE: {name} found in {len(hits)} places → canonical (newest) = "
                f"{canonical['rel']}. All: {others}")

        files[name] = entry
        path_map[name] = canonical["rel"]

    return {
        "generated": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "root": root,
        "path_map": path_map,          # {filename -> canonical relative path}
        "files": files,                # {filename -> full metadata}
        "warnings": warnings,
    }


def write_markdown(manifest: dict, out_md: str) -> None:
    m = manifest
    lines: list[str] = []
    lines.append("# Project File Manifest")
    lines.append("")
    lines.append(f"- Generated (UTC): {m['generated']}")
    lines.append(f"- Scan root: `{m['root']}`")
    lines.append(f"- Paths below are **relative to the scan root** unless noted.")
    lines.append("")

    warns = m["warnings"]
    lines.append(f"## ⚠️ Warnings ({len(warns)})")
    if warns:
        for w in warns:
            lines.append(f"- {w}")
    else:
        lines.append("- None. All targets found, single-copy, CSVs BOM-clean.")
    lines.append("")

    lines.append("## Canonical path map")
    lines.append("")
    lines.append("| File | Status | Canonical path | Size | Modified (UTC) | BOM |")
    lines.append("|---|---|---|---|---|---|")
    for name, e in m["files"].items():
        if e["status"] == "MISSING":
            lines.append(f"| `{name}` | **MISSING** | — | — | — | — |")
            continue
        bom = e.get("has_bom")
        bom_s = "—" if bom is None else ("yes" if bom else "**no**")
        star = " ⚠️dup" if e["status"] == "DUPLICATE" else ""
        lines.append(
            f"| `{name}` | {e['status']}{star} | `{e['canonical_path']}` | "
            f"{e.get('size_h','—')} | {e.get('modified','—')[:16].replace('T',' ')} | {bom_s} |")
    lines.append("")

    dups = {n: e for n, e in m["files"].items()
            if e["status"] == "DUPLICATE"}
    if dups:
        lines.append("## Duplicate details (resolve these)")
        lines.append("")
        for name, e in dups.items():
            lines.append(f"- `{name}` — canonical (newest) `{e['canonical_path']}`; "
                         f"other copies: " + ", ".join(f"`{p}`" for p in e["all_paths"][1:]))
        lines.append("")

    lines.append("## Local-only large files (git-excluded)")
    lines.append("")
    large = {n: e for n, e in m["files"].items() if e["large_local"]}
    for name, e in large.items():
        if e["status"] == "MISSING":
            lines.append(f"- `{name}` — **MISSING locally**; needed by the matcher/checks.")
        else:
            lines.append(f"- `{name}` — `{e['canonical_path']}` "
                         f"(abs: `{e.get('canonical_abs','?')}`, {e.get('size_h','?')})")
    lines.append("")

    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Locate project files and emit a manifest.")
    ap.add_argument("--root", default=".", help="Directory to scan (default: cwd).")
    ap.add_argument("--out-json", default="file_manifest.json")
    ap.add_argument("--out-md", default="file_manifest.md")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"ERROR: root not a directory: {args.root}")
        return 2

    found = scan(args.root)
    manifest = build_manifest(args.root, found)

    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    write_markdown(manifest, args.out_md)

    # console summary
    n_missing = sum(1 for e in manifest["files"].values() if e["status"] == "MISSING")
    n_dup = sum(1 for e in manifest["files"].values() if e["status"] == "DUPLICATE")
    n_ok = sum(1 for e in manifest["files"].values() if e["status"] == "OK")
    print(f"Scanned: {manifest['root']}")
    print(f"  OK: {n_ok}   DUPLICATE: {n_dup}   MISSING: {n_missing}")
    for w in manifest["warnings"]:
        print("  ! " + w)
    print(f"Wrote {args.out_json} and {args.out_md}")
    # non-zero exit if anything needs attention, handy in a preflight
    return 1 if (n_missing or n_dup) else 0


if __name__ == "__main__":
    raise SystemExit(main())
