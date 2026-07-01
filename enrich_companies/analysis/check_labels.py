#!/usr/bin/env python3
"""
check_labels_v2.py — score your existing hand-labels against the v2 matcher outputs.

For every row you marked N (and the rest, for context), pull the CORRECT Companies
House number you wrote in `note`, then check whether the v2 matcher now lands it:
  • in ch_exact_v2.csv            → resolved deterministically  ("fixed_exact")
  • as the TOP candidate          → model will almost certainly pick it ("fixed_top_candidate")
  • somewhere in its candidates   → recoverable, just not first  ("in_candidates_not_top")
  • nowhere                       → still a miss                  ("still_absent")

This reuses your day of labelling to SCORE v2 instead of re-labelling. The correct
number is read from `note` (you wrote it there); the probe's own pick in the label
sheet's ch_company_number column was the WRONG one for N rows, so it is ignored.

Usage
─────
    python check_labels_v2.py
    python check_labels_v2.py --labels ch_label_sample.csv \
        --exact ch_exact_v2.csv --candidates ch_candidates_v2.csv \
        --out label_v2_check.csv
    # optional v1 baseline to print a before/after delta:
    python check_labels_v2.py --v1-exact ch_exact.csv --v1-candidates ch_candidates.csv
"""

import argparse
import re
import sys

import pandas as pd

NUM_COL = "ch_company_number"
# canonical CH number forms: 2-letter+6-digit (SC/NI/OC/OE/…) or 8 digits
_CANON_RE = re.compile(r"\b([A-Z]{2}\d{6}|\d{8})\b")
# fallback: a 6-7 digit run (number written without leading zeros), used only if no canonical
_LOOSE_RE = re.compile(r"\b(\d{6,7})\b")


def norm_num(n):
    """Normalise a company number for comparison: upper, 8-char zero-pad if pure digits."""
    s = str(n).strip().upper()
    if s.isdigit():
        return s.zfill(8)
    return s


def numbers_from_note(note):
    """All plausible CH numbers mentioned in a note, normalised."""
    if not isinstance(note, str) or not note.strip():
        return set()
    found = set(_CANON_RE.findall(note))
    if not found:
        found = set(_LOOSE_RE.findall(note))
    return {norm_num(x) for x in found}


def per_supplier(df):
    """exact: supplier -> normalised number; candidates: supplier -> (top_num, set(all_nums))
    Top = first row for that supplier in file order (the matcher's ranked order)."""
    exact_map, top_map, all_map = {}, {}, {}
    if df is None or not len(df):
        return exact_map, top_map, all_map
    for sup, grp in df.groupby("supplier_clean", sort=False):
        nums = [norm_num(x) for x in grp[NUM_COL].tolist() if str(x).strip()]
        if nums:
            top_map[sup] = nums[0]
            all_map[sup] = set(nums)
    return top_map, all_map


def load(path, what):
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    except FileNotFoundError:
        sys.exit(f"{what} not found: {path}")


def verdict_for(sup, correct_nums, exact_top, cand_top, cand_all):
    """Return (verdict, detail) for one supplier given its correct number set."""
    if not correct_nums:
        return "no_reference", ""
    if sup in exact_top and exact_top[sup] in correct_nums:
        return "fixed_exact", exact_top[sup]
    if sup in cand_top and cand_top[sup] in correct_nums:
        return "fixed_top_candidate", cand_top[sup]
    if sup in cand_all and (correct_nums & cand_all[sup]):
        return "in_candidates_not_top", next(iter(correct_nums & cand_all[sup]))
    return "still_absent", ""


def build_maps(exact, candidates):
    exact_top, _ = per_supplier(exact)          # exact has one row/supplier → top == only
    cand_top, cand_all = per_supplier(candidates)
    return exact_top, cand_top, cand_all


def score(labels, exact_top, cand_top, cand_all):
    rows = []
    for _, r in labels.iterrows():
        sup = r["supplier_clean"]
        correct = numbers_from_note(r.get("note", ""))
        v, detail = verdict_for(sup, correct, exact_top, cand_top, cand_all)
        rows.append({
            "supplier_clean": sup,
            "label": str(r.get("label", "")).strip(),
            "probe_scope": r.get("probe_scope", ""),
            "match_tier": r.get("match_tier", ""),
            "correct_number": "|".join(sorted(correct)) if correct else "",
            "data_gap_note": "missing in download" in str(r.get("note", "")).lower()
                             or "exists online" in str(r.get("note", "")).lower(),
            "verdict": v,
            "matched_number": detail,
            "note": r.get("note", ""),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Score existing labels against v2 outputs")
    ap.add_argument("--labels", default="ch_label_sample.csv")
    ap.add_argument("--exact", default="ch_exact_v2.csv")
    ap.add_argument("--candidates", default="ch_candidates_v2.csv")
    ap.add_argument("--out", default="label_v2_check.csv")
    ap.add_argument("--v1-exact", default=None, help="optional: v1 ch_exact.csv for a delta")
    ap.add_argument("--v1-candidates", default=None, help="optional: v1 ch_candidates.csv")
    args = ap.parse_args()

    labels = load(args.labels, "labels")
    exact = load(args.exact, "exact (v2)")
    candidates = load(args.candidates, "candidates (v2)")

    exact_top, cand_top, cand_all = build_maps(exact, candidates)
    res = score(labels, exact_top, cand_top, cand_all)
    res.to_csv(args.out, index=False, encoding="utf-8-sig")

    FIXED = {"fixed_exact", "fixed_top_candidate"}

    def block(df, title):
        if not len(df):
            return
        vc = df["verdict"].value_counts()
        fixed = int(df["verdict"].isin(FIXED).sum())
        ref = int((df["verdict"] != "no_reference").sum())
        print(f"\n{title}  (n={len(df)}, with a correct-number reference={ref})")
        for v in ["fixed_exact", "fixed_top_candidate", "in_candidates_not_top",
                  "still_absent", "no_reference"]:
            if v in vc:
                print(f"    {v:24}: {vc[v]:>3}")
        if ref:
            print(f"    → resolved (exact or top cand): {fixed}/{ref} = {fixed/ref*100:.0f}%")

    # headline: the N rows (what you asked about)
    N = res[res["label"] == "N"]
    block(N, "N rows (labelled wrong by the probe — did v2 fix them?)")

    # split the N rows by whether they were a dissolved/data-gap note
    gap = N[N["data_gap_note"]]
    rec = N[~N["data_gap_note"]]
    block(rec, "  ↳ N excluding data-gap notes (matcher-recoverable)")
    block(gap, "  ↳ N that were 'missing in download' (now back via dissolved merge?)")

    # context: did any Y rows regress? Use the probe's CONFIRMED-correct pick (the label
    # sheet's own ch_company_number), which for Y rows you verified as right.
    Yrows = labels[labels["label"].astype(str).str.strip() == "Y"]
    checked = broke = 0
    broke_list = []
    for _, r in Yrows.iterrows():
        sup = r["supplier_clean"]
        pick = norm_num(r.get(NUM_COL, ""))
        if not pick:
            continue
        checked += 1
        present = (sup in exact_top and exact_top[sup] == pick) or \
                  (sup in cand_all and pick in cand_all[sup])
        if not present:
            broke += 1
            broke_list.append((sup, pick))
    print(f"\nY rows: {len(Yrows)} | re-checkable (had a confirmed pick): {checked} | "
          f"now absent in v2 (possible regression): {broke}")
    for sup, pick in broke_list:
        print(f"    ! {sup}  (was {pick})")

    print(f"\nPer-row detail → {args.out}")

    # optional v1 baseline delta
    if args.v1_exact and args.v1_candidates:
        v1e = load(args.v1_exact, "exact (v1)")
        v1c = load(args.v1_candidates, "candidates (v1)")
        e1, t1, a1 = build_maps(v1e, v1c)
        res_v1 = score(labels, e1, t1, a1)
        f1 = int(res_v1[res_v1["label"] == "N"]["verdict"].isin(FIXED).sum())
        f2 = int(res[res["label"] == "N"]["verdict"].isin(FIXED).sum())
        print("\n── v1 → v2 delta on N rows (resolved = exact or top candidate) ──")
        print(f"    v1: {f1}    v2: {f2}    Δ +{f2 - f1}")


if __name__ == "__main__":
    main()
