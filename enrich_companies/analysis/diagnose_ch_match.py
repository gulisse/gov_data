#!/usr/bin/env python3
"""
diagnose_ch_match.py — calibration diagnostics for bt_stp1b_ch_match.py outputs.

Prints three views to help set --auto-cutoff, --fuzzy-cutoff and K:
  1) AUTO-ACCEPTS (match_basis=fuzzy_auto in ch_exact.csv) — the silent-risk
     bucket: score spread + a random sample of supplier → CH name to eyeball.
  2) FUZZY candidates (ch_candidates.csv) — score histogram + borderline sample,
     to decide whether --fuzzy-cutoff (82) is letting in noise.
  3) HOMONYMS at the K cap — suppliers whose true match may be beyond the K
     candidates shown to the model.

Usage:
    python diagnose_ch_match.py
    python diagnose_ch_match.py --exact ch_exact.csv --candidates ch_candidates.csv \
        --k 5 --sample 25
"""
import argparse
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 match calibration diagnostics")
    p.add_argument("--exact", default="ch_exact.csv")
    p.add_argument("--candidates", default="ch_candidates.csv")
    p.add_argument("--k", type=int, default=5, help="K used in the matcher (default 5)")
    p.add_argument("--sample", type=int, default=25, help="rows to print per sample")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def score_hist(scores, edges):
    """Left-closed integer buckets [lo, hi): unambiguous, no pandas.cut edge games."""
    s = pd.to_numeric(scores, errors="coerce").dropna().astype(int)
    total = max(len(s), 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        n = int(((s >= lo) & (s < hi)).sum())
        bar = "█" * int(round(n / total * 40))
        print(f"     {lo:>3}-{hi-1:<3} | {n:>8,} ({n/total*100:5.1f}%) {bar}")


def sample_pairs(df, n, seed):
    n = min(n, len(df))
    if n == 0:
        print("     (none)")
        return
    for _, r in df.sample(n, random_state=seed).iterrows():
        sup = str(r["supplier_clean"])[:38]
        chn = str(r["ch_company_name"])[:42]
        print(f"     {sup:38}  →  {chn:42} | {int(r['score'])}")


def main():
    a = parse_args()
    ex = pd.read_csv(a.exact, encoding="utf-8-sig", dtype=str)
    ca = pd.read_csv(a.candidates, encoding="utf-8-sig", dtype=str)
    for df in (ex, ca):
        df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # ── 1. auto-accepts ─────────────────────────────────
    print("=" * 62)
    print("1) ch_exact.csv — match_basis counts")
    print(ex["match_basis"].value_counts().to_string())
    auto = ex[ex["match_basis"] == "fuzzy_auto"]
    print(f"\n   AUTO-ACCEPTS (fuzzy_auto): {len(auto):,}  ← resolved with NO model check")
    if len(auto):
        print("   score spread:")
        score_hist(auto["score"], [95, 97, 99, 101])
        print(f"\n   random {min(a.sample, len(auto))} to eyeball "
              f"(supplier_clean → ch_company_name | score):")
        sample_pairs(auto, a.sample, a.seed)
        print("   → if any are wrong matches, raise --auto-cutoff (97/98).")

    # ── 2. fuzzy candidate quality ──────────────────────
    print("\n" + "=" * 62)
    print("2) ch_candidates.csv — match_basis counts")
    print(ca["match_basis"].value_counts().to_string())
    fz = ca[ca["match_basis"] == "fuzzy"]
    print(f"\n   FUZZY candidates: {len(fz):,}  — score histogram (drives --fuzzy-cutoff):")
    if len(fz):
        score_hist(fz["score"], [82, 85, 88, 90, 93, 95, 101])
        bl = fz[(fz["score"] >= 82) & (fz["score"] <= 85)]
        print(f"\n   borderline (82-85) sample — are these real or noise?")
        sample_pairs(bl, min(10, a.sample), a.seed)
        print("   → mostly noise ⇒ raise --fuzzy-cutoff; real near-matches ⇒ keep/lower.")

    # ── 3. homonyms at the K cap ────────────────────────
    print("\n" + "=" * 62)
    print(f"3) homonyms at the K cap (K={a.k})")
    hom = ca[ca["match_basis"] == "exact_homonym"]
    if len(hom):
        per = hom.groupby("supplier_clean").size()
        at_cap = int((per >= a.k).sum())
        print(f"   homonym suppliers      : {per.size:,}")
        print(f"   sitting at the K cap   : {at_cap:,}  "
              f"({at_cap/max(per.size,1)*100:.1f}%)")
        print("   → these may have additional same-name companies not shown to the")
        print("     model; consider a larger K for homonym cases if this is high.")
        print("\n   most-collided names (capped at K in the file):")
        for name, cnt in per.sort_values(ascending=False).head(10).items():
            print(f"     {str(name)[:48]:48} | {cnt} shown")
    else:
        print("   none")


if __name__ == "__main__":
    main()
