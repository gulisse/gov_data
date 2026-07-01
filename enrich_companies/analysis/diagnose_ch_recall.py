#!/usr/bin/env python3
"""
diagnose_ch_recall.py — recall / missed-match diagnostics for bt_stp1b_ch_match.py.

WHY THIS EXISTS
───────────────
diagnose_ch_match.py answers "is my CUTOFF letting in noise?" (score histograms,
auto-accept eyeball, homonym K-cap). It cannot see what the matcher FAILED to find.
This script answers the opposite question — "what real matches are we MISSING, and
which name-transformation rules would recover them?" — without any ground-truth labels
and without re-reading the 2.75 GB CH CSV.

It implements the first two layers of the agreed plan:
  (1) STRATIFY the residual by spend  — emit the no-match list (which the matcher only
      counts, never writes), join supplier spend, flag the >£50k "worth-fixing" cohort.
  (2) RELAXED PROBE + DIFF TABLES — over the misses, run a deliberately over-generous
      matcher (distinctive-token blocking instead of first-token-only; token_set_ratio
      instead of WRatio; no production cutoff) to dredge up the CH record we PROBABLY
      missed, then for every pair record the transformation diff (which tokens differ,
      which known rule would bridge them). Aggregating those diffs across thousands of
      pairs is what turns five hand-picked examples into sized, named rule families.

The (supplier -> relaxed candidate) pairs are PLAUSIBLE, not confirmed — confirming them
is the later "oracle" step (manual stratified sample or LLM-judge). ch_probe_pairs.csv is
the exact feedstock for that step.

TWO SCOPES (run one after the other; tagged in `probe_scope`):
  • no_match   — suppliers the matcher gave up on entirely (Pertemps-type).
  • candidates — suppliers that DID get candidates, re-probed to catch "offered five,
                 right one missing" (British-Telecom-on-the-crowded-'british'-block type).
                 For these, `new_vs_production` flags a relaxed hit the matcher never showed.

INPUTS (all already produced by your calibrated run — no 5.7M re-read):
  • ch_slim.parquet                         (CH cache: norm_name/block_key + metadata)
  • ch_token_index.npz                       (token→positions index; build ONCE via
                                              build_ch_token_index.py — the streamed,
                                              memory-lean step A. NOT rebuilt here.)
  • ch_exact.csv, ch_candidates.csv         (what the matcher resolved / offered)
  • supplier_context.csv  (--suppliers)     (universe; mirror of the matcher's --suppliers)
  • distinct_suppliers_review_updated.csv   (--review: type=Business filter + spend)

Normalisation is IMPORTED from bt_stp1b_ch_match so supplier-side cleaning is byte-identical
to the CH-side cleaning baked into the parquet.

OUTPUTS (all utf-8-sig):
  • ch_no_match.csv      — no-match suppliers + debug + spend + business_type.
  • ch_segmented_out.csv — suppliers excluded as non-company (schools/churches/public bodies).
  • ch_probe_pairs.csv   — every supplier->relaxed-candidate pair + scores + transformation
                           diff + match_tier (clean/head_modifier/loose/weak) + rule flags.
  • ch_label_sample.csv  — stratified sample (scope × tier) with empty `label`/`note` columns
                           for human true/false marking; this calibrates the gate.
  • ch_probe_report.txt  — per scope: tier counts, plausible (clean+head) rule-family +
                           token tables, and spend-stratified samples per tier.

CONFIDENCE TIERS (the gate that replaced the leaky subset match):
  clean         — distinctive content matches exactly; extras all generic/legal/numeric
  head_modifier — supplier fully explained AND shares the leading distinctive token; CH adds
                  distinctive modifiers after it (Pertemps→PERTEMPS NETWORK GROUP)
  loose         — a distinctive token matches but each side keeps unexplained distinctive
                  tokens (the Mace→ALIBAY-MACE false-positive class)
  weak          — no distinctive shared content
"plausible" = clean + head_modifier. Non-company entities are segmented out before probing.

USAGE
─────
    python diagnose_ch_recall.py                       # both scopes, all misses
    python diagnose_ch_recall.py --min-amount 50000    # money-first: only >£50k misses
    python diagnose_ch_recall.py --scope no_match --sample 3000
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))

# Identical cleaning to production — import, do not re-implement.
from bt_stp1b_ch_match import normalise, initials_sig, block_key, load_ch, is_active
from ch_token_index import TokenIndex


# ──────────────────────────────────────────────────────────────────────
# Rule-family vocabulary (drives the diff categorisation)
# ──────────────────────────────────────────────────────────────────────
# Generic structural / location / descriptor words that are often noise when they
# appear on only one side. Kept to a DEFENSIBLE CORE — the extra-token frequency tables
# surface every other one-sided word empirically, so this list only steers the
# `trailing_generic` flag, not what gets discovered.
GENERIC_WORDS = {
    "estate", "estates", "of", "companies", "company", "services", "service",
    "trading", "international", "uk", "gb", "the", "holdings", "group", "groups",
    "solutions", "partnership", "partnerships", "for", "to", "and",
}

# Legal-form abbreviation -> the words its EXPANSION leaves behind AFTER production
# normalise() (which already drops limited/company/incorporated/etc.). e.g. for a
# supplier ending 'CIC' (stripped), the CH "COMMUNITY INTEREST COMPANY" leaves the
# residue {community, interest}. Detecting the residue on the CH-only side both
# explains the miss and names the fix (collapse the phrase, or extend the suffix set).
EXPANSION_RESIDUES = {
    "plc": {"public"},                                  # public [limited company]
    "cic": {"community", "interest"},                   # community interest [company]
    "cio": {"charitable", "organisation"},              # charitable [incorporated] organisation
    "llp": {"liability", "partnership"},                # [limited] liability partnership
    "llc": {"liability"},                               # [limited] liability [company]
}
# flat set of all residue tokens, for quick membership tests
RESIDUE_TOKENS = set().union(*EXPANSION_RESIDUES.values())


# ──────────────────────────────────────────────────────────────────────
# Entity segmentation uses the AUTHORITATIVE business_type classification from the
# review file (joined on supplier_clean), not name keywords. Categories that are
# predominantly NOT registered limited companies are excluded from probing by default
# so we don't force CH matches onto them. This is a judgement call per category and is
# fully configurable via --exclude-business-types; the report prints the full breakdown
# of what was excluded vs kept so nothing is hidden.
# Caveat: 'School / education' includes academies/MATs (which ARE on CH) and
# 'Charity / voluntary' includes charitable companies/CIOs (also on CH) — hence Charity
# is NOT excluded by default. Adjust the list once labels tell us which truly never match.
# ──────────────────────────────────────────────────────────────────────
DEFAULT_EXCLUDE_BTYPES = ["School / education", "Nursery / childcare", "Religious organisation"]


# ──────────────────────────────────────────────────────────────────────
# Coverage / match-tier gate. The old subset gate let token_set=100 coincidences
# through (Mace→ALIBAY-MACE). The fix: judge whether the DISTINCTIVE content matches,
# stem-aware, and bucket by how cleanly:
#   clean         — distinctive content matches exactly; all extras are generic/legal/numeric
#   head_modifier — supplier fully explained AND shares the leading distinctive token,
#                   CH only adds distinctive modifiers AFTER it (Pertemps→PERTEMPS NETWORK GROUP)
#   loose         — a distinctive token matches but each side keeps unexplained distinctive
#                   tokens (the Mace→ALIBAY-MACE risk class)
#   weak          — no distinctive shared content at all
# clean + head_modifier = the 'plausible' set; loose/weak are bucketed for inspection.
# ──────────────────────────────────────────────────────────────────────
def _stem(t):
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("es") and len(t) > 4:
        return t[:-2]
    if t.endswith("s") and len(t) > 3:
        return t[:-1]
    return t


def _distinctive(t):
    return (t not in GENERIC_WORDS and t not in RESIDUE_TOKENS
            and not t.isdigit() and len(t) > 1)


def match_tier(stoks, ctoks, anchor_df, anchor_max_df):
    s_d = {_stem(t) for t in stoks if _distinctive(t)}
    c_d = {_stem(t) for t in ctoks if _distinctive(t)}
    shared_d = s_d & c_d
    resid_s = s_d - c_d
    resid_c = c_d - s_d
    if not shared_d:
        return "weak"
    # a shared distinctive token that is still very common is weak evidence
    if anchor_df > anchor_max_df:
        return "loose"
    if not resid_s and not resid_c:
        return "clean"
    head_ok = (stoks and ctoks and _stem(stoks[0]) == _stem(ctoks[0])
               and _stem(stoks[0]) in shared_d and not resid_s)
    if head_ok:
        return "head_modifier"
    return "loose"


def parse_args():
    p = argparse.ArgumentParser(description="Recall / missed-match diagnostics for the CH matcher")
    p.add_argument("--ch-parquet", default="ch_slim.parquet")
    p.add_argument("--token-index", default="ch_token_index.npz",
                   help="Persisted token index from build_ch_token_index.py (step A)")
    p.add_argument("--ch", default="Companies_House_companies_list.csv",
                   help="Only used to (re)build the parquet if it is missing")
    p.add_argument("--exact", default="ch_exact.csv")
    p.add_argument("--candidates", default="ch_candidates.csv")
    p.add_argument("--suppliers", default="supplier_context.csv",
                   help="Same --suppliers file the matcher used (universe)")
    p.add_argument("--review", default="distinct_suppliers_review_updated.csv",
                   help="Same --review file the matcher used (Business filter + spend)")
    p.add_argument("--scope", choices=["no_match", "candidates", "both"], default="both")
    p.add_argument("--min-amount", type=float, default=0.0,
                   help="Only probe misses with spend >= this (0 = all). Try 50000 first.")
    p.add_argument("--sample", type=int, default=0,
                   help="Cap suppliers probed per scope (0 = all); takes the highest-spend first")
    p.add_argument("--top-n", type=int, default=5, help="Relaxed candidates kept per supplier")
    p.add_argument("--relaxed-floor", type=int, default=60,
                   help="Min token_set_ratio to keep a relaxed candidate (kept low on purpose)")
    p.add_argument("--good-floor", type=int, default=90,
                   help="token_set_ratio at/above which a pair is counted 'likely real' in the report")
    p.add_argument("--anchor-max-df", type=int, default=25_000,
                   help="A shared distinctive token appearing in more than this many CH rows "
                        "is treated as weak evidence (pair downgraded to 'loose').")
    p.add_argument("--exclude-business-types", default=",".join(DEFAULT_EXCLUDE_BTYPES),
                   help="Comma-separated business_type values (from the review file) to segment "
                        "out before probing, since they're predominantly not on Companies House. "
                        "Pass '' to probe everything. Default: "
                        + " | ".join(DEFAULT_EXCLUDE_BTYPES))
    p.add_argument("--block-tokens", type=int, default=3,
                   help="How many of a supplier's rarest tokens to pool candidates from")
    p.add_argument("--max-df", type=int, default=150_000,
                   help="Skip a token as a blocking key if it appears in more CH rows than this")
    p.add_argument("--max-pool", type=int, default=4000,
                   help="Cap candidate pool size per supplier (rarest tokens first)")
    p.add_argument("--threshold-amount", type=float, default=50000.0,
                   help="Spend cohort highlighted in the report (default £50k)")
    p.add_argument("--label-sample", type=int, default=180,
                   help="Rows to write to the stratified labelling sheet (0 = skip)")
    p.add_argument("--label-out", default="ch_label_sample.csv")
    p.add_argument("--out-no-match", default="ch_no_match.csv")
    p.add_argument("--out-segmented", default="ch_segmented_out.csv")
    p.add_argument("--out-pairs", default="ch_probe_pairs.csv")
    p.add_argument("--report", default="ch_probe_report.txt")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Universe + spend (mirror the matcher's --suppliers ∩ --review Business)
# ──────────────────────────────────────────────────────────────────────
def load_universe_and_spend(suppliers_path, review_path):
    sup = pd.read_csv(suppliers_path, dtype=str, encoding="utf-8-sig", low_memory=False)
    if "supplier_clean" not in sup.columns:
        sys.exit(f"{suppliers_path} needs a supplier_clean column")

    spend = {}            # supplier_clean -> amount (float)
    cumpct = {}           # supplier_clean -> cum_amount_pct
    btype = {}            # supplier_clean -> business_type (authoritative classification)
    biz = None
    if review_path and os.path.exists(review_path):
        rv = pd.read_csv(review_path, dtype=str, encoding="utf-8-sig", low_memory=False)
        if "type" in rv.columns:
            biz = set(rv.loc[rv["type"].fillna("").str.strip() == "Business", "supplier_clean"])
        if "amount" in rv.columns:
            # IMPORTANT: this column carries thousands-separators ("1,846,000,076.76");
            # strip them or to_numeric silently keeps only the sub-£1,000 values.
            amt = pd.to_numeric(rv["amount"].str.replace(",", "", regex=False), errors="coerce")
            spend = dict(zip(rv["supplier_clean"], amt))
        if "cum_amount_pct" in rv.columns:
            cumpct = dict(zip(rv["supplier_clean"],
                              pd.to_numeric(rv["cum_amount_pct"], errors="coerce")))
        if "business_type" in rv.columns:
            btype = dict(zip(rv["supplier_clean"], rv["business_type"].fillna("(blank)")))
    else:
        print(f"  (no review file at {review_path} — no Business filter, no spend)")

    if biz is not None:
        before = len(sup)
        sup = sup[sup["supplier_clean"].isin(biz)]
        print(f"  universe filtered to type=Business: {len(sup):,} of {before:,}")
    universe = sorted(set(sup["supplier_clean"].dropna()))
    return universe, spend, cumpct, btype


# ──────────────────────────────────────────────────────────────────────
# Inverted token index over ALL CH rows (so blocking isn't first-token-only)
# ──────────────────────────────────────────────────────────────────────
def candidate_pool(stoks, idx, block_tokens, max_df, max_pool):
    """Pool CH row positions sharing the supplier's rarest tokens (rarest first, capped).

    `idx` is a ch_token_index.TokenIndex (loaded from the persisted cache)."""
    present = [(t, idx.df(t)) for t in dict.fromkeys(stoks) if t in idx]   # dedup, keep order
    if not present:
        return np.empty(0, dtype=np.int64)
    present.sort(key=lambda x: x[1])                                  # rarest df first
    usable = [t for t, d in present if d <= max_df] or [present[0][0]]  # fallback: rarest
    pool = []
    seen = 0
    for t in usable[:block_tokens]:
        arr = idx.postings(t)
        pool.append(arr)
        seen += len(arr)
        if seen >= max_pool:
            break
    union = np.unique(np.concatenate(pool)) if pool else np.empty(0, dtype=np.int64)
    if len(union) > max_pool:
        union = union[:max_pool]            # arbitrary but bounded; rare in practice
    return union


# ──────────────────────────────────────────────────────────────────────
# Transformation diff + rule-family categorisation for one (supplier, CH) pair
# ──────────────────────────────────────────────────────────────────────
def diff_and_categorise(stoks, ctoks):
    sset, cset = set(stoks), set(ctoks)
    shared = sset & cset
    extra_ch = [t for t in ctoks if t not in sset]      # words CH has, supplier lacks
    extra_sup = [t for t in stoks if t not in cset]     # words supplier has, CH lacks
    flags = {
        "cat_block_miss": False, "cat_abbrev_expansion": False,
        "cat_trailing_generic": False, "cat_plural_stem": False,
        "cat_prefix_abbrev": False, "cat_extra_token_only": False,
    }

    # block_miss: production blocks on the FIRST token; if first tokens differ, production
    # fuzzy would never even have compared these two names.
    if stoks and ctoks and stoks[0] != ctoks[0]:
        flags["cat_block_miss"] = True

    # abbrev/expansion: the CH-only side carries a known legal-form expansion residue
    # while the supplier side carried the abbreviation (now stripped by normalise()).
    if any(t in RESIDUE_TOKENS for t in extra_ch):
        flags["cat_abbrev_expansion"] = True

    # trailing_generic: a one-sided extra token that is a generic word AND sits at the
    # end of its name (the "...Estate", "...of Companies", "...Services" pattern).
    def trailing_generic(extra, toks):
        tail = set(toks[-2:])                            # last two positions
        return any(t in GENERIC_WORDS and t in tail for t in extra)
    if trailing_generic(extra_sup, stoks) or trailing_generic(extra_ch, ctoks):
        flags["cat_trailing_generic"] = True

    # plural/stem: an unmatched supplier token and unmatched CH token share a stem
    # (energies/energy) — symmetric singular/plural & light morphology.
    if extra_sup and extra_ch:
        stems_c = {_stem(t) for t in extra_ch}
        if any(_stem(t) in stems_c for t in extra_sup):
            flags["cat_plural_stem"] = True
        # prefix_abbrev: one unmatched token is a prefix of the other (telecom/telecommunications)
        for a in extra_sup:
            for b in extra_ch:
                lo, hi = (a, b) if len(a) <= len(b) else (b, a)
                if len(lo) >= 4 and hi.startswith(lo) and lo != hi:
                    flags["cat_prefix_abbrev"] = True

    # extra_token_only: names are equal once one side's extra tokens are removed, and
    # those extras aren't explained above — i.e. a pure subset/superset of tokens.
    if shared and (not extra_sup or not extra_ch) and not any(
        flags[k] for k in ("cat_abbrev_expansion", "cat_trailing_generic")
    ):
        flags["cat_extra_token_only"] = True

    # one headline label (priority order) for the summary table
    priority = ["cat_abbrev_expansion", "cat_plural_stem", "cat_prefix_abbrev",
                "cat_trailing_generic", "cat_extra_token_only", "cat_block_miss"]
    primary = next((k[4:] for k in priority if flags[k]), "other")
    return shared, extra_ch, extra_sup, flags, primary


# ──────────────────────────────────────────────────────────────────────
# Probe one supplier
# ──────────────────────────────────────────────────────────────────────
def probe_supplier(sname, cols, idx, prod_numbers, btype, anchor_max_df,
                   top_n, relaxed_floor, block_tokens, max_df, max_pool):
    skey, stoks = normalise(sname if isinstance(sname, str) else "")
    if not stoks:
        return []
    pool = candidate_pool(stoks, idx, block_tokens, max_df, max_pool)
    if len(pool) == 0:
        return []
    choices = {int(p): cols["norms"][p] for p in pool}
    scored = process.extract(skey, choices, scorer=fuzz.token_set_ratio,
                             limit=top_n, score_cutoff=relaxed_floor)
    out = []
    for rank, (cnorm, ts, pos) in enumerate(scored, 1):
        ctoks = cnorm.split()
        shared, extra_ch, extra_sup, flags, primary = diff_and_categorise(stoks, ctoks)
        # rarest shared token = the strength of the anchor linking the two names.
        # A pair pooled on a rare token (e.g. 'pertemps') is strong evidence; a pair
        # sharing only a common word ('management') is a coincidence token_set inflates.
        if shared:
            anchor = min(shared, key=lambda t: idx.df(t))
            anchor_df = idx.df(anchor)
        else:
            anchor, anchor_df = "", 0
        tier = match_tier(stoks, ctoks, anchor_df, anchor_max_df)
        row = {
            "supplier_clean": sname,
            "supplier_norm": skey,
            "supplier_tokens_n": len(stoks),
            "business_type": btype,
            "match_tier": tier,
            "rank": rank,
            "ch_company_number": cols["nums"][pos],
            "ch_company_name": cols["names"][pos],
            "ch_norm": cnorm,
            "ch_status": cols["status"][pos],
            "ch_town": cols["towns"][pos],
            "sic_raw": cols["sics"][pos],
            "score_token_set": int(ts),
            "score_token_sort": int(fuzz.token_sort_ratio(skey, cnorm)),
            "score_partial": int(fuzz.partial_ratio(skey, cnorm)),
            "score_wratio": int(fuzz.WRatio(skey, cnorm)),   # the PRODUCTION scorer
            "ch_active": is_active(cols["status"][pos]),
            "anchor_token": anchor,
            "anchor_df": anchor_df,
            "shared_tokens": " ".join(sorted(shared)),
            "extra_ch_tokens": " ".join(extra_ch),
            "extra_sup_tokens": " ".join(extra_sup),
            "primary_category": primary,
            "new_vs_production": cols["nums"][pos] not in prod_numbers,
            **flags,
        }
        out.append(row)
    return out


# ──────────────────────────────────────────────────────────────────────
# Reporting helpers
# ──────────────────────────────────────────────────────────────────────
def fmt_money(x):
    return f"£{x:,.0f}" if pd.notna(x) else "£?"


def token_freq_table(pairs, col, spend, top=30):
    """Frequency of one-sided extra tokens across the BEST pair per supplier, spend-weighted."""
    from collections import Counter
    cnt, money = Counter(), Counter()
    for r in pairs:
        amt = spend.get(r["supplier_clean"], np.nan)
        for t in str(r[col]).split():
            cnt[t] += 1
            if pd.notna(amt):
                money[t] += amt
    lines = []
    for t, n in cnt.most_common(top):
        lines.append(f"     {t:<22} {n:>7,}   {fmt_money(money[t]):>16}")
    return lines or ["     (none)"]


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    a = parse_args()
    rng = np.random.default_rng(a.seed)

    # CH cache
    if not os.path.exists(a.ch_parquet):
        from bt_stp1b_ch_match import build_ch_parquet
        if not os.path.exists(a.ch):
            sys.exit(f"No parquet ({a.ch_parquet}) and no CH csv ({a.ch}) to build it from")
        build_ch_parquet(a.ch, a.ch_parquet, print)

    # Token index is built once by build_ch_token_index.py (step A) and cached. We do NOT
    # rebuild it here — that streaming build is what keeps a 10 GB box from OOMing.
    if not os.path.exists(a.token_index):
        sys.exit(f"Token index '{a.token_index}' not found.\n"
                 f"Build it once first:  python build_ch_token_index.py "
                 f"--ch-parquet {a.ch_parquet} --out {a.token_index}")
    print(f"Loading token index: {a.token_index}")
    idx = TokenIndex.load(a.token_index)
    print(f"  {len(idx):,} tokens over {idx.n_rows:,} CH rows")

    print(f"Loading CH metadata: {a.ch_parquet}")
    ch = load_ch(a.ch_parquet)
    if len(ch) != idx.n_rows:
        sys.exit(f"Row-count mismatch: parquet {len(ch):,} vs index {idx.n_rows:,}. "
                 f"Rebuild the index from this parquet (build_ch_token_index.py).")
    # Keep columns as numpy arrays indexed by CH row position; positional metadata is only
    # needed for the handful of candidates we keep, but norm_name is the scoring hot path.
    cols = {
        "names": ch["CompanyName"].to_numpy(),
        "norms": ch["norm_name"].to_numpy(),
        "nums": ch["CompanyNumber"].to_numpy(),
        "status": ch["CompanyStatus"].astype(str).to_numpy(),
        "towns": ch["RegAddress.PostTown"].astype(str).to_numpy(),
        "sics": ch["sic_raw"].to_numpy(),
    }
    del ch                       # free the DataFrame; the numpy arrays share the strings

    # universe + matcher outputs
    universe, spend, cumpct, btype = load_universe_and_spend(a.suppliers, a.review)
    ex = pd.read_csv(a.exact, dtype=str, encoding="utf-8-sig")
    ca = pd.read_csv(a.candidates, dtype=str, encoding="utf-8-sig")
    resolved = set(ex["supplier_clean"])
    cand_suppliers = set(ca["supplier_clean"])
    # production candidate numbers per supplier (for new_vs_production on the re-probe)
    prod_nums = ca.groupby("supplier_clean")["ch_company_number"].apply(set).to_dict()
    no_match = [s for s in universe if s not in resolved and s not in cand_suppliers]

    uni_amt = sum(v for v in (spend.get(s) for s in universe) if pd.notna(v))
    print(f"\nUniverse {len(universe):,} | resolved {len(resolved):,} | "
          f"candidates {len(cand_suppliers):,} | no_match {len(no_match):,}")

    probe_pop = set(no_match) | cand_suppliers
    norm_cache = {s: normalise(s if isinstance(s, str) else "") for s in probe_pop}
    bt_of = {s: btype.get(s, "(blank)") for s in probe_pop}   # authoritative classification

    # ── ch_no_match.csv — the missing instrumentation (full list, tagged) ──
    nm_rows = []
    for s in no_match:
        skey, stoks = norm_cache[s]
        amt = spend.get(s, np.nan)
        nm_rows.append({
            "supplier_clean": s, "supplier_norm": skey,
            "tokens": " ".join(stoks), "block_key": block_key(stoks),
            "initials_sig": initials_sig(stoks), "business_type": bt_of[s],
            "amount": amt, "cum_amount_pct": cumpct.get(s, np.nan),
            "over_threshold": bool(pd.notna(amt) and amt >= a.threshold_amount),
            "empty_key": (skey == ""),
        })
    pd.DataFrame(nm_rows).to_csv(a.out_no_match, index=False, encoding="utf-8-sig")
    print(f"→ {a.out_no_match}: {len(nm_rows):,} no-match suppliers")

    # ── segment out business_types that are predominantly NOT on Companies House ──
    exclude_btypes = {b.strip() for b in a.exclude_business_types.split(",") if b.strip()}
    seg_rows, excluded = [], set()
    if exclude_btypes:
        for s in probe_pop:
            if bt_of[s] in exclude_btypes:
                excluded.add(s)
                scope = "no_match" if s in set(no_match) else "candidates"
                seg_rows.append({"supplier_clean": s, "business_type": bt_of[s],
                                 "scope": scope, "amount": spend.get(s, np.nan)})
        pd.DataFrame(seg_rows).to_csv(a.out_segmented, index=False, encoding="utf-8-sig")
        seg_amt = sum(v for v in (spend.get(s) for s in excluded) if pd.notna(v))
        from collections import Counter as _C
        kc = _C(bt_of[s] for s in excluded)
        print(f"→ {a.out_segmented}: {len(excluded):,} suppliers excluded by business_type "
              f"({fmt_money(seg_amt)}) | {dict(kc)}")

    no_match_probe = [s for s in no_match if s not in excluded]
    cand_probe = [s for s in cand_suppliers if s not in excluded]

    # ── decide what to probe, per scope, money-first ──
    def select(suppliers):
        rows = [(s, spend.get(s, np.nan)) for s in suppliers]
        if a.min_amount > 0:
            rows = [(s, v) for s, v in rows if pd.notna(v) and v >= a.min_amount]
        rows.sort(key=lambda x: (x[1] if pd.notna(x[1]) else -1), reverse=True)
        if a.sample > 0:
            rows = rows[: a.sample]
        return [s for s, _ in rows]

    scopes = []
    if a.scope in ("no_match", "both"):
        scopes.append(("no_match", select(no_match_probe)))
    if a.scope in ("candidates", "both"):
        scopes.append(("candidates", select(sorted(cand_probe))))

    all_pairs = []
    report = ["Companies House — Stage 1 RECALL probe", "=" * 60,
              f"universe (Business) : {len(universe):,}   total spend {fmt_money(uni_amt)}",
              f"resolved (no model) : {len(resolved):,}",
              f"candidates (->model): {len(cand_suppliers):,}",
              f"NO MATCH            : {len(no_match):,}",
              f"segmented out (non-company, not probed): {len(excluded):,}",
              f"probe settings      : token_set blocking on {a.block_tokens} rarest tokens, "
              f"floor {a.relaxed_floor}, anchor_max_df {a.anchor_max_df:,}, top-{a.top_n}",
              f"plausible = match_tier in (clean, head_modifier); loose/weak = bucketed",
              f"spend cohort shown  : amount >= {fmt_money(a.threshold_amount)}", ""]

    from collections import Counter
    for scope_name, suppliers in scopes:
        t0 = time.time()
        print(f"\nProbing scope={scope_name}: {len(suppliers):,} suppliers")
        pairs = []
        for i, s in enumerate(suppliers, 1):
            pn = prod_nums.get(s, set()) if scope_name == "candidates" else set()
            for r in probe_supplier(s, cols, idx, pn, bt_of.get(s, "(blank)"),
                                    a.anchor_max_df, a.top_n, a.relaxed_floor,
                                    a.block_tokens, a.max_df, a.max_pool):
                r["probe_scope"] = scope_name
                pairs.append(r)
            if i % 2000 == 0:
                print(f"  …{i:,}/{len(suppliers):,}")
        all_pairs.extend(pairs)

        # best pair per supplier (rank==1), then split by confidence tier
        best = {}
        for r in pairs:
            if r["rank"] == 1:
                best[r["supplier_clean"]] = r
        bests = list(best.values())
        if scope_name == "candidates":
            bests = [r for r in bests if r["new_vs_production"]]   # only NEW finds matter
        by_tier = {t: [r for r in bests if r["match_tier"] == t]
                   for t in ("clean", "head_modifier", "loose", "weak")}
        plausible = by_tier["clean"] + by_tier["head_modifier"]

        def amt_of(rows):
            return sum(v for v in (spend.get(r["supplier_clean"]) for r in rows) if pd.notna(v))

        report += [
            "─" * 60,
            f"SCOPE: {scope_name}   probed {len(suppliers):,} suppliers in {time.time()-t0:.1f}s"
            + ("   [NEW vs production only]" if scope_name == "candidates" else ""),
            "─" * 60,
            "  best-candidate-per-supplier by confidence tier (count | spend):",
        ]
        for t in ("clean", "head_modifier", "loose", "weak"):
            report.append(f"     {t:<14} {len(by_tier[t]):>7,}   {fmt_money(amt_of(by_tier[t])):>16}")
        report += [
            "",
            f"  PLAUSIBLE (clean + head_modifier): {len(plausible):,}   {fmt_money(amt_of(plausible))}",
            f"    of which spend >= {fmt_money(a.threshold_amount)}: "
            f"{sum(1 for r in plausible if pd.notna(spend.get(r['supplier_clean'], np.nan)) and spend[r['supplier_clean']] >= a.threshold_amount):,}",
            "    (loose = the Mace→ALIBAY-MACE risk class; sampled in the label sheet to confirm)",
            "",
            "  rule-family breakdown over PLAUSIBLE (count | spend):",
        ]
        cat_cnt, cat_money, flag_cnt = Counter(), Counter(), Counter()
        for r in plausible:
            amt = spend.get(r["supplier_clean"], np.nan)
            cat_cnt[r["primary_category"]] += 1
            if pd.notna(amt):
                cat_money[r["primary_category"]] += amt
            for k in ("cat_block_miss", "cat_abbrev_expansion", "cat_trailing_generic",
                      "cat_plural_stem", "cat_prefix_abbrev", "cat_extra_token_only"):
                if r[k]:
                    flag_cnt[k] += 1
        for cat, n in cat_cnt.most_common():
            report.append(f"     {cat:<22} {n:>7,}   {fmt_money(cat_money[cat]):>16}")
        report += ["", "  rule flags over PLAUSIBLE (a pair can hit several):"]
        for k in ("cat_abbrev_expansion", "cat_trailing_generic", "cat_plural_stem",
                  "cat_prefix_abbrev", "cat_extra_token_only", "cat_block_miss"):
            report.append(f"     {k[4:]:<22} {flag_cnt[k]:>7,}")

        report += ["", "  extra CH tokens over PLAUSIBLE (words CH has, supplier lacks):"]
        report += token_freq_table(plausible, "extra_ch_tokens", spend)
        report += ["", "  extra SUPPLIER tokens over PLAUSIBLE (words supplier has, CH lacks):"]
        report += token_freq_table(plausible, "extra_sup_tokens", spend)

        # spend-stratified eyeball samples, split clean vs head_modifier vs loose
        for tname in ("clean", "head_modifier", "loose"):
            rows = sorted(by_tier[tname], key=lambda x: spend.get(x["supplier_clean"], -1),
                          reverse=True)[:10]
            if not rows:
                continue
            report += ["", f"  TOP-SPEND {tname} (plausible if clean/head; loose = likely junk):"]
            for r in rows:
                report.append(
                    f"     {fmt_money(spend.get(r['supplier_clean'], np.nan)):>14}  "
                    f"{str(r['supplier_clean'])[:32]:32} → {str(r['ch_company_name'])[:38]:38} "
                    f"| ts{r['score_token_set']} w{r['score_wratio']} | {r['primary_category']}")
        report.append("")

    # ── stratified labelling sheet (scope × tier), for human true/false marking ──
    if a.label_sample > 0:
        rng2 = np.random.default_rng(a.seed)
        best = {}
        for r in all_pairs:
            if r["rank"] == 1:
                best[(r["probe_scope"], r["supplier_clean"])] = r
        strata = {}
        for r in best.values():
            if r["match_tier"] == "weak":
                continue
            strata.setdefault((r["probe_scope"], r["match_tier"]), []).append(r)
        keys = [k for k in strata if strata[k]]
        picked, seen = [], set()
        if keys:
            per = max(1, a.label_sample // len(keys))
            for k in keys:
                pool = strata[k]
                ix = rng2.choice(len(pool), size=min(per, len(pool)), replace=False)
                for i in ix:
                    picked.append(pool[i])
                    seen.add((pool[i]["probe_scope"], pool[i]["supplier_clean"]))
            # top up to target from anything not yet picked
            leftover = [r for r in best.values() if r["match_tier"] != "weak"
                        and (r["probe_scope"], r["supplier_clean"]) not in seen]
            if leftover and len(picked) < a.label_sample:
                ix = rng2.choice(len(leftover),
                                 size=min(a.label_sample - len(picked), len(leftover)),
                                 replace=False)
                picked += [leftover[i] for i in ix]
        lab_cols = ["label", "note", "supplier_clean", "ch_company_name", "ch_active",
                    "ch_town", "match_tier", "probe_scope", "primary_category",
                    "business_type", "supplier_tokens_n", "supplier_norm", "ch_norm",
                    "shared_tokens", "extra_sup_tokens", "extra_ch_tokens",
                    "score_token_set", "score_wratio", "anchor_token", "anchor_df",
                    "amount", "ch_company_number"]
        lab_rows = []
        for r in picked:
            d = {c: r.get(c, "") for c in lab_cols}
            d["label"] = ""
            d["note"] = ""
            d["amount"] = spend.get(r["supplier_clean"], np.nan)
            lab_rows.append(d)
        lab_df = pd.DataFrame(lab_rows, columns=lab_cols)
        # shuffle so the marker isn't biased by tier ordering
        lab_df = lab_df.sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
        lab_df.to_csv(a.label_out, index=False, encoding="utf-8-sig")
        from collections import Counter as _C2
        tc = _C2((r["probe_scope"], r["match_tier"]) for r in picked)
        print(f"→ {a.label_out}: {len(lab_df):,} pairs to label | strata {dict(tc)}")

    # ── write pairs + report ──
    pairs_df = pd.DataFrame(all_pairs)
    if len(pairs_df):
        front = ["probe_scope", "match_tier", "business_type", "supplier_clean",
                 "ch_company_name", "score_token_set", "score_wratio",
                 "primary_category", "new_vs_production"]
        cols_order = front + [c for c in pairs_df.columns if c not in front]
        pairs_df = pairs_df[cols_order]
    pairs_df.to_csv(a.out_pairs, index=False, encoding="utf-8-sig")
    print(f"→ {a.out_pairs}: {len(pairs_df):,} probe pairs")

    text = "\n".join(report)
    with open(a.report, "w", encoding="utf-8-sig") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\n→ {a.report}")


if __name__ == "__main__":
    main()
