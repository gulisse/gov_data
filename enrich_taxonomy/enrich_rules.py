"""
enrich_rules.py — Shared deterministic detection rules.

Used by enrich_stp0_flag_scope.py (go-forward pipeline flagging) and
remediate.py (historical data remediation) so both apply identical
logic. All patterns/thresholds live in config.py.

Every function takes a DataFrame with the taxonomy-base columns and
returns masks / findings; nothing here mutates the input frame.
"""

import re

import pandas as pd

from config import (
    ACCOMMODATION_REGEX,
    ACQUISITION_REGEX,
    CATCHALL_LEVEL1,
    CONCESSION_REGEX,
    CONSENSUS_THRESHOLD,
    FLOW_INCOME,
    FLOW_REFUND,
    FLOW_REVERSAL,
    FLOW_SPEND,
    INCOME_TERM_REGEX,
    LEDGER_TERM_REGEX,
    REDACTED_SUPPLIER_REGEX,
    SCOPE_IN,
    SCOPE_LEDGER,
    SCOPE_STATUTORY,
    STATUTORY_PAYEE_REGEX,
    STATUTORY_TERM_REGEX,
    VENDOR_PINS,
)


# ──────────────────────────────────────────────
# Column helpers
# ──────────────────────────────────────────────

def _low(df: pd.DataFrame, col: str) -> pd.Series:
    """Lower-cased, NaN-safe string view of a column ('' if missing)."""
    if col not in df.columns:
        return pd.Series("", index=df.index)
    return df[col].fillna("").astype(str).str.lower()


def blank_row_mask(df: pd.DataFrame) -> pd.Series:
    """Rows where every value is NaN/empty — data-integrity removals."""
    return df.isna().all(axis=1) | (
        df.astype(str).apply(lambda s: s.str.strip(), axis=0).eq("").all(axis=1)
    )


# ──────────────────────────────────────────────
# procurement_scope
# ──────────────────────────────────────────────

def compute_scope(df: pd.DataFrame) -> pd.Series:
    """
    IN_SCOPE / STATUTORY_TRANSFER / LEDGER_CONTROL per row.

    Statutory fires on a tax-authority payee (decisive on its own), or a
    statutory expense term where the row landed in Financial Services.
    Ledger fires on control/holding/suspense-account expense labels.
    Concessionary-travel rows correctly in Passenger Transport are
    protected from the statutory term match.
    """
    et = _low(df, "expense_type")
    sc = _low(df, "supplier_clean")
    l1 = df["Level 1"].fillna("") if "Level 1" in df.columns else pd.Series("", index=df.index)

    payee = sc.str.contains(STATUTORY_PAYEE_REGEX, regex=True)
    term = et.str.contains(STATUTORY_TERM_REGEX, regex=True)
    ledger = et.str.contains(LEDGER_TERM_REGEX, regex=True)
    concession = et.str.contains(CONCESSION_REGEX, regex=True) & (
        l1 == "Passenger Transport"
    )

    statutory = (payee | (term & (l1 == "Financial Services"))) & ~concession

    scope = pd.Series(SCOPE_IN, index=df.index)
    scope[ledger & ~statutory] = SCOPE_LEDGER
    scope[statutory] = SCOPE_STATUTORY
    return scope


# ──────────────────────────────────────────────
# flow_type_taxonomy_key_aggregate
# ──────────────────────────────────────────────

def compute_flow_type(df: pd.DataFrame, scope: pd.Series) -> pd.Series:
    """
    SPEND / REFUND / INCOME / ACCOUNTING_REVERSAL per row, judged on the
    aggregated total_amount (see README note on netting within groups).

    Positive rows are SPEND. Negative rows split:
      • ledger/statutory scope  → ACCOUNTING_REVERSAL
      • income expense terms    → INCOME
      • everything else         → REFUND (genuine credits — keep netting)
    """
    amt = pd.to_numeric(df.get("total_amount"), errors="coerce").fillna(0)
    et = _low(df, "expense_type")

    flow = pd.Series(FLOW_SPEND, index=df.index)
    neg = amt < 0

    income = neg & et.str.contains(INCOME_TERM_REGEX, regex=True)
    reversal = neg & (scope != SCOPE_IN)

    flow[neg] = FLOW_REFUND
    flow[income] = FLOW_INCOME
    flow[reversal] = FLOW_REVERSAL          # reversal outranks income label
    return flow


# ──────────────────────────────────────────────
# Anchor rules
# ──────────────────────────────────────────────

def anchor_accommodation_mask(df: pd.DataFrame) -> pd.Series:
    """Temporary/emergency accommodation classified outside Housing."""
    et = _low(df, "expense_type")
    l1 = df["Level 1"].fillna("") if "Level 1" in df.columns else pd.Series("", index=df.index)
    return et.str.contains(ACCOMMODATION_REGEX, regex=True) & (l1 != "Housing Management")


def anchor_acquisition_mask(df: pd.DataFrame) -> pd.Series:
    """Property/dwelling acquisition classified into a catch-all."""
    et = _low(df, "expense_type")
    l1 = df["Level 1"].fillna("") if "Level 1" in df.columns else pd.Series("", index=df.index)
    return et.str.contains(ACQUISITION_REGEX, regex=True) & l1.isin(CATCHALL_LEVEL1)


# ──────────────────────────────────────────────
# Vendor pins
# ──────────────────────────────────────────────

def vendor_pin_hits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rows matching a VENDOR_PINS pattern whose current code differs from
    the pin target. Returns DataFrame[index, pin_pattern, pin_code].
    """
    sc = _low(df, "supplier_clean")
    code = pd.to_numeric(df.get("taxonomy_code"), errors="coerce")

    hits = []
    for pattern, target in VENDOR_PINS.items():
        m = sc.str.contains(pattern, regex=True) & (code != target)
        for idx in df.index[m]:
            hits.append({"index": idx, "pin_pattern": pattern, "pin_code": target})
    return pd.DataFrame(hits)


def vendor_pin_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per pin: rows matched, £, current Level-1 spread, and the
    share already in the pin target's Level 1 (concentration guard).
    """
    sc = _low(df, "supplier_clean")
    code = pd.to_numeric(df.get("taxonomy_code"), errors="coerce")

    out = []
    for pattern, target in VENDOR_PINS.items():
        m = sc.str.contains(pattern, regex=True)
        rows = df[m]
        if rows.empty:
            out.append({"pin_pattern": pattern, "pin_code": target, "rows": 0,
                        "total_amount": 0, "already_on_target": 0,
                        "concentration": None, "level1_spread": "", "warning": ""})
            continue
        on_target = int((code[m] == target).sum())
        conc = on_target / len(rows)
        spread = "; ".join(
            f"{k}:{v}" for k, v in rows["Level 1"].fillna("?").value_counts().items()
        )
        out.append({
            "pin_pattern": pattern,
            "pin_code": target,
            "rows": len(rows),
            "total_amount": round(rows["total_amount"].sum(), 2),
            "already_on_target": on_target,
            "concentration": round(conc, 3),
            "level1_spread": spread,
            "warning": "" if conc >= 0.5 else "DIVERSE VENDOR — review before apply",
        })
    return pd.DataFrame(out)


# ──────────────────────────────────────────────
# Supplier-consensus reconciliation
# ──────────────────────────────────────────────

def consensus_findings(df: pd.DataFrame) -> pd.DataFrame:
    """
    For suppliers with >1 Level 1 and a modal share >= CONSENSUS_THRESHOLD,
    return the minority rows:
        index, consensus_l1, consensus_code, kind
    kind = 'CATCHALL' (minority sits in a magnet category) or 'REVIEW'.
    consensus_code = the supplier's modal taxonomy_code within the modal
    Level 1 (the best deterministic suggestion available).
    """
    known = df["supplier_clean"].notna() & ~df["supplier_clean"].str.contains(
        REDACTED_SUPPLIER_REGEX, na=False, regex=True
    )
    findings = []
    for supplier, rows in df[known].groupby("supplier_clean"):
        if rows["Level 1"].nunique() < 2:
            continue
        vc = rows["Level 1"].value_counts()
        top, top_n = vc.index[0], vc.iloc[0]
        if top_n / len(rows) < CONSENSUS_THRESHOLD:
            continue
        modal_code = (
            pd.to_numeric(rows.loc[rows["Level 1"] == top, "taxonomy_code"],
                          errors="coerce").mode()
        )
        modal_code = int(modal_code.iloc[0]) if len(modal_code) else None
        for idx, r in rows[rows["Level 1"] != top].iterrows():
            findings.append({
                "index": idx,
                "consensus_l1": top,
                "consensus_code": modal_code,
                "kind": "CATCHALL" if r["Level 1"] in CATCHALL_LEVEL1 else "REVIEW",
            })
    return pd.DataFrame(findings)


# ──────────────────────────────────────────────
# NEC default detection
# ──────────────────────────────────────────────

def nec_mask(df: pd.DataFrame) -> pd.Series:
    """Codes ending 99/999/9999 — Not-Elsewhere-Classified defaults."""
    code = pd.to_numeric(df.get("taxonomy_code"), errors="coerce").astype("Int64")
    return code.astype(str).str.match(r".*(99|999|9999)$").fillna(False)
