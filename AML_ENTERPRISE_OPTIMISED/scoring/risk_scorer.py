"""
scoring/risk_scorer.py
════════════════════════
OWS-aligned risk scoring module.

Implements three separate scores (matching OWS output attributes):
  1. match_score     — how well the name matched (0–1 float, from RuleChain)
  2. risk_score      — how risky the watchlist record is (0–100 int)
  3. priority_score  — combined alert queue priority (0–100 int)

Formula (OWS section 3.5):
  wl_risk  = Σ(element_score × weight)   for watchlist record elements
  cust_risk= Σ(element_score × weight)   for customer record elements
  risk_score = (wl_risk × WL_WEIGHT) + (cust_risk × CUST_WEIGHT)
  priority_score = round((match_score × MATCH_W + risk_score/100 × RISK_W) × 100)

Default blending weights (OWS default = 50/50):
  WL_WEIGHT   = 0.50
  CUST_WEIGHT = 0.50
  MATCH_W     = 0.60   (match score weighted slightly higher in priority)
  RISK_W      = 0.40
"""
from __future__ import annotations

import pandas as pd
from typing import Optional

from scoring.risk_element_weights import (
    get_weights,
    CATEGORY_RISK,
    PEP_TIER_RISK,
)
from data_layer.generators.ctry import get_country_risk   # existing country lookup


# ── Blending constants ─────────────────────────────────────────────────────
WL_WEIGHT    = 0.50     # watchlist record risk weight
CUST_WEIGHT  = 0.50     # customer record risk weight
MATCH_WEIGHT = 0.60     # match score contribution to priority
RISK_WEIGHT  = 0.40     # risk score contribution to priority


# ══════════════════════════════════════════════════════════════════════════════
#  Element extractors — one per risk element
# ══════════════════════════════════════════════════════════════════════════════

def _country_risk_score(country_str: str) -> float:
    """Lookup country risk [0–1] from CTRY register. Defaults 0.5 for unknown."""
    if not country_str or country_str.strip().upper() in ("", "UNKNOWN"):
        return 0.50
    try:
        rec = get_country_risk(country_str.strip().upper())
        return float(rec.get("risk_score", 0.50))
    except Exception:
        return 0.50


def _extract_res_ope_countries(row: pd.Series, entity_type: str) -> float:
    """
    Residence (individuals) or Operating country (entities) risk score.
    Uses the highest-risk country if multiple are present.
    """
    if entity_type.upper() == "INDIVIDUAL":
        fields = ["address_country", "country_of_residence", "nationality"]
    else:
        fields = ["operating_country", "branch_country"]

    scores = [_country_risk_score(str(row.get(f, ""))) for f in fields]
    return max(scores) if scores else 0.50


def _extract_nat_reg_countries(row: pd.Series, entity_type: str) -> float:
    """
    Nationality (individuals) or Registration country (entities).
    """
    if entity_type.upper() == "INDIVIDUAL":
        fields = ["nationality", "id_number"]   # id_number prefix often encodes nationality
        # For nationality use direct lookup; id_number prefix is not a country
        scores = [_country_risk_score(str(row.get("nationality", "")))]
    else:
        fields = ["incorporation_country", "registration_number"]
        scores = [_country_risk_score(str(row.get("incorporation_country", "")))]

    return max(scores) if scores else 0.50


def _extract_membership(row: pd.Series) -> float:
    """
    Risk contribution from which list the record appears on.
    Based on dataset_type hierarchy: OFAC/UN/EU = 1.0, others lower.
    """
    ds = str(row.get("dataset_type", "")).upper()
    membership_risk = {
        "OFAC_SDN":        1.00,
        "UN_CONSOLIDATED": 1.00,
        "EU_SANCTIONS":    0.95,
        "HM_TREASURY":     0.95,
        "INTERPOL_RED":    0.90,
        "WORLD_CHECK":     0.80,
        "DOW_JONES":       0.75,
        "COMPLY_ADVANTAGE":0.75,
        "ACCUITY_FIRCO":   0.75,
        "SCION":           0.85,
        "PEP_DATABASE":    0.65,
    }
    return membership_risk.get(ds, 0.65)


def _extract_category(row: pd.Series) -> float:
    """Risk from the crime / sanction category."""
    cat = str(row.get("category", row.get("category_type", "UNKNOWN"))).upper()
    # Direct lookup
    if cat in CATEGORY_RISK:
        return CATEGORY_RISK[cat]
    # Partial match
    for key, score in CATEGORY_RISK.items():
        if key in cat:
            return score
    return CATEGORY_RISK["UNKNOWN"]


def _extract_occupation(row: pd.Series) -> float:
    """PEP tier → occupation risk."""
    tier = row.get("pep_tier", None)
    if tier is not None:
        try:
            return PEP_TIER_RISK.get(int(tier), 0.50)
        except (ValueError, TypeError):
            pass
    role = str(row.get("political_role", "")).upper()
    if role:
        return 0.70   # has some role but tier unknown
    return 0.40


def _extract_deceased(row: pd.Series) -> float:
    """Deceased flag reduces risk — returns 1-deceased_bool (so deceased=1 → 0 risk)."""
    dec = row.get("is_deceased", False)
    if isinstance(dec, str):
        dec = dec.upper() in ("TRUE", "YES", "1")
    return 0.0 if bool(dec) else 1.0


def _extract_active(row: pd.Series) -> float:
    """Active on list / mandate active — boosts risk."""
    active = row.get("is_active", True)
    if isinstance(active, str):
        active = active.upper() in ("TRUE", "YES", "1")
    return 1.0 if bool(active) else 0.30


def _extract_external_risk(row: pd.Series) -> float:
    """Pre-computed external risk score from the data provider (0–1)."""
    rv = row.get("risk_weight", row.get("external_risk", None))
    if rv is not None:
        try:
            return float(rv)
        except (ValueError, TypeError):
            pass
    return 0.50


# ── Master extractor dispatch ─────────────────────────────────────────────

_ELEMENT_EXTRACTORS = {
    "res_ope_countries": _extract_res_ope_countries,
    "nat_reg_countries": _extract_nat_reg_countries,
    "membership":        lambda row, _et: _extract_membership(row),
    "category":          lambda row, _et: _extract_category(row),
    "occupation":        lambda row, _et: _extract_occupation(row),
    "deceased":          lambda row, _et: _extract_deceased(row),
    "active":            lambda row, _et: _extract_active(row),
    "external_risk":     lambda row, _et: _extract_external_risk(row),
}


# ══════════════════════════════════════════════════════════════════════════════
#  Core risk score computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_watchlist_risk(
    row: pd.Series,
    entity_type: str,
    dataset_type: str,
    is_pep: bool = False,
) -> tuple[float, dict]:
    """
    Compute the watchlist-record risk score (0.0–1.0).

    Returns
    -------
    (risk_score_0_to_1, element_scores_dict)
    """
    weights = get_weights(dataset_type, entity_type, is_pep)
    element_scores: dict[str, float] = {}
    total = 0.0

    for element, weight in weights.items():
        extractor = _ELEMENT_EXTRACTORS.get(element)
        if extractor:
            val = extractor(row, entity_type)
        else:
            val = 0.50
        element_scores[element] = round(val, 4)
        total += val * weight

    return round(min(1.0, total), 4), element_scores


def compute_customer_risk(
    input_record: dict,
    entity_type: str,
) -> tuple[float, dict]:
    """
    Compute the customer-side risk score (0.0–1.0) from the input record fields.

    Returns
    -------
    (risk_score_0_to_1, element_scores_dict)
    """
    weights = get_weights("CUST", entity_type)
    element_scores: dict[str, float] = {}
    total = 0.0

    # Build a synthetic row from the input dict for the extractors
    row = pd.Series(input_record)

    for element, weight in weights.items():
        extractor = _ELEMENT_EXTRACTORS.get(element)
        if extractor:
            val = extractor(row, entity_type)
        else:
            val = 0.50
        element_scores[element] = round(val, 4)
        total += val * weight

    return round(min(1.0, total), 4), element_scores


def compute_blended_risk_score(
    wl_risk: float,
    cust_risk: float,
    wl_weight: float = WL_WEIGHT,
    cust_weight: float = CUST_WEIGHT,
) -> int:
    """
    Blend watchlist and customer risk into the final OWS-style risk score (0–100).

    Parameters
    ----------
    wl_risk    : Watchlist record risk [0–1]
    cust_risk  : Customer record risk  [0–1]
    wl_weight  : Weight for watchlist side  (default 0.50)
    cust_weight: Weight for customer side   (default 0.50)

    Returns
    -------
    Integer 0–100 (OWS convention: risk score is an integer)
    """
    blended = (wl_risk * wl_weight) + (cust_risk * cust_weight)
    return round(min(100, blended * 100))


def compute_priority_score(
    match_score: float,
    risk_score_0_100: int,
    match_weight: float = MATCH_WEIGHT,
    risk_weight: float = RISK_WEIGHT,
) -> int:
    """
    Compute the alert priority score (0–100) used to rank the alert queue.

    A higher priority score means this alert should be reviewed sooner.

    Parameters
    ----------
    match_score      : Name match score [0–1]
    risk_score_0_100 : Blended risk score [0–100]
    match_weight     : Contribution of match score  (default 0.60)
    risk_weight      : Contribution of risk score   (default 0.40)

    Returns
    -------
    Integer 0–100
    """
    normalised_risk = risk_score_0_100 / 100.0
    priority = (match_score * match_weight) + (normalised_risk * risk_weight)
    return round(min(100, priority * 100))


# ══════════════════════════════════════════════════════════════════════════════
#  Convenience: all-in-one scorer
# ══════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass


@dataclass
class RiskScoringResult:
    wl_risk_score:        float          # watchlist element score [0–1]
    cust_risk_score:      float          # customer element score  [0–1]
    blended_risk_score:   int            # final OWS risk score    [0–100]
    priority_score:       int            # alert queue priority    [0–100]
    wl_element_scores:    dict           # per-element watchlist scores
    cust_element_scores:  dict           # per-element customer scores


def score_record(
    candidate_row: pd.Series,
    input_record: dict,
    entity_type: str,
    match_score: float,
    dataset_type: Optional[str] = None,
    is_pep: bool = False,
) -> RiskScoringResult:
    """
    One-call interface: compute all risk and priority scores for an alert.

    Parameters
    ----------
    candidate_row  : The watchlist record row (pd.Series)
    input_record   : The customer input dict
    entity_type    : "INDIVIDUAL" or "ENTITY"
    match_score    : The name match score [0–1] from RuleChain
    dataset_type   : Dataset code string (falls back to row's dataset_type)
    is_pep         : True if this is a PEP record

    Returns
    -------
    RiskScoringResult with all four scores and element breakdowns
    """
    ds = dataset_type or str(candidate_row.get("dataset_type", "OFAC_SDN"))

    wl_risk, wl_elements = compute_watchlist_risk(candidate_row, entity_type, ds, is_pep)
    cust_risk, cust_elements = compute_customer_risk(input_record, entity_type)
    blended = compute_blended_risk_score(wl_risk, cust_risk)
    priority = compute_priority_score(match_score, blended)

    return RiskScoringResult(
        wl_risk_score=wl_risk,
        cust_risk_score=cust_risk,
        blended_risk_score=blended,
        priority_score=priority,
        wl_element_scores=wl_elements,
        cust_element_scores=cust_elements,
    )
