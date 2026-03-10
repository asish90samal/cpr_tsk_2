"""
scoring/risk_element_weights.py
══════════════════════════════════
OWS-aligned Risk Element Weighting table.

Implements the formula from OWS section 3.5:
  Risk Score = E1·w1 + E2·w2 + … + En·wn

Where each Ei is a risk element score (0–1) and wi is the weight for
that element under the given record type.  Weights in each row sum to 1.0.

Record type keys follow OWS convention:
  <LIST_PREFIX>_I  = individual records
  <LIST_PREFIX>_E  = entity records
  CUST_I / CUST_E  = customer-side risk

Reference: OWS Implementation Guide v11.1.1.7, Appendix D, section D.11
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


# ══════════════════════════════════════════════════════════════════════════════
#  Risk element names (column headers in OWS reference data table)
# ══════════════════════════════════════════════════════════════════════════════

RISK_ELEMENTS = [
    "res_ope_countries",    # Residence / Operating countries
    "nat_reg_countries",    # Nationality / Registration countries
    "membership",           # List membership / list type
    "category",             # Sanction / crime category
    "occupation",           # Occupation / role (PEP-relevant)
    "deceased",             # Deceased flag (reduces risk)
    "active",               # Currently active on list
    "external_risk",        # External risk score from data provider
]


# ══════════════════════════════════════════════════════════════════════════════
#  Weight table — keyed by (list_prefix, entity_code)
#  entity_code: "I" = individual, "E" = entity, "PEP_I" = PEP individual
# ══════════════════════════════════════════════════════════════════════════════

# Each entry is a dict of {element: weight}.  Elements not listed have weight 0.
# Weights in each row MUST sum to 1.0.

RISK_ELEMENT_WEIGHTS: Dict[tuple, Dict[str, float]] = {

    # ── HM Treasury (OFSI) ────────────────────────────────────────────────
    ("HMT", "I"): {
        "res_ope_countries": 0.20,
        "nat_reg_countries": 0.20,
        "membership":        0.30,
        "category":          0.30,
    },
    ("HMT", "E"): {
        "nat_reg_countries": 0.25,
        "membership":        0.35,
        "category":          0.40,
    },

    # ── OFAC SDN ──────────────────────────────────────────────────────────
    ("OFAC", "I"): {
        "res_ope_countries": 0.20,
        "nat_reg_countries": 0.20,
        "membership":        0.30,
        "category":          0.30,
    },
    ("OFAC", "E"): {
        "nat_reg_countries": 0.20,
        "membership":        0.30,
        "category":          0.30,
        "external_risk":     0.20,
    },

    # ── EU Consolidated ───────────────────────────────────────────────────
    ("EU", "I"): {
        "res_ope_countries": 0.20,
        "nat_reg_countries": 0.20,
        "membership":        0.30,
        "category":          0.30,
    },
    ("EU", "E"): {
        "nat_reg_countries": 0.25,
        "membership":        0.35,
        "category":          0.40,
    },

    # ── UN Consolidated ───────────────────────────────────────────────────
    ("UN", "I"): {
        "res_ope_countries": 0.25,
        "nat_reg_countries": 0.25,
        "membership":        0.25,
        "category":          0.25,
    },
    ("UN", "E"): {
        "nat_reg_countries": 0.30,
        "membership":        0.35,
        "category":          0.35,
    },

    # ── World-Check Individual (non-PEP) ──────────────────────────────────
    ("WC", "I"): {
        "res_ope_countries": 0.20,
        "nat_reg_countries": 0.20,
        "membership":        0.30,
        "occupation":        0.30,
    },
    ("WC", "E"): {
        "nat_reg_countries": 0.20,
        "membership":        0.30,
        "category":          0.30,
        "external_risk":     0.20,
    },

    # ── World-Check PEP Individual ────────────────────────────────────────
    ("WC", "PEP_I"): {
        "res_ope_countries": 0.20,
        "nat_reg_countries": 0.20,
        "membership":        0.30,
        "occupation":        0.30,
    },

    # ── Dow Jones Watchlist ───────────────────────────────────────────────
    ("DJ", "I"): {
        "res_ope_countries": 0.20,
        "nat_reg_countries": 0.20,
        "membership":        0.25,
        "category":          0.25,
        "external_risk":     0.10,
    },
    ("DJ", "E"): {
        "nat_reg_countries": 0.20,
        "membership":        0.30,
        "category":          0.30,
        "external_risk":     0.20,
    },

    # ── Dow Jones Anti-Corruption (PEP) ───────────────────────────────────
    ("DJAC", "PEP_I"): {
        "res_ope_countries": 0.15,
        "nat_reg_countries": 0.15,
        "membership":        0.30,
        "occupation":        0.25,
        "active":            0.15,
    },

    # ── INTERPOL ─────────────────────────────────────────────────────────
    ("INTERPOL", "I"): {
        "res_ope_countries": 0.15,
        "nat_reg_countries": 0.25,
        "membership":        0.30,
        "category":          0.30,
    },

    # ── Internal SCION list ───────────────────────────────────────────────
    ("SCION", "I"): {
        "res_ope_countries": 0.20,
        "nat_reg_countries": 0.20,
        "membership":        0.30,
        "category":          0.30,
    },
    ("SCION", "E"): {
        "nat_reg_countries": 0.25,
        "membership":        0.35,
        "category":          0.40,
    },

    # ── Customer side (CUST) ──────────────────────────────────────────────
    # Used to compute customer-side risk before blending with watchlist risk
    ("CUST", "I"): {
        "res_ope_countries": 0.50,
        "nat_reg_countries": 0.50,
    },
    ("CUST", "E"): {
        "nat_reg_countries": 0.40,
        "res_ope_countries": 0.40,
        "external_risk":     0.20,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  Element → dataset_type mapping
# ══════════════════════════════════════════════════════════════════════════════

# Maps dataset_type codes from the data layer to their list_prefix for lookup
DATASET_PREFIX_MAP: Dict[str, str] = {
    "OFAC_SDN":        "OFAC",
    "UN_CONSOLIDATED": "UN",
    "EU_SANCTIONS":    "EU",
    "HM_TREASURY":     "HMT",
    "INTERPOL_RED":    "INTERPOL",
    "WORLD_CHECK":     "WC",
    "DOW_JONES":       "DJ",
    "COMPLY_ADVANTAGE":"DJ",   # treated as DJ-equivalent weighting
    "ACCUITY_FIRCO":   "DJ",   # treated as DJ-equivalent weighting
    "SCION":           "SCION",
    "PEP_DATABASE":    "WC",   # PEP uses World-Check weighting by default
}

# Category severity → risk score mapping (used for "category" element)
CATEGORY_RISK: Dict[str, float] = {
    "TERRORISM":               1.00,
    "WEAPONS_OF_MASS_DESTRUCTION": 1.00,
    "DRUG_TRAFFICKING":        0.90,
    "HUMAN_TRAFFICKING":       0.90,
    "PROLIFERATION_FINANCING": 0.95,
    "MONEY_LAUNDERING":        0.85,
    "FRAUD":                   0.75,
    "CORRUPTION":              0.80,
    "ORGANISED_CRIME":         0.85,
    "CYBERCRIME":              0.75,
    "SANCTIONS_EVASION":       0.90,
    "PEP":                     0.65,
    "ADVERSE_MEDIA":           0.55,
    "UNKNOWN":                 0.50,
}

# PEP tier → occupation risk
PEP_TIER_RISK: Dict[int, float] = {
    1: 1.00,   # Head of state / minister level
    2: 0.75,   # Senior official
    3: 0.50,   # Family / associates
}


def get_weights(dataset_type: str, entity_type: str, is_pep: bool = False) -> Dict[str, float]:
    """
    Look up the risk element weight row for a given dataset type and entity type.

    Parameters
    ----------
    dataset_type : e.g. "OFAC_SDN", "WORLD_CHECK"
    entity_type  : "INDIVIDUAL" or "ENTITY"
    is_pep       : True if this is a PEP record (selects PEP_I weighting)

    Returns
    -------
    Dict of {element: weight} — defaults to equal 0.25 weights on the 4 core
    elements if no match is found.
    """
    prefix = DATASET_PREFIX_MAP.get(dataset_type.upper(), "OFAC")
    ec = "PEP_I" if (is_pep and entity_type.upper() == "INDIVIDUAL") else (
         "I"     if entity_type.upper() == "INDIVIDUAL" else "E")

    key = (prefix, ec)
    if key in RISK_ELEMENT_WEIGHTS:
        return RISK_ELEMENT_WEIGHTS[key]

    # Fallback: try without PEP_I
    fallback_key = (prefix, "I" if entity_type.upper() == "INDIVIDUAL" else "E")
    if fallback_key in RISK_ELEMENT_WEIGHTS:
        return RISK_ELEMENT_WEIGHTS[fallback_key]

    # Last resort: equal 4-element default
    return {
        "res_ope_countries": 0.25,
        "nat_reg_countries": 0.25,
        "membership":        0.25,
        "category":          0.25,
    }
