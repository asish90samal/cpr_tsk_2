"""
config/job_types.py
════════════════════
Defines the 10 screening jobs your department runs:

  SAN_IND / SAN_ENT     — Sanctions (government + commercial lists)
  SCION_IND / SCION_ENT — Private / proprietary lists (single merged list)
  PEP_IND / PEP_ENT     — Politically Exposed Persons
  NNS_IND / NNS_ENT     — Negative News Screening (articles + structured)
  CTRY_IND / CTRY_ENT   — Country Risk (lookup, not name-match)

Each job has:
  • Its own ordered list of datasets to screen against
  • Its own alert threshold  (10 unique values)
  • Its own entity_type      (INDIVIDUAL or ENTITY)
  • Per-dataset flags        (auto_alert_on_exact_id, threshold_override)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

# ── Types ──────────────────────────────────────────────────────────────────
DatasetCode = Literal[
    # SAN — government lists
    "OFAC_SDN", "UN_CONSOLIDATED", "EU_SANCTIONS",
    "HM_TREASURY", "INTERPOL_RED",
    # SAN — commercial lists
    "WORLD_CHECK", "DOW_JONES", "COMPLY_ADVANTAGE", "ACCUITY_FIRCO",
    # SCION — single merged private list
    "SCION",
    # PEP
    "PEP_DATABASE",
    # NNS
    "NNS_ARTICLES", "NNS_STRUCTURED",
    # CTRY
    "COUNTRY_RISK",
]

EntityType = Literal["INDIVIDUAL", "ENTITY"]

JobCode = Literal[
    "SAN_IND", "SAN_ENT",
    "SCION_IND", "SCION_ENT",
    "PEP_IND", "PEP_ENT",
    "NNS_IND", "NNS_ENT",
    "CTRY_IND", "CTRY_ENT",
]


@dataclass
class DatasetConfig:
    code:                   DatasetCode
    entity_types:           list[EntityType]
    priority:               int           = 99
    auto_alert_on_exact_id: bool          = False
    threshold_override:     float | None  = None
    description:            str           = ""


@dataclass
class JobConfig:
    job_code:    JobCode
    entity_type: EntityType
    description: str
    threshold:   float            # alert threshold specific to this job
    datasets:    list[DatasetConfig]
    is_country_lookup: bool = False   # True for CTRY jobs — no name scoring


# ══════════════════════════════════════════════════════════════════════════
# SAN — SANCTIONS
# Strictest thresholds: false negatives are a regulatory breach.
# IND: all gov lists + all commercial. ENT: same minus INTERPOL (individual-only).
# Exact passport / registration number → always auto-ALERT.
# ══════════════════════════════════════════════════════════════════════════
SAN_IND = JobConfig(
    job_code="SAN_IND", entity_type="INDIVIDUAL",
    description="Sanctions screening — individual",
    threshold=0.65,
    datasets=[
        DatasetConfig("OFAC_SDN",         ["INDIVIDUAL"], priority=1, auto_alert_on_exact_id=True,
                      description="OFAC Specially Designated Nationals"),
        DatasetConfig("UN_CONSOLIDATED",  ["INDIVIDUAL"], priority=2, auto_alert_on_exact_id=True,
                      description="UN Security Council Consolidated List"),
        DatasetConfig("EU_SANCTIONS",     ["INDIVIDUAL"], priority=3, auto_alert_on_exact_id=True,
                      description="EU Financial Sanctions"),
        DatasetConfig("HM_TREASURY",      ["INDIVIDUAL"], priority=4, auto_alert_on_exact_id=True,
                      description="UK OFSI / HM Treasury"),
        DatasetConfig("INTERPOL_RED",     ["INDIVIDUAL"], priority=5,
                      description="INTERPOL Red Notices"),
        DatasetConfig("WORLD_CHECK",      ["INDIVIDUAL"], priority=6,
                      description="Refinitiv World-Check"),
        DatasetConfig("DOW_JONES",        ["INDIVIDUAL"], priority=7,
                      description="Dow Jones Risk & Compliance"),
        DatasetConfig("COMPLY_ADVANTAGE", ["INDIVIDUAL"], priority=8,
                      description="ComplyAdvantage"),
        DatasetConfig("ACCUITY_FIRCO",    ["INDIVIDUAL"], priority=9,
                      description="Accuity / Firco Compliance Link"),
    ],
)

SAN_ENT = JobConfig(
    job_code="SAN_ENT", entity_type="ENTITY",
    description="Sanctions screening — entity",
    threshold=0.68,
    datasets=[
        DatasetConfig("OFAC_SDN",         ["ENTITY"], priority=1, auto_alert_on_exact_id=True),
        DatasetConfig("UN_CONSOLIDATED",  ["ENTITY"], priority=2, auto_alert_on_exact_id=True),
        DatasetConfig("EU_SANCTIONS",     ["ENTITY"], priority=3, auto_alert_on_exact_id=True),
        DatasetConfig("HM_TREASURY",      ["ENTITY"], priority=4, auto_alert_on_exact_id=True),
        # INTERPOL_RED excluded — individual-only list
        DatasetConfig("WORLD_CHECK",      ["ENTITY"], priority=5),
        DatasetConfig("DOW_JONES",        ["ENTITY"], priority=6),
        DatasetConfig("COMPLY_ADVANTAGE", ["ENTITY"], priority=7),
        DatasetConfig("ACCUITY_FIRCO",    ["ENTITY"], priority=8),
    ],
)


# ══════════════════════════════════════════════════════════════════════════
# SCION — PRIVATE / PROPRIETARY LISTS
# Single merged list. Sources tagged via scion_source column.
# Sources: bank blacklist, correspondent lists, regulatory lists,
#          fraud confirmed, commercial watchlists (World-Check private feed).
# Account / reference number exact match → auto-ALERT.
# ══════════════════════════════════════════════════════════════════════════
SCION_IND = JobConfig(
    job_code="SCION_IND", entity_type="INDIVIDUAL",
    description="SCION private list screening — individual",
    threshold=0.70,
    datasets=[
        DatasetConfig("SCION", ["INDIVIDUAL"], priority=1, auto_alert_on_exact_id=True,
                      description="Merged private list (bank BL + correspondent + regulatory + fraud + commercial)"),
    ],
)

SCION_ENT = JobConfig(
    job_code="SCION_ENT", entity_type="ENTITY",
    description="SCION private list screening — entity",
    threshold=0.72,
    datasets=[
        DatasetConfig("SCION", ["ENTITY"], priority=1, auto_alert_on_exact_id=True,
                      description="Merged private list — entity"),
    ],
)


# ══════════════════════════════════════════════════════════════════════════
# PEP — POLITICALLY EXPOSED PERSONS
# Strictest threshold (false negatives → regulatory breach + reputational risk).
# IND: direct PEP individuals (Tier 1/2/3).
# ENT: entities directly on PEP list (state-owned enterprises)
#      PLUS entities whose beneficial_owner_id links to a PEP individual.
# ══════════════════════════════════════════════════════════════════════════
PEP_IND = JobConfig(
    job_code="PEP_IND", entity_type="INDIVIDUAL",
    description="PEP screening — individual (Tier 1/2/3)",
    threshold=0.60,
    datasets=[
        DatasetConfig("PEP_DATABASE", ["INDIVIDUAL"], priority=1,
                      description="PEP database — individuals (Tier 1=Head of State, Tier 2=Senior Official, Tier 3=Associate)"),
    ],
)

PEP_ENT = JobConfig(
    job_code="PEP_ENT", entity_type="ENTITY",
    description="PEP screening — entity (direct + beneficial ownership)",
    threshold=0.65,
    datasets=[
        DatasetConfig("PEP_DATABASE", ["ENTITY"], priority=1,
                      description="PEP database — entities directly listed (state-owned enterprises, party funds)"),
        # Second pass uses beneficial_owner_id linkage — handled in PEPEntityRule
    ],
)


# ══════════════════════════════════════════════════════════════════════════
# NNS — NEGATIVE NEWS SCREENING
# Two sub-datasets screened together:
#   NNS_ARTICLES   — unstructured: headline, source, pub_date, sentiment, url
#                    Score = name_score × recency_weight × category_severity
#   NNS_STRUCTURED — structured: case_id, category, source, date, country,
#                    linked_entities, sentiment_score
#                    case_id exact match → flag (not auto-ALERT — needs review)
# ══════════════════════════════════════════════════════════════════════════
NNS_IND = JobConfig(
    job_code="NNS_IND", entity_type="INDIVIDUAL",
    description="Negative news screening — individual",
    threshold=0.75,
    datasets=[
        DatasetConfig("NNS_STRUCTURED", ["INDIVIDUAL"], priority=1,
                      description="Structured NNS database with case IDs"),
        DatasetConfig("NNS_ARTICLES",   ["INDIVIDUAL"], priority=2,
                      description="Unstructured news articles with recency decay"),
    ],
)

NNS_ENT = JobConfig(
    job_code="NNS_ENT", entity_type="ENTITY",
    description="Negative news screening — entity",
    threshold=0.78,
    datasets=[
        DatasetConfig("NNS_STRUCTURED", ["ENTITY"], priority=1),
        DatasetConfig("NNS_ARTICLES",   ["ENTITY"], priority=2),
    ],
)


# ══════════════════════════════════════════════════════════════════════════
# CTRY — COUNTRY RISK (pure lookup — no name fuzzy scoring)
# Checks multiple country fields against the COUNTRY_RISK register.
# IND checks: nationality, country_of_birth, passport_country, country_of_residence
# ENT checks: incorporation_country, operating_country, branch_country
# Result: ALERT / REVIEW / NO_ALERT based on country risk tier only.
# threshold = minimum country risk score to trigger ALERT (0–1 scale).
# ══════════════════════════════════════════════════════════════════════════
CTRY_IND = JobConfig(
    job_code="CTRY_IND", entity_type="INDIVIDUAL",
    description="Country risk screening — individual",
    threshold=0.75,          # country_risk_score >= 0.75 → ALERT
    is_country_lookup=True,
    datasets=[
        DatasetConfig("COUNTRY_RISK", ["INDIVIDUAL"], priority=1,
                      description="Country risk register — individual country fields"),
    ],
)

CTRY_ENT = JobConfig(
    job_code="CTRY_ENT", entity_type="ENTITY",
    description="Country risk screening — entity",
    threshold=0.75,
    is_country_lookup=True,
    datasets=[
        DatasetConfig("COUNTRY_RISK", ["ENTITY"], priority=1,
                      description="Country risk register — entity country fields"),
    ],
)


# ══════════════════════════════════════════════════════════════════════════
# REGISTRY — all 10 jobs in one dict
# ══════════════════════════════════════════════════════════════════════════
JOB_REGISTRY: dict[JobCode, JobConfig] = {
    "SAN_IND":   SAN_IND,   "SAN_ENT":   SAN_ENT,
    "SCION_IND": SCION_IND, "SCION_ENT": SCION_ENT,
    "PEP_IND":   PEP_IND,   "PEP_ENT":   PEP_ENT,
    "NNS_IND":   NNS_IND,   "NNS_ENT":   NNS_ENT,
    "CTRY_IND":  CTRY_IND,  "CTRY_ENT":  CTRY_ENT,
}

# Threshold summary for quick reference
THRESHOLD_TABLE: dict[str, dict[str, float]] = {
    "SAN":   {"IND": 0.65, "ENT": 0.68},
    "SCION": {"IND": 0.70, "ENT": 0.72},
    "PEP":   {"IND": 0.60, "ENT": 0.65},
    "NNS":   {"IND": 0.75, "ENT": 0.78},
    "CTRY":  {"IND": 0.75, "ENT": 0.75},   # country risk score, not name score
}


def get_job(job_code: str) -> JobConfig:
    key = job_code.strip().upper()
    if key not in JOB_REGISTRY:
        raise ValueError(f"Unknown job '{key}'. Valid: {list(JOB_REGISTRY.keys())}")
    return JOB_REGISTRY[key]  # type: ignore[index]


def get_job_for_entity(job_type: str, entity_type: str) -> JobConfig:
    """Convenience: get_job_for_entity('SAN', 'INDIVIDUAL') → SAN_IND"""
    etype = entity_type.strip().upper()
    suffix = "IND" if etype == "INDIVIDUAL" else "ENT"
    return get_job(f"{job_type.strip().upper()}_{suffix}")
