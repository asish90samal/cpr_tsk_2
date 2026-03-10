"""
generators/nns.py
=================
NNS — Negative News Screening — TWO sub-datasets:

NNS_ARTICLES  (dataset_type='NNS_ARTICLES')
  Unstructured news: headline, source, publication_date,
  sentiment_score, url, recency_weight.
  Score = name_fuzzy × recency_weight × category_severity.

NNS_STRUCTURED (dataset_type='NNS_STRUCTURED')
  Compiled structured database with:
  case_id, category, source_publication, source_date,
  sentiment_score, country_of_subject, linked_entity_ids.
  case_id exact match → flag for review (not auto-ALERT).

Both cover INDIVIDUAL and ENTITY.
"""
from __future__ import annotations
import random
from datetime import date, timedelta

import pandas as pd
from faker import Faker

from data_layer.utils.alias_engine import generate_aliases

# ── Shared ────────────────────────────────────────────────────────────────
_CATEGORIES = [
    "FRAUD", "CORRUPTION", "MONEY_LAUNDERING", "TERRORISM_FINANCING",
    "DRUG_TRAFFICKING", "HUMAN_TRAFFICKING", "CYBERCRIME", "SANCTIONS_EVASION",
    "BRIBERY", "TAX_EVASION", "INSIDER_TRADING", "EMBEZZLEMENT",
    "ORGANISED_CRIME", "ARMS_TRAFFICKING", "ENVIRONMENTAL_CRIME",
]
_CATEGORY_SEVERITY = {
    "FRAUD": 0.70, "CORRUPTION": 0.75, "MONEY_LAUNDERING": 0.95,
    "TERRORISM_FINANCING": 1.00, "DRUG_TRAFFICKING": 0.90,
    "HUMAN_TRAFFICKING": 0.85, "CYBERCRIME": 0.75, "SANCTIONS_EVASION": 0.95,
    "BRIBERY": 0.70, "TAX_EVASION": 0.65, "INSIDER_TRADING": 0.60,
    "EMBEZZLEMENT": 0.70, "ORGANISED_CRIME": 0.90,
    "ARMS_TRAFFICKING": 0.95, "ENVIRONMENTAL_CRIME": 0.55,
}
_SOURCES = [
    "REUTERS", "BBC", "AP_NEWS", "BLOOMBERG", "FINANCIAL_TIMES",
    "WALL_STREET_JOURNAL", "THE_GUARDIAN", "LE_MONDE", "OCCRP",
    "TRANSPARENCY_INTERNATIONAL", "GLOBAL_WITNESS", "FATF_REPORTS",
    "LOCAL_PRESS_IRAN", "LOCAL_PRESS_RUSSIA", "LOCAL_PRESS_SYRIA",
]
_COUNTRIES = ["IRAN","RUSSIA","NORTH KOREA","SYRIA","MYANMAR","NIGERIA",
              "VENEZUELA","CHINA","TURKEY","UKRAINE","PAKISTAN","UNKNOWN"]
_ETYPES  = ["INDIVIDUAL", "ENTITY"]
_LOCALES = ["en_US","ru_RU","ar_SA","zh_CN","de_DE","fr_FR"]


def _pub_date(max_years: int = 8) -> str:
    days_back = random.randint(1, max_years * 365)
    return (date.today() - timedelta(days=days_back)).isoformat()


def _recency_weight(pub_str: str) -> float:
    try:
        days = (date.today() - date.fromisoformat(pub_str)).days
        yr   = days / 365.0
    except ValueError:
        return 0.5
    if yr <= 1:   return 1.00
    if yr <= 2:   return 0.90
    if yr <= 3:   return 0.75
    if yr <= 5:   return 0.60
    if yr <= 7:   return 0.45
    return 0.30


# ── NNS_ARTICLES ──────────────────────────────────────────────────────────

def generate_articles(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Unstructured news article records."""
    random.seed(seed)
    Faker.seed(seed)
    recs = []

    for i in range(n):
        fake  = Faker(random.choice(_LOCALES))
        etype = random.choice(_ETYPES)
        name  = (fake.name() if etype == "INDIVIDUAL"
                 else f"{fake.last_name().upper()} {random.choice(['TRADING','GROUP','HOLDINGS','CAPITAL'])} LTD").upper()
        nat   = random.choice(_COUNTRIES)
        cat   = random.choice(_CATEGORIES)
        pub   = _pub_date()
        rec_w = _recency_weight(pub)
        sev   = _CATEGORY_SEVERITY[cat]
        src   = random.choice(_SOURCES)

        raw     = generate_aliases(name, entity_type=etype)
        aliases = "|".join(sorted({a for a in raw if a and a != name}))

        recs.append({
            "entity_id":        f"NNS_ART_{i:06d}",
            "dataset_type":     "NNS_ARTICLES",
            "entity_type":      etype,
            "primary_name":     name,
            "aliases":          aliases,
            "dob":              (date(random.randint(1940, 2000), random.randint(1, 12), random.randint(1, 28)).isoformat()
                                 if etype == "INDIVIDUAL" else ""),
            "nationality":      nat,
            "source_publication": src,
            "publication_date": pub,
            "recency_weight":   rec_w,
            "category":         cat,
            "category_severity": sev,
            "sentiment_score":  round(random.uniform(-1.0, -0.20), 3),
            "headline_snippet": f"{cat.replace('_',' ')} ALLEGATIONS — {name.split()[0]}",
            "url":              f"https://{src.lower().replace('_','.')}.com/article/{random.randint(100000,999999)}",
            "risk_weight":      round(sev * rec_w, 4),
            "risk_label":       int(random.random() < 0.25),
        })

    return pd.DataFrame(recs)


# ── NNS_STRUCTURED ────────────────────────────────────────────────────────

def generate_structured(n: int = 2000, seed: int = 43) -> pd.DataFrame:
    """Structured NNS database with case IDs."""
    random.seed(seed)
    Faker.seed(seed)
    recs = []

    for i in range(n):
        fake  = Faker(random.choice(_LOCALES))
        etype = random.choice(_ETYPES)
        name  = (fake.name() if etype == "INDIVIDUAL"
                 else f"{fake.last_name().upper()} {random.choice(['TRADING','GROUP','SERVICES','CONSULTING'])} LTD").upper()
        nat   = random.choice(_COUNTRIES)
        cat   = random.choice(_CATEGORIES)
        sev   = _CATEGORY_SEVERITY[cat]
        src   = random.choice(_SOURCES)
        pub   = _pub_date(max_years=10)

        raw     = generate_aliases(name, entity_type=etype)
        aliases = "|".join(sorted({a for a in raw if a and a != name}))

        # Linked entity IDs (2-4 associates)
        n_linked  = random.randint(0, 4)
        linked    = "|".join([f"NNS_ART_{random.randint(0, 2999):06d}" for _ in range(n_linked)])

        recs.append({
            "entity_id":         f"NNS_STR_{i:06d}",
            "dataset_type":      "NNS_STRUCTURED",
            "entity_type":       etype,
            "primary_name":      name,
            "aliases":           aliases,
            "dob":               (date(random.randint(1940, 2000), random.randint(1, 12), random.randint(1, 28)).isoformat()
                                  if etype == "INDIVIDUAL" else ""),
            "nationality":       nat,
            "case_id":           f"NNS-{date.today().year}-{i:06d}",
            "category":          cat,
            "category_severity": sev,
            "source_publication": src,
            "source_date":       pub,
            "sentiment_score":   round(random.uniform(-1.0, -0.20), 3),
            "country_of_subject": nat,
            "linked_entity_ids": linked,
            "risk_weight":       round(sev * _recency_weight(pub), 4),
            "risk_label":        int(random.random() < 0.25),
        })

    return pd.DataFrame(recs)
