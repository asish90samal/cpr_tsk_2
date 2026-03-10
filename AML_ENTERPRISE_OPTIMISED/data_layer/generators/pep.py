"""
generators/pep.py
=================
PEP_DATABASE records — INDIVIDUAL (Tier 1/2/3) + ENTITY (direct PEP + beneficial-owner linked).

INDIVIDUAL columns:
  entity_id, dataset_type, entity_type, primary_name, aliases,
  dob, gender, nationality, country_of_office,
  pep_tier (1/2/3), political_role, party,
  mandate_start, mandate_end, is_active,
  associated_pep_id (Tier-3 link to Tier-1),
  beneficial_owner_pep_id (empty for individuals),
  risk_weight, risk_label

ENTITY columns (all above plus):
  beneficial_owner_pep_id — links entity to a PEP_IND entity_id
  legal_suffix, name_without_suffix
"""
from __future__ import annotations
import random
from datetime import date, timedelta

import pandas as pd
from faker import Faker

from data_layer.utils.alias_engine import generate_aliases
from data_layer.utils.transliteration_engine import transliterate

_ROLES_T1 = [
    "PRESIDENT", "PRIME MINISTER", "MINISTER OF FINANCE", "MINISTER OF DEFENCE",
    "MINISTER OF FOREIGN AFFAIRS", "CENTRAL BANK GOVERNOR", "SPEAKER OF PARLIAMENT",
    "SUPREME COURT CHIEF JUSTICE", "HEAD OF NATIONAL INTELLIGENCE",
    "COMMANDER OF ARMED FORCES", "MINISTER OF ENERGY", "ATTORNEY GENERAL",
]
_ROLES_T2 = [
    "DEPUTY MINISTER", "STATE SECRETARY", "REGIONAL GOVERNOR", "AMBASSADOR",
    "CONSUL GENERAL", "SENIOR PARTY OFFICIAL", "MEMBER OF PARLIAMENT", "SENATOR",
    "DIRECTOR GENERAL", "CHIEF OF STAFF", "VICE PRESIDENT", "DEPUTY PRIME MINISTER",
    "DIRECTOR OF STATE BANK", "HEAD OF CUSTOMS", "DIRECTOR OF TAX AUTHORITY",
]
_ROLES_T3 = [
    "SPOUSE OF PEP", "CHILD OF PEP", "PARENT OF PEP", "SIBLING OF PEP",
    "BUSINESS ASSOCIATE OF PEP", "CLOSE FRIEND OF PEP", "LEGAL REPRESENTATIVE OF PEP",
]
_HR = ["IRAN","RUSSIA","NORTH KOREA","SYRIA","MYANMAR","BELARUS","VENEZUELA",
       "CUBA","SUDAN","SOMALIA","AFGHANISTAN","LIBYA","IRAQ","YEMEN","ZIMBABWE"]
_MR = ["NIGERIA","KENYA","ETHIOPIA","UKRAINE","PAKISTAN","BANGLADESH",
       "CAMBODIA","TURKMENISTAN","UZBEKISTAN","TAJIKISTAN","KAZAKHSTAN","AZERBAIJAN"]
_PEP_ENT_LABELS = [
    "STATE OIL COMPANY", "NATIONAL BANK", "MINISTRY TRADING ARM",
    "NATIONAL INFRASTRUCTURE FUND", "PARTY INVESTMENT VEHICLE",
    "STATE DEFENCE ENTERPRISE", "SOVEREIGN WEALTH FUND",
]
_PARTIES = ["RULING PARTY","OPPOSITION PARTY","COMMUNIST PARTY","NATIONAL FRONT","WORKERS PARTY",""]
_LOCALES = ["en_US","ru_RU","ar_SA","zh_CN","fa_IR","de_DE","fr_FR"]


def _mandate():
    start = date.today() - timedelta(days=random.randint(0, 20) * 365 + random.randint(0, 364))
    if random.random() < 0.40:
        end = start + timedelta(days=random.randint(365, 365 * 8))
        return start.isoformat(), end.isoformat(), False
    return start.isoformat(), "", True


def generate(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    Faker.seed(seed)

    recs: list[dict] = []
    tier1_ids: list[str] = []

    # ── INDIVIDUAL PEPs (70 % of records) ─────────────────────────────────
    n_ind = int(n * 0.70)
    for i in range(n_ind):
        fake = Faker(random.choice(_LOCALES))
        name = fake.name().upper()

        t = random.random()
        if t < 0.30:
            tier, role = 1, random.choice(_ROLES_T1)
        elif t < 0.70:
            tier, role = 2, random.choice(_ROLES_T2)
        else:
            tier, role = 3, random.choice(_ROLES_T3)

        pool = _HR + _MR if tier <= 2 else _HR + _MR + ["UNKNOWN"]
        country = random.choice(pool)
        ms, me, active = _mandate()

        assoc = ""
        if tier == 3 and tier1_ids:
            assoc = random.choice(tier1_ids)
        if tier == 1:
            tier1_ids.append(f"PEP_IND_{i:06d}")

        raw = generate_aliases(name, entity_type="INDIVIDUAL") + transliterate(name)
        aliases = "|".join(sorted({a for a in raw if a and a != name}))

        base = {1: 0.85, 2: 0.70, 3: 0.55}[tier]
        if active:
            base = min(1.0, base + 0.10)

        recs.append({
            "entity_id":             f"PEP_IND_{i:06d}",
            "dataset_type":          "PEP_DATABASE",
            "entity_type":           "INDIVIDUAL",
            "primary_name":          name,
            "aliases":               aliases,
            "dob":                   date(random.randint(1940, 1985), random.randint(1, 12), random.randint(1, 28)).isoformat(),
            "gender":                random.choice(["M", "F", "UNKNOWN"]),
            "nationality":           country,
            "country_of_office":     country,
            "pep_tier":              tier,
            "political_role":        role,
            "party":                 random.choice(_PARTIES),
            "mandate_start":         ms,
            "mandate_end":           me,
            "is_active":             active,
            "associated_pep_id":     assoc,
            "beneficial_owner_pep_id": "",
            "risk_weight":           round(base + random.uniform(-0.05, 0.05), 4),
            "risk_label":            int(random.random() < 0.20),
        })

    # ── ENTITY PEPs (30 % of records) ─────────────────────────────────────
    n_ent = n - n_ind
    for j in range(n_ent):
        country = random.choice(_HR)
        label   = random.choice(_PEP_ENT_LABELS)
        name    = f"{country} {label}"
        raw     = generate_aliases(name, entity_type="ENTITY")
        aliases = "|".join(sorted({a for a in raw if a and a != name}))
        bop     = random.choice(tier1_ids) if tier1_ids else ""

        recs.append({
            "entity_id":             f"PEP_ENT_{j:06d}",
            "dataset_type":          "PEP_DATABASE",
            "entity_type":           "ENTITY",
            "primary_name":          name,
            "aliases":               aliases,
            "dob":                   "",
            "gender":                "",
            "nationality":           country,
            "country_of_office":     country,
            "pep_tier":              1,
            "political_role":        "STATE_OWNED_ENTERPRISE",
            "party":                 "",
            "mandate_start":         "",
            "mandate_end":           "",
            "is_active":             True,
            "associated_pep_id":     "",
            "beneficial_owner_pep_id": bop,
            "risk_weight":           round(random.uniform(0.70, 1.0), 4),
            "risk_label":            int(random.random() < 0.25),
        })

    return pd.DataFrame(recs)
