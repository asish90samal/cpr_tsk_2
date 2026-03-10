"""generators/scion.py — SCION merged private list (IND + ENT).
scion_source tags each record: BANK_BL | CORRESPONDENT | REGULATORY | FRAUD | COMMERCIAL
account_number / reference_number exact match = auto-ALERT.
"""
from __future__ import annotations
import random
from datetime import date, timedelta
from faker import Faker
import pandas as pd
from data_layer.utils.alias_engine import generate_aliases

_SOURCES = ["BANK_BL","CORRESPONDENT","REGULATORY","FRAUD","COMMERCIAL"]
_SOURCE_W = [0.30, 0.20, 0.20, 0.20, 0.10]
_STATUS   = ["BLACKLIST","GREYLIST","REVIEW","DECLINED_ONBOARDING"]
_REASONS  = ["SUSPICIOUS_TRANSACTIONS","FAILED_KYC","SAR_FILED",
             "FRAUD_CONFIRMED","SANCTIONS_EVASION","STRUCTURING",
             "PEP_UNDISCLOSED","ADVERSE_MEDIA_HIT","TERRORIST_FINANCING"]
_ETYPES   = ["INDIVIDUAL","ENTITY"]
_LOCALES  = ["en_US","ru_RU","ar_SA","de_DE","fr_FR","es_ES"]

def _acct(): return f"ACCT{random.randint(10000000,99999999)}"
def _ref():  return f"REF{random.randint(100000,9999999)}"
def _flag_date():
    return (date.today() - timedelta(days=random.randint(30,1825))).isoformat()

def generate(n=2000, seed=42):
    random.seed(seed); Faker.seed(seed)
    recs = []
    for i in range(n):
        fake   = Faker(random.choice(_LOCALES))
        etype  = random.choice(_ETYPES)
        source = random.choices(_SOURCES, weights=_SOURCE_W)[0]
        if etype == "INDIVIDUAL":
            name = fake.name().upper()
            dob  = date(random.randint(1950,1995),random.randint(1,12),random.randint(1,28)).isoformat()
            nat  = random.choice(["IRAN","RUSSIA","NIGERIA","CHINA","PAKISTAN","UKRAINE","VENEZUELA","UNKNOWN"])
        else:
            name = f"{fake.last_name().upper()} {random.choice(["TRADING","GROUP","SERVICES","CONSULTING"])} LTD"
            dob  = ""
            nat  = random.choice(["IRAN","RUSSIA","NIGERIA","CHINA","UNKNOWN"])
        raw = generate_aliases(name, entity_type=etype)
        aliases = "|".join(sorted({a for a in raw if a and a != name}))
        status  = random.choice(_STATUS)
        recs.append({
            "entity_id":       f"SCION_{i:06d}",
            "dataset_type":    "SCION",
            "entity_type":     etype,
            "primary_name":    name,
            "aliases":         aliases,
            "dob":             dob,
            "nationality":     nat,
            "scion_source":    source,
            "account_number":  _acct(),
            "reference_number":_ref(),
            "watchlist_status":status,
            "flagging_reason":  random.choice(_REASONS),
            "flagged_date":     _flag_date(),
            "flagged_by":       random.choice(["COMPLIANCE_AUTO","COMPLIANCE_MANUAL","FRAUD_TEAM"]),
            "risk_weight":      round(random.uniform(0.5, 1.0), 4),
            "risk_label":       int(status in {"BLACKLIST","DECLINED_ONBOARDING"}),
        })
    return pd.DataFrame(recs)
