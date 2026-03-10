"""generators/san_entity.py — Sanctioned ENTITY records."""
from __future__ import annotations
import random
from faker import Faker
import pandas as pd
from data_layer.utils.alias_engine import generate_aliases

_DATASETS = ["OFAC_SDN","UN_CONSOLIDATED","EU_SANCTIONS","HM_TREASURY",
             "WORLD_CHECK","DOW_JONES","COMPLY_ADVANTAGE","ACCUITY_FIRCO"]
_SUFFIXES = ["LTD","LIMITED","LLC","INC","CORP","PLC","JSC","GMBH","AG","SA","BV","NV","SRL","CO","GROUP","HOLDINGS","TRADING"]
_SECTORS  = ["TRADE","EXPORT","ENERGY","OIL","GAS","FINANCE","CAPITAL","SHIPPING","LOGISTICS","CONSTRUCTION","MINING","CHEMICALS","MARITIME"]
_GEO      = ["GLOBAL","INTERNATIONAL","EASTERN","CASPIAN","EURO","TRANS","GULF","PACIFIC"]
_COUNTRIES= {"IRAN":0.20,"RUSSIA":0.18,"NORTH KOREA":0.10,"SYRIA":0.08,"BELARUS":0.07,"VENEZUELA":0.06,"MYANMAR":0.06,"CUBA":0.05,"SUDAN":0.05,"AFGHANISTAN":0.04,"LIBYA":0.04,"UNKNOWN":0.03}
_LOCALES  = ["en_US","ru_RU","ar_SA","de_DE","fr_FR"]

def _co_name(fake):
    patterns = [
        lambda: f"{random.choice(_GEO)} {random.choice(_SECTORS)}",
        lambda: f"{fake.last_name().upper()} {random.choice(_SECTORS)}",
        lambda: f"{random.choice(_SECTORS)} AND {random.choice(_SECTORS)}",
    ]
    return f"{random.choice(patterns)()} {random.choice(_SUFFIXES)}"

def _reg_num(country):
    p = {"IRAN":"IR","RUSSIA":"RU","NORTH KOREA":"KP","SYRIA":"SY","BELARUS":"BY","VENEZUELA":"VE"}.get(country,"XX")
    return f"{p}{random.randint(1000000,9999999):07d}"

def generate(n=3000, seed=42):
    random.seed(seed); Faker.seed(seed)
    ctry=list(_COUNTRIES.keys()); wt=list(_COUNTRIES.values()); recs=[]
    for i in range(n):
        fake = Faker(random.choice(_LOCALES))
        name = _co_name(fake)
        toks = name.split()
        suffix = toks[-1] if toks[-1] in _SUFFIXES else ""
        name_ns = " ".join(toks[:-1]) if suffix else name
        raw = generate_aliases(name, entity_type="ENTITY")
        raw.append(name_ns)
        aliases = "|".join(sorted({a for a in raw if a and a != name}))
        inc = random.choices(ctry, weights=wt)[0]
        op  = random.choices(ctry + ["UNKNOWN"], weights=wt + [0.08])[0]
        owners = "|".join([Faker("en_US").name().upper() for _ in range(random.randint(1,3))])
        recs.append({
            "entity_id": f"SAN_ENT_{i:06d}",
            "dataset_type": random.choice(_DATASETS),
            "entity_type": "ENTITY",
            "primary_name": name, "aliases": aliases,
            "legal_suffix": suffix, "name_without_suffix": name_ns,
            "registration_number": _reg_num(inc),
            "incorporation_country": inc, "operating_country": op,
            "sector": random.choice(_SECTORS),
            "beneficial_owner_names": owners,
            "risk_weight": round(random.uniform(0.55, 1.0), 4),
            "risk_label": int(random.random() < 0.15),
        })
    return pd.DataFrame(recs)
