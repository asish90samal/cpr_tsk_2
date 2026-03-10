"""
generators/san_individual.py
Sanctioned INDIVIDUAL records for all SAN datasets:
  OFAC_SDN, UN_CONSOLIDATED, EU_SANCTIONS, HM_TREASURY, INTERPOL_RED,
  WORLD_CHECK, DOW_JONES, COMPLY_ADVANTAGE, ACCUITY_FIRCO
"""
from __future__ import annotations
import random
from datetime import date
import pandas as pd
from faker import Faker
from data_layer.utils.alias_engine import generate_aliases
from data_layer.utils.transliteration_engine import transliterate

_DATASETS = [
    "OFAC_SDN","UN_CONSOLIDATED","EU_SANCTIONS","HM_TREASURY","INTERPOL_RED",
    "WORLD_CHECK","DOW_JONES","COMPLY_ADVANTAGE","ACCUITY_FIRCO",
]
_COUNTRIES = {
    "IRAN":0.18,"RUSSIA":0.16,"NORTH KOREA":0.09,"SYRIA":0.08,"BELARUS":0.07,
    "VENEZUELA":0.06,"MYANMAR":0.06,"CUBA":0.05,"SUDAN":0.05,"SOMALIA":0.04,
    "AFGHANISTAN":0.04,"LIBYA":0.04,"UKRAINE":0.03,"IRAQ":0.02,"YEMEN":0.02,"UNKNOWN":0.01,
}
_LOCALES = ["en_US","ru_RU","ar_SA","zh_CN","fa_IR","de_DE","fr_FR","tr_TR"]

def _dob():
    return date(random.randint(1940,2000),random.randint(1,12),random.randint(1,28)).isoformat()

def _id_num(country):
    p={"IRAN":"IR","RUSSIA":"RU","NORTH KOREA":"KP","SYRIA":"SY","BELARUS":"BY","VENEZUELA":"VE"}.get(country,"XX")
    return f"{p}{random.randint(1000000,9999999)}"

def generate(n=5000, seed=42):
    random.seed(seed); Faker.seed(seed)
    ctry=list(_COUNTRIES.keys()); wt=list(_COUNTRIES.values()); recs=[]
    for i in range(n):
        fake=Faker(random.choice(_LOCALES))
        name=fake.name().upper()
        nat=random.choices(ctry,weights=wt)[0]
        raw=generate_aliases(name,entity_type="INDIVIDUAL")+transliterate(name)
        aliases="|".join(sorted({a for a in raw if a and a!=name}))
        recs.append({
            "entity_id":f"SAN_IND_{i:06d}","dataset_type":random.choice(_DATASETS),
            "entity_type":"INDIVIDUAL","primary_name":name,"aliases":aliases,
            "dob":_dob(),"gender":random.choice(["M","F","UNKNOWN"]),
            "nationality":nat,"id_number":_id_num(nat),"address_country":nat,
            "is_deceased":random.random()<0.03,  # ~3% deceased — filter in secondary screening
            "risk_weight":round(random.uniform(0.55,1.0),4),"risk_label":int(random.random()<0.15),
        })
    return pd.DataFrame(recs)
