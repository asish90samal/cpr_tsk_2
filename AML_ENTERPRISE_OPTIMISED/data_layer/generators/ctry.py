"""generators/ctry.py — COUNTRY_RISK register for CTRY_IND and CTRY_ENT jobs."""
from __future__ import annotations
import pandas as pd

_REGISTER = [
    {"country":"NORTH KOREA","iso2":"KP","risk_score":1.00,"risk_tier":"HIGH","fatf_status":"BLACK","un_sanctions":True,"eu_sanctions":True,"ofac_sanctions":True},
    {"country":"IRAN","iso2":"IR","risk_score":1.00,"risk_tier":"HIGH","fatf_status":"BLACK","un_sanctions":True,"eu_sanctions":True,"ofac_sanctions":True},
    {"country":"MYANMAR","iso2":"MM","risk_score":0.95,"risk_tier":"HIGH","fatf_status":"BLACK","un_sanctions":True,"eu_sanctions":True,"ofac_sanctions":True},
    {"country":"RUSSIA","iso2":"RU","risk_score":0.92,"risk_tier":"HIGH","fatf_status":"GREY","un_sanctions":False,"eu_sanctions":True,"ofac_sanctions":True},
    {"country":"SYRIA","iso2":"SY","risk_score":0.90,"risk_tier":"HIGH","fatf_status":"GREY","un_sanctions":True,"eu_sanctions":True,"ofac_sanctions":True},
    {"country":"BELARUS","iso2":"BY","risk_score":0.88,"risk_tier":"HIGH","fatf_status":"GREY","un_sanctions":False,"eu_sanctions":True,"ofac_sanctions":True},
    {"country":"VENEZUELA","iso2":"VE","risk_score":0.85,"risk_tier":"HIGH","fatf_status":"GREY","un_sanctions":False,"eu_sanctions":False,"ofac_sanctions":True},
    {"country":"CUBA","iso2":"CU","risk_score":0.82,"risk_tier":"HIGH","fatf_status":"GREY","un_sanctions":False,"eu_sanctions":False,"ofac_sanctions":True},
    {"country":"SUDAN","iso2":"SD","risk_score":0.80,"risk_tier":"HIGH","fatf_status":"GREY","un_sanctions":True,"eu_sanctions":True,"ofac_sanctions":True},
    {"country":"SOMALIA","iso2":"SO","risk_score":0.80,"risk_tier":"HIGH","fatf_status":"GREY","un_sanctions":True,"eu_sanctions":True,"ofac_sanctions":False},
    {"country":"AFGHANISTAN","iso2":"AF","risk_score":0.78,"risk_tier":"HIGH","fatf_status":"GREY","un_sanctions":True,"eu_sanctions":True,"ofac_sanctions":True},
    {"country":"LIBYA","iso2":"LY","risk_score":0.78,"risk_tier":"HIGH","fatf_status":"GREY","un_sanctions":True,"eu_sanctions":True,"ofac_sanctions":False},
    {"country":"IRAQ","iso2":"IQ","risk_score":0.76,"risk_tier":"HIGH","fatf_status":"GREY","un_sanctions":True,"eu_sanctions":False,"ofac_sanctions":True},
    {"country":"YEMEN","iso2":"YE","risk_score":0.76,"risk_tier":"HIGH","fatf_status":"GREY","un_sanctions":True,"eu_sanctions":False,"ofac_sanctions":False},
    {"country":"ZIMBABWE","iso2":"ZW","risk_score":0.75,"risk_tier":"HIGH","fatf_status":"GREY","un_sanctions":False,"eu_sanctions":True,"ofac_sanctions":True},
    {"country":"NIGERIA","iso2":"NG","risk_score":0.68,"risk_tier":"MEDIUM","fatf_status":"GREY","un_sanctions":False,"eu_sanctions":False,"ofac_sanctions":False},
    {"country":"PAKISTAN","iso2":"PK","risk_score":0.65,"risk_tier":"MEDIUM","fatf_status":"GREY","un_sanctions":False,"eu_sanctions":False,"ofac_sanctions":False},
    {"country":"UKRAINE","iso2":"UA","risk_score":0.60,"risk_tier":"MEDIUM","fatf_status":"GREY","un_sanctions":False,"eu_sanctions":False,"ofac_sanctions":False},
    {"country":"KENYA","iso2":"KE","risk_score":0.58,"risk_tier":"MEDIUM","fatf_status":"GREY","un_sanctions":False,"eu_sanctions":False,"ofac_sanctions":False},
    {"country":"ETHIOPIA","iso2":"ET","risk_score":0.55,"risk_tier":"MEDIUM","fatf_status":"GREY","un_sanctions":False,"eu_sanctions":False,"ofac_sanctions":False},
    {"country":"BANGLADESH","iso2":"BD","risk_score":0.55,"risk_tier":"MEDIUM","fatf_status":"GREY","un_sanctions":False,"eu_sanctions":False,"ofac_sanctions":False},
    {"country":"CHINA","iso2":"CN","risk_score":0.50,"risk_tier":"MEDIUM","fatf_status":"NORMAL","un_sanctions":False,"eu_sanctions":False,"ofac_sanctions":False},
    {"country":"TURKEY","iso2":"TR","risk_score":0.48,"risk_tier":"MEDIUM","fatf_status":"NORMAL","un_sanctions":False,"eu_sanctions":False,"ofac_sanctions":False},
    {"country":"INDIA","iso2":"IN","risk_score":0.40,"risk_tier":"MEDIUM","fatf_status":"NORMAL","un_sanctions":False,"eu_sanctions":False,"ofac_sanctions":False},
    {"country":"BRAZIL","iso2":"BR","risk_score":0.38,"risk_tier":"MEDIUM","fatf_status":"NORMAL","un_sanctions":False,"eu_sanctions":False,"ofac_sanctions":False},
    {"country":"UNITED STATES","iso2":"US","risk_score":0.15,"risk_tier":"LOW","fatf_status":"NORMAL","un_sanctions":False,"eu_sanctions":False,"ofac_sanctions":False},
    {"country":"UNITED KINGDOM","iso2":"GB","risk_score":0.12,"risk_tier":"LOW","fatf_status":"NORMAL","un_sanctions":False,"eu_sanctions":False,"ofac_sanctions":False},
    {"country":"GERMANY","iso2":"DE","risk_score":0.10,"risk_tier":"LOW","fatf_status":"NORMAL","un_sanctions":False,"eu_sanctions":False,"ofac_sanctions":False},
    {"country":"FRANCE","iso2":"FR","risk_score":0.10,"risk_tier":"LOW","fatf_status":"NORMAL","un_sanctions":False,"eu_sanctions":False,"ofac_sanctions":False},
    {"country":"SINGAPORE","iso2":"SG","risk_score":0.10,"risk_tier":"LOW","fatf_status":"NORMAL","un_sanctions":False,"eu_sanctions":False,"ofac_sanctions":False},
    {"country":"JAPAN","iso2":"JP","risk_score":0.10,"risk_tier":"LOW","fatf_status":"NORMAL","un_sanctions":False,"eu_sanctions":False,"ofac_sanctions":False},
    {"country":"UNKNOWN","iso2":"XX","risk_score":0.80,"risk_tier":"HIGH","fatf_status":"UNKNOWN","un_sanctions":False,"eu_sanctions":False,"ofac_sanctions":False},
]

_BY_NAME = {r["country"]: r for r in _REGISTER}
_BY_ISO2 = {r["iso2"]:    r for r in _REGISTER}

CTRY_IND_FIELDS = ["nationality","country_of_birth","passport_country","country_of_residence"]
CTRY_ENT_FIELDS = ["incorporation_country","operating_country","branch_country"]

def get_country_risk(country: str) -> dict:
    c = country.strip().upper()
    if c in _BY_NAME: return _BY_NAME[c]
    if c in _BY_ISO2: return _BY_ISO2[c]
    return {"country":c,"iso2":"??","risk_score":0.50,"risk_tier":"MEDIUM","fatf_status":"UNKNOWN","un_sanctions":False,"eu_sanctions":False,"ofac_sanctions":False}

def generate() -> pd.DataFrame:
    df = pd.DataFrame(_REGISTER)
    df["dataset_type"] = "COUNTRY_RISK"
    df["entity_id"]    = [f"CTRY_{r['iso2']}" for r in _REGISTER]
    return df
