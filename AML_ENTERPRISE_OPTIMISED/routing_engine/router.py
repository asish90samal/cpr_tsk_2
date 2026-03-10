from __future__ import annotations
from dataclasses import dataclass
from config.job_types import get_job

_HR = {"IRAN","RUSSIA","NORTH KOREA","SYRIA","MYANMAR","BELARUS","VENEZUELA","CUBA","SUDAN","SOMALIA","AFGHANISTAN","LIBYA","IRAQ","YEMEN","ZIMBABWE","UNKNOWN"}
_MR = {"NIGERIA","PAKISTAN","UKRAINE","KENYA","ETHIOPIA","BANGLADESH","CAMBODIA","CHINA","TURKEY","INDIA","BRAZIL"}

@dataclass
class RoutingDecision:
    job_code: str
    entity_type: str
    risk_tier: str
    reason: str
    threshold: float

def route(job_type, entity_type, country=""):
    etype = entity_type.strip().upper()
    jt = job_type.strip().upper()
    suffix = "IND" if etype == "INDIVIDUAL" else "ENT"
    job_code = f"{jt}_{suffix}"
    job = get_job(job_code)
    c = country.strip().upper()
    if c in _HR or not c: tier,reason="HIGH",f"Country {c!r} high-risk"
    elif c in _MR:        tier,reason="MEDIUM",f"Country {c!r} medium-risk"
    else:                 tier,reason="LOW",f"Country {c!r} low-risk"
    return RoutingDecision(job_code=job_code, entity_type=etype, risk_tier=tier, reason=reason, threshold=job.threshold)
