from config.job_types import get_job
from dataclasses import dataclass
from typing import Literal
REVIEW_BAND = 0.08
AlertDecisionType = Literal["ALERT","REVIEW","NO_ALERT"]
@dataclass
class AlertDecision:
    decision: AlertDecisionType
    score: float
    threshold: float
    job_code: str
    reason: str
    escalate: bool

def apply_threshold(score, job_code, threshold=None):
    if threshold is None: threshold = get_job(job_code).threshold
    score = round(float(score), 4)
    in_band = abs(score - threshold) <= REVIEW_BAND
    if score >= threshold: return AlertDecision("ALERT",score,threshold,job_code,f"Score {score:.4f} >= {threshold:.4f}",in_band)
    if in_band: return AlertDecision("REVIEW",score,threshold,job_code,f"In review band {threshold:.4f}",True)
    return AlertDecision("NO_ALERT",score,threshold,job_code,f"Score {score:.4f} < {threshold:.4f}",False)
def bulk_apply(scores, job_code): return [apply_threshold(s, job_code) for s in scores]
