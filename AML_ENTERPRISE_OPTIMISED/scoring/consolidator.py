from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any
from matching_engine.rules.base_rule import RuleResult
from policy_engine.thresholds import apply_threshold, AlertDecision

MULTI_LIST_MULTIPLIER = 1.15

@dataclass
class DatasetHit:
    dataset_code: str
    rule_name: str
    entity_id: Any
    primary_name: str
    matched_name: str
    name_score: float
    rule_score: float
    auto_alert: bool
    review_flag: bool
    matched_on_alias: bool
    match_details: dict = field(default_factory=dict)
    # OWS extended fields
    rule_code:      str = "I020O"
    rule_label:     str = "Fuzzy composite match"
    risk_score:     int = 0
    priority_score: int = 0
    all_rule_scores: dict = field(default_factory=dict)

@dataclass
class ConsolidatedResult:
    input_name: str
    input_entity_type: str
    job_code: str
    hits: list
    top_hit: Any
    consolidated_score: float
    list_hit_count: int
    auto_alert: bool
    review_flag: bool
    decision: Any
    dataset_summary: dict

def consolidate(input_name, input_entity_type, job_code, all_results, score_threshold=0.30):
    hits = []
    auto_alert = False
    review_flag = False
    for r in all_results:
        if r.auto_alert or r.review_flag or r.rule_score >= score_threshold:
            h = DatasetHit(dataset_code=r.dataset_code, rule_name=r.rule_name,
                entity_id=r.entity_id, primary_name=r.primary_name, matched_name=r.matched_name,
                name_score=r.name_score, rule_score=r.rule_score, auto_alert=r.auto_alert,
                review_flag=r.review_flag, matched_on_alias=r.matched_on_alias, match_details=r.match_details,
                rule_code=getattr(r,"rule_code","I020O"), rule_label=getattr(r,"rule_label","Fuzzy composite match"),
                risk_score=getattr(r,"risk_score",0), priority_score=getattr(r,"priority_score",0),
                all_rule_scores=getattr(r,"all_rule_scores",{}))
            hits.append(h)
            if r.auto_alert: auto_alert = True
            if r.review_flag: review_flag = True
    ds_best = defaultdict(float)
    for h in hits:
        if h.rule_score > ds_best[h.dataset_code]: ds_best[h.dataset_code] = h.rule_score
    n_ds = len(ds_best)
    if not hits:
        score = 0.0; top = None
    else:
        best = max(h.rule_score for h in hits)
        boost = MULTI_LIST_MULTIPLIER ** max(0, n_ds - 1)
        score = min(1.0, round(best * boost, 4))
        top = max(hits, key=lambda h: h.rule_score)
    if auto_alert:
        dec = AlertDecision("ALERT", score, 0.0, job_code, "Auto-alert: exact ID match", False)
    else:
        dec = apply_threshold(score, job_code)
    return ConsolidatedResult(input_name=input_name, input_entity_type=input_entity_type,
        job_code=job_code, hits=hits, top_hit=top, consolidated_score=score,
        list_hit_count=n_ds, auto_alert=auto_alert, review_flag=review_flag,
        decision=dec, dataset_summary=dict(ds_best))
