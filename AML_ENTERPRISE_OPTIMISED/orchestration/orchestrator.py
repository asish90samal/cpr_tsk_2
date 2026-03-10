"""
orchestration/orchestrator.py
==============================
ScreeningOrchestrator — coordinates all 10 job types.

For NAME-MATCH jobs (SAN/SCION/PEP/NNS):
  1. Route -> get job config -> get datasets
  2. InvertedIndex query per dataset -> candidates
  3. Apply per-dataset MatchingRule -> RuleResults
  4. Consolidate -> ConsolidatedResult
  5. Audit + Alert

For CTRY jobs:
  1. CtryRule.screen(input_record, entity_type) -> CtryRuleResult
  2. Map CtryRuleResult to AlertDecision
  3. Audit + Alert
"""
from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from config.job_types import get_job, JOB_REGISTRY
from blocking_engine.inverted_index import MultiDatasetIndex
from matching_engine.rules.base_rule import MatchingRuleEngine, CtryRuleResult
from routing_engine.router import route
from scoring.consolidator import consolidate, ConsolidatedResult
from policy_engine.thresholds import apply_threshold, AlertDecision
from governance.audit import log_decision
from workflow.alert import create_alert
from etl_layer.normalization import normalize_name


@dataclass
class ScreeningInput:
    name:           str
    entity_type:    str = "INDIVIDUAL"
    job_type:       str = "SAN"
    dob:            str = ""
    country:        str = ""
    id_number:      str = ""
    account_number: str = ""
    reference_number: str = ""
    case_id:        str = ""
    # CTRY-specific country fields
    nationality:    str = ""
    country_of_birth: str = ""
    passport_country: str = ""
    country_of_residence: str = ""
    incorporation_country: str = ""
    operating_country: str = ""
    branch_country: str = ""
    metadata:       dict = None

    def __post_init__(self):
        if self.metadata is None: self.metadata = {}
        self.name        = normalize_name(self.name)
        self.entity_type = self.entity_type.strip().upper()

    def get_identifier(self):
        for v in [self.id_number, self.account_number, self.reference_number, self.case_id]:
            if v and v.strip(): return v.strip().upper()
        return None

    def to_rule_dict(self):
        return {"name": self.name, "entity_type": self.entity_type,
                "dob": self.dob, "country": self.country or self.nationality}


@dataclass
class ScreeningOutput:
    input:              ScreeningInput
    job_code:           str
    result_type:        str           # "NAME_MATCH" or "COUNTRY_LOOKUP"
    overall_decision:   str           # ALERT/REVIEW/NO_ALERT/ERROR
    overall_score:      float
    auto_alert:         bool
    review_flag:        bool
    datasets_screened:  list
    latency_ms:         float
    alert_ids:          list
    consolidated:       Any = None    # ConsolidatedResult for name-match jobs
    ctry_result:        Any = None    # CtryRuleResult for CTRY jobs


class ScreeningOrchestrator:

    def __init__(self, registry, multi_index: MultiDatasetIndex,
                 rule_engine: MatchingRuleEngine,
                 score_threshold: float = 0.30,
                 max_candidates: int = 500):
        self._registry  = registry
        self._multi_idx = multi_index
        self._rules     = rule_engine
        self._threshold = score_threshold
        self._max_cands = max_candidates

    def screen(self, inp: ScreeningInput) -> ScreeningOutput:
        t0 = time.perf_counter()
        routing  = route(inp.job_type, inp.entity_type, inp.country or inp.nationality)
        job_code = routing.job_code
        job      = get_job(job_code)

        # ── CTRY job: pure country lookup ──────────────────────────────
        if job.is_country_lookup:
            return self._screen_ctry(inp, job_code, t0)

        # ── Name-match jobs: SAN / SCION / PEP / NNS ──────────────────
        return self._screen_name_match(inp, job_code, job, t0)

    # ── CTRY screening ────────────────────────────────────────────────────

    def _screen_ctry(self, inp, job_code, t0):
        ctry_res = self._rules.ctry_rule.screen(
            input_record=vars(inp),
            entity_type=inp.entity_type,
            alert_threshold=get_job(job_code).threshold,
        )
        score = ctry_res.highest_risk_score
        if ctry_res.auto_alert:
            decision = "ALERT"
        elif ctry_res.review_flag:
            decision = "REVIEW"
        else:
            decision = "NO_ALERT"

        dec_obj = apply_threshold(score, job_code)

        alert_ids = []
        if decision in ("ALERT","REVIEW"):
            alert = create_alert(
                entity_id=ctry_res.highest_risk_country,
                input_name=inp.name,
                matched_name=ctry_res.highest_risk_country,
                decision_obj=dec_obj,
            )
            if alert: alert_ids.append(alert.alert_id)

        return ScreeningOutput(
            input=inp, job_code=job_code, result_type="COUNTRY_LOOKUP",
            overall_decision=decision, overall_score=round(score, 4),
            auto_alert=ctry_res.auto_alert, review_flag=ctry_res.review_flag,
            datasets_screened=["COUNTRY_RISK"],
            latency_ms=round((time.perf_counter()-t0)*1000, 2),
            alert_ids=alert_ids, ctry_result=ctry_res,
        )

    # ── Name-match screening ──────────────────────────────────────────────

    def _screen_name_match(self, inp, job_code, job, t0):
        from matching_engine.rules.base_rule import RuleResult
        identifier = inp.get_identifier()
        inp_dict   = inp.to_rule_dict()
        all_results: list[RuleResult] = []
        datasets_screened: list[str] = []

        for ds_cfg in sorted(job.datasets, key=lambda x: x.priority):
            for etype in ds_cfg.entity_types:
                candidates, exact_id_hit = self._multi_idx.query_dataset(
                    ds_cfg.code, etype, inp.name, identifier=identifier)
                if candidates.empty: continue
                if len(candidates) > self._max_cands:
                    candidates = candidates.head(self._max_cands)
                datasets_screened.append(ds_cfg.code)
                for _, row in candidates.iterrows():
                    result = self._rules.apply(inp_dict, row, job_code, ds_cfg.code, exact_id_hit)
                    all_results.append(result)

        cr = consolidate(inp.name, inp.entity_type, job_code, all_results, self._threshold)

        alert_ids = []
        if cr.top_hit:
            dec_obj = cr.decision
            log_decision(cr.top_hit.entity_id, inp.name, cr.top_hit.matched_name, dec_obj, "")
            alert = create_alert(cr.top_hit.entity_id, inp.name, cr.top_hit.matched_name, dec_obj)
            if alert: alert_ids.append(alert.alert_id)

        return ScreeningOutput(
            input=inp, job_code=job_code, result_type="NAME_MATCH",
            overall_decision=cr.decision.decision,
            overall_score=cr.consolidated_score,
            auto_alert=cr.auto_alert, review_flag=cr.review_flag,
            datasets_screened=list(set(datasets_screened)),
            latency_ms=round((time.perf_counter()-t0)*1000, 2),
            alert_ids=alert_ids, consolidated=cr,
        )

    # ── Batch ─────────────────────────────────────────────────────────────

    def screen_batch(self, inputs, max_workers=4, verbose=True):
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self.screen, inp): i for i, inp in enumerate(inputs)}
            done = 0
            for future in as_completed(futures):
                i = futures[future]
                try:    results[i] = future.result()
                except Exception as e:
                    results[i] = ScreeningOutput(
                        input=inputs[i], job_code="ERROR", result_type="ERROR",
                        overall_decision="ERROR", overall_score=0.0,
                        auto_alert=False, review_flag=False,
                        datasets_screened=[], latency_ms=0.0, alert_ids=[])
                done += 1
                if verbose and done % 100 == 0:
                    print(f"[Orchestrator] {done}/{len(inputs)} screened...")
        if verbose: print(f"[Orchestrator] Batch complete: {len(inputs)} records")
        return [results[i] for i in range(len(inputs))]

    def batch_summary(self, outputs):
        total = len(outputs)
        return {
            "total":          total,
            "alerts":         sum(1 for o in outputs if o.overall_decision=="ALERT"),
            "reviews":        sum(1 for o in outputs if o.overall_decision=="REVIEW"),
            "no_alerts":      sum(1 for o in outputs if o.overall_decision=="NO_ALERT"),
            "auto_alerts":    sum(1 for o in outputs if o.auto_alert),
            "alert_rate_pct": round(sum(1 for o in outputs if o.overall_decision=="ALERT")/max(total,1)*100,2),
            "avg_latency_ms": round(sum(o.latency_ms for o in outputs)/max(total,1),2),
        }
