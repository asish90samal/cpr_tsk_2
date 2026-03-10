"""
matching_engine/rules/base_rule.py
===================================
Per-job matching rules for all 10 jobs.

WHAT CHANGED vs original:
  1. RuleResult gains four new fields:
       rule_code, rule_label, risk_score (0-100), priority_score (0-100),
       all_rule_scores (audit dict of every atomic rule's score)
  2. All name-match rules delegate to RuleChain (atomic rules engine)
  3. _composite / _best_alias_score kept for CtryRule / backward compat
  4. SanctionIndividualDOBRule removed — I080O_DOB_CLUSTER is inside RuleChain
  5. MatchingRuleEngine.apply() populates risk_score + priority_score via risk_scorer

Rule map (unchanged jobs, upgraded internals):
  SAN_IND   -> SanctionIndividualRule
  SAN_ENT   -> SanctionEntityRule
  SCION_*   -> ScionRule
  PEP_IND   -> PEPIndividualRule
  PEP_ENT   -> PEPEntityRule
  NNS_* (articles)   -> NNSArticleRule
  NNS_* (structured) -> NNSStructuredRule
  CTRY_*    -> CtryRule  (unchanged)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import pandas as pd

try:
    from rapidfuzz import fuzz, distance as _rfdist
    _RAPIDFUZZ_OK = True
except ImportError:
    _RAPIDFUZZ_OK = False
    fuzz = None
    _rfdist = None


# ── backward-compat helpers ────────────────────────────────────────────────

def _composite(a: str, b: str) -> float:
    a, b = a.upper().strip(), b.upper().strip()
    if not a or not b: return 0.0
    if not _RAPIDFUZZ_OK:
        if a == b: return 1.0
        if a in b or b in a: return 0.75
        return 0.0
    ts  = fuzz.token_sort_ratio(a, b) / 100.0
    tse = fuzz.token_set_ratio(a, b)  / 100.0
    pr  = fuzz.partial_ratio(a, b)    / 100.0
    jw  = _rfdist.JaroWinkler.normalized_similarity(a, b)
    return round(0.30*ts + 0.30*tse + 0.20*pr + 0.20*jw, 4)


def _best_alias_score(input_name: str, row: pd.Series) -> tuple:
    names = [(str(row.get("primary_name", "")), False)]
    als = row.get("aliases", "")
    if als:
        for a in str(als).split("|"):
            a = a.strip()
            if a: names.append((a, True))
    best, bname, is_al = 0.0, str(row.get("primary_name", "")), False
    for name, alias_flag in names:
        if input_name.upper() == name.upper():
            return 1.0, name, alias_flag
        s = _composite(input_name, name)
        if s > best:
            best, bname, is_al = s, name, alias_flag
    return best, bname, is_al


_SUFFIXES = {"LTD","LIMITED","LLC","INC","CORP","CORPORATION","PLC","JSC","OJSC","PJSC",
             "GMBH","AG","SA","BV","NV","SRL","CO","COMPANY","GROUP","HOLDINGS","TRADING"}

def _strip_suffix(name: str) -> str:
    toks = name.upper().strip().split()
    return " ".join(toks[:-1]) if toks and toks[-1] in _SUFFIXES else name.upper().strip()


# ══════════════════════════════════════════════════════════════════════════
#  RuleResult — extended with OWS output fields
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class RuleResult:
    entity_id:          Any
    primary_name:       str
    matched_name:       str
    dataset_code:       str
    job_code:           str
    rule_name:          str
    rule_code:          str   = "I020O"
    rule_label:         str   = "Fuzzy composite match"
    name_score:         float = 0.0
    rule_score:         float = 0.0
    risk_score:         int   = 0
    priority_score:     int   = 0
    auto_alert:         bool  = False
    review_flag:        bool  = False
    matched_on_alias:   bool  = False
    match_details:      dict  = field(default_factory=dict)
    all_rule_scores:    dict  = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════
#  Abstract base
# ══════════════════════════════════════════════════════════════════════════

class BaseMatchingRule(ABC):
    @property
    @abstractmethod
    def rule_name(self) -> str: ...
    @property
    @abstractmethod
    def job_codes(self) -> list: ...
    @property
    @abstractmethod
    def dataset_codes(self) -> list: ...
    @abstractmethod
    def score(self, inp: dict, row: pd.Series, exact_id_hit: bool = False) -> RuleResult: ...

    def _chain_score(self, job_code: str, inp: dict, row: pd.Series):
        try:
            from matching_engine.rules.rule_chain import RuleChain
            chain = RuleChain(job_code)
            return chain.run(
                inp_name=str(inp.get("name", "")),
                candidate_row=row,
                inp_dob=str(inp.get("dob", "")),
                inp_given=str(inp.get("given_name", "")),
                inp_family=str(inp.get("family_name", "")),
            )
        except Exception:
            ns, mn, ia = _best_alias_score(str(inp.get("name", "")), row)
            class _Fb:
                score=ns; matched_name=mn; matched_on_alias=ia
                rule_code="I020O"; rule_label="Fuzzy composite match"
                all_rule_scores: dict = {}
            return _Fb()

    def _risk_scores(self, match_score: float, row: pd.Series,
                     inp: dict, entity_type: str, is_pep: bool = False) -> tuple:
        try:
            from scoring.risk_scorer import score_record
            r = score_record(candidate_row=row, input_record=inp,
                             entity_type=entity_type, match_score=match_score, is_pep=is_pep)
            return r.blended_risk_score, r.priority_score
        except Exception:
            return 50, round(match_score * 100)


# ══════════════════════════════════════════════════════════════════════════
#  SAN Individual
# ══════════════════════════════════════════════════════════════════════════

class SanctionIndividualRule(BaseMatchingRule):
    rule_name    = "SAN_INDIVIDUAL"
    job_codes    = ["SAN_IND"]
    dataset_codes = ["OFAC_SDN","UN_CONSOLIDATED","EU_SANCTIONS","HM_TREASURY",
                     "INTERPOL_RED","WORLD_CHECK","DOW_JONES","COMPLY_ADVANTAGE","ACCUITY_FIRCO"]

    def score(self, inp: dict, row: pd.Series, exact_id_hit: bool = False) -> RuleResult:
        eid=row.get("entity_id",""); pname=str(row.get("primary_name","")); ds=str(row.get("dataset_type",""))
        if exact_id_hit:
            risk,pri=self._risk_scores(1.0,row,inp,"INDIVIDUAL")
            return RuleResult(entity_id=eid,primary_name=pname,matched_name=pname,dataset_code=ds,
                job_code="SAN_IND",rule_name=self.rule_name,rule_code="I010O",rule_label="Exact ID match",
                name_score=1.0,rule_score=1.0,risk_score=risk,priority_score=pri,
                auto_alert=True,review_flag=False,matched_on_alias=False,match_details={"trigger":"exact_id_match"})
        cr=self._chain_score("SAN_IND",inp,row)
        risk,pri=self._risk_scores(cr.score,row,inp,"INDIVIDUAL")
        return RuleResult(entity_id=eid,primary_name=pname,matched_name=cr.matched_name,dataset_code=ds,
            job_code="SAN_IND",rule_name=self.rule_name,rule_code=cr.rule_code,rule_label=cr.rule_label,
            name_score=cr.score,rule_score=cr.score,risk_score=risk,priority_score=pri,
            auto_alert=False,review_flag=False,matched_on_alias=cr.matched_on_alias,
            match_details={"chain_fired":cr.rule_code},all_rule_scores=cr.all_rule_scores)


# ══════════════════════════════════════════════════════════════════════════
#  SAN Entity
# ══════════════════════════════════════════════════════════════════════════

class SanctionEntityRule(BaseMatchingRule):
    rule_name    = "SAN_ENTITY"
    job_codes    = ["SAN_ENT"]
    dataset_codes = ["OFAC_SDN","UN_CONSOLIDATED","EU_SANCTIONS","HM_TREASURY",
                     "WORLD_CHECK","DOW_JONES","COMPLY_ADVANTAGE","ACCUITY_FIRCO"]

    def score(self, inp: dict, row: pd.Series, exact_id_hit: bool = False) -> RuleResult:
        eid=row.get("entity_id",""); pname=str(row.get("primary_name","")); ds=str(row.get("dataset_type",""))
        if exact_id_hit:
            risk,pri=self._risk_scores(1.0,row,inp,"ENTITY")
            return RuleResult(entity_id=eid,primary_name=pname,matched_name=pname,dataset_code=ds,
                job_code="SAN_ENT",rule_name=self.rule_name,rule_code="I010O",rule_label="Exact ID match",
                name_score=1.0,rule_score=1.0,risk_score=risk,priority_score=pri,
                auto_alert=True,review_flag=False,matched_on_alias=False,match_details={"trigger":"exact_id_match"})
        cr=self._chain_score("SAN_ENT",inp,row)
        inp_name=str(inp.get("name","")).upper()
        s_stripped=_composite(_strip_suffix(inp_name),_strip_suffix(pname))
        nws=str(row.get("name_without_suffix",""))
        s_nws=_composite(_strip_suffix(inp_name),nws) if nws else 0.0
        final=max(cr.score,s_stripped,s_nws)
        mn=cr.matched_name; ia=cr.matched_on_alias
        if s_stripped>cr.score: mn,ia=_strip_suffix(pname),True
        if s_nws>s_stripped and s_nws>cr.score: mn,ia=nws,True
        risk,pri=self._risk_scores(final,row,inp,"ENTITY")
        return RuleResult(entity_id=eid,primary_name=pname,matched_name=mn,dataset_code=ds,
            job_code="SAN_ENT",rule_name=self.rule_name,rule_code=cr.rule_code,rule_label=cr.rule_label,
            name_score=cr.score,rule_score=round(final,4),risk_score=risk,priority_score=pri,
            auto_alert=False,review_flag=False,matched_on_alias=ia,
            match_details={"score_stripped":s_stripped,"score_nws":s_nws,"chain_fired":cr.rule_code},
            all_rule_scores=cr.all_rule_scores)


# ══════════════════════════════════════════════════════════════════════════
#  SCION
# ══════════════════════════════════════════════════════════════════════════

class ScionRule(BaseMatchingRule):
    rule_name    = "SCION"
    job_codes    = ["SCION_IND","SCION_ENT"]
    dataset_codes = ["SCION"]

    def score(self, inp: dict, row: pd.Series, exact_id_hit: bool = False) -> RuleResult:
        eid=row.get("entity_id",""); pname=str(row.get("primary_name",""))
        etype=str(inp.get("entity_type","INDIVIDUAL")); jc="SCION_IND" if etype=="INDIVIDUAL" else "SCION_ENT"
        src=str(row.get("scion_source",""))
        if exact_id_hit:
            risk,pri=self._risk_scores(1.0,row,inp,etype)
            return RuleResult(entity_id=eid,primary_name=pname,matched_name=pname,dataset_code="SCION",
                job_code=jc,rule_name=self.rule_name,rule_code="I010O",rule_label="Exact account/reference match",
                name_score=1.0,rule_score=1.0,risk_score=risk,priority_score=pri,
                auto_alert=True,review_flag=False,matched_on_alias=False,
                match_details={"trigger":"account_reference_match","scion_source":src})
        cr=self._chain_score(jc,inp,row)
        status=str(row.get("watchlist_status","")).upper()
        boost={"BLACKLIST":0.10,"GREYLIST":0.05}.get(status,0.0)
        rs=min(1.0,round(cr.score+boost,4))
        risk,pri=self._risk_scores(rs,row,inp,etype)
        return RuleResult(entity_id=eid,primary_name=pname,matched_name=cr.matched_name,dataset_code="SCION",
            job_code=jc,rule_name=self.rule_name,rule_code=cr.rule_code,rule_label=cr.rule_label,
            name_score=cr.score,rule_score=rs,risk_score=risk,priority_score=pri,
            auto_alert=False,review_flag=False,matched_on_alias=cr.matched_on_alias,
            match_details={"watchlist_status":status,"status_boost":boost,
                           "chain_fired":cr.rule_code,"scion_source":src},
            all_rule_scores=cr.all_rule_scores)


# ══════════════════════════════════════════════════════════════════════════
#  PEP Individual
# ══════════════════════════════════════════════════════════════════════════

class PEPIndividualRule(BaseMatchingRule):
    rule_name    = "PEP_INDIVIDUAL"
    job_codes    = ["PEP_IND"]
    dataset_codes = ["PEP_DATABASE"]

    def score(self, inp: dict, row: pd.Series, exact_id_hit: bool = False) -> RuleResult:
        eid=row.get("entity_id",""); pname=str(row.get("primary_name",""))
        if str(row.get("entity_type",""))=="ENTITY":
            return RuleResult(entity_id=eid,primary_name=pname,matched_name=pname,
                dataset_code="PEP_DATABASE",job_code="PEP_IND",rule_name=self.rule_name,
                rule_code="I020O",name_score=0.0,rule_score=0.0,risk_score=0,priority_score=0,
                auto_alert=False,review_flag=False,matched_on_alias=False,match_details={"skip":"entity_row"})
        cr=self._chain_score("PEP_IND",inp,row)
        tier=int(row.get("pep_tier",3)); active=bool(row.get("is_active",False))
        tier_b={1:0.08,2:0.04,3:0.00}[tier]
        if active: tier_b+=0.03
        ctry_b=0.05 if (str(inp.get("country","")).upper()==str(row.get("country_of_office","")).upper() and inp.get("country","")) else 0.0
        rs=min(1.0,round(cr.score+tier_b+ctry_b,4))
        risk,pri=self._risk_scores(rs,row,inp,"INDIVIDUAL",is_pep=True)
        return RuleResult(entity_id=eid,primary_name=pname,matched_name=cr.matched_name,
            dataset_code="PEP_DATABASE",job_code="PEP_IND",rule_name=self.rule_name,
            rule_code=cr.rule_code,rule_label=cr.rule_label,
            name_score=cr.score,rule_score=rs,risk_score=risk,priority_score=pri,
            auto_alert=False,review_flag=False,matched_on_alias=cr.matched_on_alias,
            match_details={"pep_tier":tier,"is_active":active,"tier_boost":tier_b,
                           "country_boost":ctry_b,"political_role":str(row.get("political_role","")),
                           "chain_fired":cr.rule_code},
            all_rule_scores=cr.all_rule_scores)


# ══════════════════════════════════════════════════════════════════════════
#  PEP Entity
# ══════════════════════════════════════════════════════════════════════════

class PEPEntityRule(BaseMatchingRule):
    rule_name    = "PEP_ENTITY"
    job_codes    = ["PEP_ENT"]
    dataset_codes = ["PEP_DATABASE"]

    def score(self, inp: dict, row: pd.Series, exact_id_hit: bool = False) -> RuleResult:
        eid=row.get("entity_id",""); pname=str(row.get("primary_name",""))
        if str(row.get("entity_type",""))!="ENTITY":
            return RuleResult(entity_id=eid,primary_name=pname,matched_name=pname,
                dataset_code="PEP_DATABASE",job_code="PEP_ENT",rule_name=self.rule_name,
                rule_code="I020O",name_score=0.0,rule_score=0.0,risk_score=0,priority_score=0,
                auto_alert=False,review_flag=False,matched_on_alias=False,match_details={"skip":"individual_row"})
        cr=self._chain_score("PEP_ENT",inp,row)
        bop=str(row.get("beneficial_owner_pep_id","")); has_bop=bool(bop and bop.strip())
        boost=0.05 if bool(row.get("is_active",True)) else 0.0
        rs=min(1.0,round(cr.score+boost,4))
        risk,pri=self._risk_scores(rs,row,inp,"ENTITY",is_pep=True)
        return RuleResult(entity_id=eid,primary_name=pname,matched_name=cr.matched_name,
            dataset_code="PEP_DATABASE",job_code="PEP_ENT",rule_name=self.rule_name,
            rule_code=cr.rule_code,rule_label=cr.rule_label,
            name_score=cr.score,rule_score=rs,risk_score=risk,priority_score=pri,
            auto_alert=False,review_flag=has_bop,matched_on_alias=cr.matched_on_alias,
            match_details={"beneficial_owner_pep_id":bop,"has_beneficial_owner_link":has_bop,
                           "chain_fired":cr.rule_code},
            all_rule_scores=cr.all_rule_scores)


# ══════════════════════════════════════════════════════════════════════════
#  NNS Article
# ══════════════════════════════════════════════════════════════════════════

class NNSArticleRule(BaseMatchingRule):
    rule_name    = "NNS_ARTICLES"
    job_codes    = ["NNS_IND","NNS_ENT"]
    dataset_codes = ["NNS_ARTICLES"]

    def __init__(self, use_llm: bool = True):
        """
        Parameters
        ----------
        use_llm : Enable LLM article analysis (default True).
                  Set False in unit tests or when API key is not available.
        """
        self._use_llm = use_llm

    def score(self, inp: dict, row: pd.Series, exact_id_hit: bool = False) -> RuleResult:
        eid=row.get("entity_id",""); pname=str(row.get("primary_name",""))
        jc="NNS_IND" if str(inp.get("entity_type","INDIVIDUAL"))=="INDIVIDUAL" else "NNS_ENT"
        cr=self._chain_score(jc,inp,row)
        rw=float(row.get("recency_weight",0.5)); sev=float(row.get("category_severity",0.5))

        # ── Base rule score (name × recency × vendor severity) ────────────
        base_rs=round(cr.score*rw*sev,4)

        # ── LLM article analysis (optional, graceful fallback) ────────────
        llm_analysis = None
        if self._use_llm:
            try:
                from nns_engine.article_analyser import analyse_article
                llm_analysis = analyse_article(
                    inp=inp, row=row,
                    name_score=cr.score,
                    original_rule_score=base_rs,
                )
            except Exception:
                pass  # always fallback — screening must never be blocked

        # Use LLM-adjusted score when available, else base score
        if llm_analysis and llm_analysis.llm_used:
            final_rs = llm_analysis.adjusted_rule_score
            sev_used = llm_analysis.blended_severity_score
            suppress = llm_analysis.should_suppress
        else:
            final_rs = base_rs
            sev_used = sev
            suppress = False

        etype="INDIVIDUAL" if jc=="NNS_IND" else "ENTITY"
        risk,pri=self._risk_scores(final_rs,row,inp,etype)

        # Build LLM detail dict for audit trail
        llm_detail: dict = {}
        if llm_analysis:
            llm_detail = {
                "llm_used":            llm_analysis.llm_used,
                "entity_confirmed":    llm_analysis.entity_confirmed,
                "entity_confidence":   llm_analysis.entity_confidence,
                "disambiguation_note": llm_analysis.disambiguation_note,
                "llm_severity_score":  llm_analysis.llm_severity_score,
                "blended_severity":    llm_analysis.blended_severity_score,
                "severity_rationale":  llm_analysis.severity_rationale,
                "relationships":       llm_analysis.relationships,
                "suppressed":          suppress,
                "llm_latency_ms":      llm_analysis.latency_ms,
                "fallback_reason":     llm_analysis.fallback_reason,
            }

        return RuleResult(entity_id=eid,primary_name=pname,matched_name=cr.matched_name,
            dataset_code="NNS_ARTICLES",job_code=jc,rule_name=self.rule_name,
            rule_code=cr.rule_code,rule_label=cr.rule_label,
            name_score=cr.score,rule_score=final_rs,risk_score=risk,priority_score=pri,
            auto_alert=False,
            # review_flag=True when LLM is uncertain (confidence 0.3–0.6)
            review_flag=(llm_analysis.entity_confidence < 0.60 if llm_analysis and llm_analysis.llm_used else False),
            matched_on_alias=cr.matched_on_alias,
            match_details={
                "recency_weight":rw, "category_severity_vendor":sev,
                "category_severity_used":sev_used,
                "category":str(row.get("category","")),
                "publication_date":str(row.get("publication_date","")),
                "source":str(row.get("source_publication","")),
                "headline":str(row.get("headline_snippet","")),
                "chain_fired":cr.rule_code,
                **llm_detail,
            },
            all_rule_scores=cr.all_rule_scores)


# ══════════════════════════════════════════════════════════════════════════
#  NNS Structured
# ══════════════════════════════════════════════════════════════════════════

class NNSStructuredRule(BaseMatchingRule):
    rule_name    = "NNS_STRUCTURED"
    job_codes    = ["NNS_IND","NNS_ENT"]
    dataset_codes = ["NNS_STRUCTURED"]

    def score(self, inp: dict, row: pd.Series, exact_id_hit: bool = False) -> RuleResult:
        eid=row.get("entity_id",""); pname=str(row.get("primary_name",""))
        jc="NNS_IND" if str(inp.get("entity_type","INDIVIDUAL"))=="INDIVIDUAL" else "NNS_ENT"
        cr=self._chain_score(jc,inp,row)
        sev=float(row.get("category_severity",0.5)); rs=round(cr.score*sev,4)
        etype="INDIVIDUAL" if jc=="NNS_IND" else "ENTITY"
        risk,pri=self._risk_scores(rs,row,inp,etype)
        return RuleResult(entity_id=eid,primary_name=pname,matched_name=cr.matched_name,
            dataset_code="NNS_STRUCTURED",job_code=jc,rule_name=self.rule_name,
            rule_code=cr.rule_code,rule_label=cr.rule_label,
            name_score=cr.score,rule_score=rs,risk_score=risk,priority_score=pri,
            auto_alert=False,review_flag=exact_id_hit,matched_on_alias=cr.matched_on_alias,
            match_details={"case_id":str(row.get("case_id","")),"category":str(row.get("category","")),
                           "category_severity":sev,"country_of_subject":str(row.get("country_of_subject","")),
                           "linked_entity_ids":str(row.get("linked_entity_ids","")),
                           "chain_fired":cr.rule_code},
            all_rule_scores=cr.all_rule_scores)


# ══════════════════════════════════════════════════════════════════════════
#  CTRY Rule — unchanged
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class CtryRuleResult:
    job_code:             str
    entity_type:          str
    checked_fields:       dict
    field_scores:         dict
    highest_risk_score:   float
    highest_risk_country: str
    highest_risk_field:   str
    risk_tier:            str
    fatf_status:          str
    un_sanctions:         bool
    eu_sanctions:         bool
    ofac_sanctions:       bool
    auto_alert:           bool
    review_flag:          bool


class CtryRule:
    rule_name = "CTRY_LOOKUP"

    def screen(self, input_record: dict, entity_type: str = "INDIVIDUAL",
               alert_threshold: float = 0.75, review_band: float = 0.10) -> CtryRuleResult:
        from data_layer.generators.ctry import get_country_risk, CTRY_IND_FIELDS, CTRY_ENT_FIELDS
        fields = CTRY_IND_FIELDS if entity_type == "INDIVIDUAL" else CTRY_ENT_FIELDS
        jc = "CTRY_IND" if entity_type == "INDIVIDUAL" else "CTRY_ENT"
        checked, scores = {}, {}
        for f in fields:
            val = str(input_record.get(f, "")).strip()
            if val:
                rec = get_country_risk(val)
                checked[f] = rec["country"]; scores[f] = rec["risk_score"]
        if not scores:
            rec = get_country_risk("UNKNOWN"); checked["nationality"] = "UNKNOWN"; scores["nationality"] = rec["risk_score"]
        top_field = max(scores, key=lambda k: scores[k])
        top_score = scores[top_field]; top_country = checked[top_field]
        top_rec = get_country_risk(top_country)
        auto_alert  = top_score >= alert_threshold
        review_flag = (not auto_alert) and (top_score >= alert_threshold - review_band)
        return CtryRuleResult(job_code=jc,entity_type=entity_type,checked_fields=checked,
            field_scores=scores,highest_risk_score=top_score,highest_risk_country=top_country,
            highest_risk_field=top_field,risk_tier=top_rec["risk_tier"],fatf_status=top_rec["fatf_status"],
            un_sanctions=top_rec["un_sanctions"],eu_sanctions=top_rec["eu_sanctions"],
            ofac_sanctions=top_rec["ofac_sanctions"],auto_alert=auto_alert,review_flag=review_flag)


# ══════════════════════════════════════════════════════════════════════════
#  Matching Rule Engine
# ══════════════════════════════════════════════════════════════════════════

class MatchingRuleEngine:
    def __init__(self, use_llm: bool = True):
        self._rules = {}
        self.ctry_rule = CtryRule()
        self._register_defaults(use_llm=use_llm)

    def _register_defaults(self, use_llm: bool = True):
        for rule in [SanctionIndividualRule(),SanctionEntityRule(),ScionRule(),
                     PEPIndividualRule(),PEPEntityRule(),
                     NNSArticleRule(use_llm=use_llm),NNSStructuredRule()]:
            for jc in rule.job_codes:
                for ds in rule.dataset_codes:
                    self._rules[(jc, ds)] = rule

    def get_rule(self, job_code: str, dataset_code: str):
        key = (job_code.upper(), dataset_code.upper())
        if key in self._rules: return self._rules[key]
        for (jc, ds), rule in self._rules.items():
            if ds == dataset_code.upper(): return rule
        return SanctionIndividualRule()

    def apply(self, input_record: dict, candidate_row: pd.Series,
              job_code: str, dataset_code: str, exact_id_hit: bool = False) -> RuleResult:
        return self.get_rule(job_code, dataset_code).score(input_record, candidate_row, exact_id_hit=exact_id_hit)

    def list_rules(self):
        return [(jc, ds, r.rule_name) for (jc, ds), r in sorted(self._rules.items())]
