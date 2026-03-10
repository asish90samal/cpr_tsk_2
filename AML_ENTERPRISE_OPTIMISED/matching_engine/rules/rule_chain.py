"""
matching_engine/rules/rule_chain.py
═════════════════════════════════════
RuleChain — orchestrates atomic rules in priority order for a single
input-vs-candidate pair.

Design:
  • Each rule runs independently and produces a NameMatchResult.
  • The chain returns the result with the highest score.
  • Which rules run is controlled by the per-job RuleConfig from config/rule_config.py.
  • Each NameMatchResult carries the OWS rule code so analysts see WHY it matched.

OWS rule priority order (highest confidence first):
  I010O > I080O > I020O > I030O > I040O > I050O > I070O > I060O
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import pandas as pd

from matching_engine.rules.atomic_rules import (
    rule_I010O_exact,
    rule_I020O_fuzzy,
    rule_I030O_reversed,
    rule_I040O_given_family,
    rule_I050O_abbreviated,
    rule_I060O_phonetic,
    rule_I070O_transliteration,
    rule_I080O_dob_cluster,
    best_alias_score,
)
from config.rule_config import RuleConfig, get_rule_config


# ══════════════════════════════════════════════════════════════════════════════
#  Result container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NameMatchResult:
    """
    The output of one RuleChain run: the best-scoring rule's result.
    """
    score:          float           # composite or rule-specific score [0–1]
    rule_code:      str             # OWS rule code that produced best score  e.g. "I020O"
    rule_label:     str             # human label  e.g. "Fuzzy composite match"
    matched_name:   str             # which name / alias produced the score
    matched_on_alias: bool
    all_rule_scores: dict[str, float] = field(default_factory=dict)  # every rule's score for audit


# ══════════════════════════════════════════════════════════════════════════════
#  Rule metadata registry
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _RuleMeta:
    code:        str
    label:       str
    config_key:  str     # key in RuleConfig dict
    needs_dob:   bool = False
    needs_split: bool = False


_ALL_RULES: list[_RuleMeta] = [
    _RuleMeta("I010O", "Exact name match",               "I010O_EXACT"),
    _RuleMeta("I020O", "Fuzzy composite match",          "I020O_FUZZY"),
    _RuleMeta("I030O", "Reversed token order",           "I030O_REVERSED"),
    _RuleMeta("I040O", "Given/family name split match",  "I040O_GIVEN_FAMILY",  needs_split=True),
    _RuleMeta("I050O", "Abbreviated given name",         "I050O_ABBREVIATED"),
    _RuleMeta("I060O", "Phonetic match",                 "I060O_PHONETIC"),
    _RuleMeta("I070O", "Transliteration match",          "I070O_TRANSLITERATION"),
    _RuleMeta("I080O", "DOB cluster",                    "I080O_DOB_CLUSTER",   needs_dob=True),
]


# ══════════════════════════════════════════════════════════════════════════════
#  RuleChain
# ══════════════════════════════════════════════════════════════════════════════

class RuleChain:
    """
    Run the enabled atomic rules for a given job_code and return the best result.

    Usage
    -----
    chain  = RuleChain(job_code="SAN_IND")
    result = chain.run(inp_name="M. ALI", inp_dob="1970-03-15", candidate_row=row)
    """

    def __init__(self, job_code: str):
        self._job_code = job_code.upper()
        self._config: RuleConfig = get_rule_config(self._job_code)

    def run(
        self,
        inp_name: str,
        candidate_row: pd.Series,
        inp_dob: str = "",
        inp_given: str = "",
        inp_family: str = "",
    ) -> NameMatchResult:
        """
        Run all enabled rules against a single candidate row (primary + aliases).

        Parameters
        ----------
        inp_name       : Normalised query name
        candidate_row  : One row from a screened dataset
        inp_dob        : Query DOB string (YYYY-MM-DD or partial)
        inp_given      : Explicit given name field (optional)
        inp_family     : Explicit family name field (optional)

        Returns
        -------
        NameMatchResult with the highest-scoring rule and all rule scores for audit.
        """
        all_scores: dict[str, float] = {}
        best_score  = 0.0
        best_code   = "I020O"       # default fallback
        best_label  = "Fuzzy composite match"
        best_name   = str(candidate_row.get("primary_name", ""))
        best_alias  = False

        cand_dob = str(candidate_row.get("dob", ""))

        for meta in _ALL_RULES:
            # ── skip if disabled for this job ──────────────────────────────
            if not self._config.enabled(meta.config_key):
                all_scores[meta.code] = 0.0
                continue

            # ── dispatch to correct atomic rule ───────────────────────────
            if meta.code == "I010O":
                s, mn, ma = best_alias_score(rule_I010O_exact, inp_name, candidate_row)

            elif meta.code == "I020O":
                s, mn, ma = best_alias_score(rule_I020O_fuzzy, inp_name, candidate_row)

            elif meta.code == "I030O":
                s, mn, ma = best_alias_score(rule_I030O_reversed, inp_name, candidate_row)

            elif meta.code == "I040O":
                # Given/family rule — pass explicit fields if provided
                def _gf_rule(inp: str, cnd: str) -> float:
                    return rule_I040O_given_family(
                        inp_name=inp,
                        cand_name=cnd,
                        inp_given=inp_given,
                        inp_family=inp_family,
                    )
                s, mn, ma = best_alias_score(_gf_rule, inp_name, candidate_row)

            elif meta.code == "I050O":
                s, mn, ma = best_alias_score(rule_I050O_abbreviated, inp_name, candidate_row)

            elif meta.code == "I060O":
                s, mn, ma = best_alias_score(rule_I060O_phonetic, inp_name, candidate_row)

            elif meta.code == "I070O":
                s, mn, ma = best_alias_score(rule_I070O_transliteration, inp_name, candidate_row)

            elif meta.code == "I080O":
                # DOB cluster — needs DOB forwarded to alias wrapper
                def _dob_rule(inp: str, cnd: str,
                              _idob=inp_dob, _cdob=cand_dob) -> float:
                    return rule_I080O_dob_cluster(inp, cnd, _idob, _cdob)
                s, mn, ma = best_alias_score(_dob_rule, inp_name, candidate_row)

            else:
                s, mn, ma = 0.0, str(candidate_row.get("primary_name", "")), False

            all_scores[meta.code] = s

            if s > best_score:
                best_score = s
                best_code  = meta.code
                best_label = meta.label
                best_name  = mn
                best_alias = ma

        return NameMatchResult(
            score=round(best_score, 4),
            rule_code=best_code,
            rule_label=best_label,
            matched_name=best_name,
            matched_on_alias=best_alias,
            all_rule_scores=all_scores,
        )
