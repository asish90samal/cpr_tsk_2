"""
matcher.py
──────────
Name matching / similarity scoring engine.

Fixes applied vs original:
  1. Aliases are now screened — best score across primary + all aliases returned
  2. Multiple fuzzy algorithms combined (token_sort, token_set, partial_ratio, jaro_winkler)
  3. Exact-match fast-path added
  4. Returns structured MatchResult dict (not just a float)
  5. Batch match helper for vectorised use
"""

from __future__ import annotations
from dataclasses import dataclass, asdict

import pandas as pd
from rapidfuzz import fuzz, distance


@dataclass
class MatchResult:
    entity_id:          int | str
    primary_name:       str
    best_match_name:    str          # which name/alias produced best score
    score_token_sort:   float
    score_token_set:    float
    score_partial:      float
    score_jaro_winkler: float
    composite_score:    float        # weighted combination
    matched_on_alias:   bool


# ── Weights for composite score ────────────────────────────────────────────
_W_TOKEN_SORT:   float = 0.30
_W_TOKEN_SET:    float = 0.30
_W_PARTIAL:      float = 0.20
_W_JARO_WINKLER: float = 0.20


def _score_pair(input_name: str, candidate: str) -> dict[str, float]:
    """Compute all similarity metrics between two normalised name strings."""
    a, b = input_name.upper().strip(), candidate.upper().strip()
    ts  = fuzz.token_sort_ratio(a, b) / 100.0
    tse = fuzz.token_set_ratio(a, b)  / 100.0
    pr  = fuzz.partial_ratio(a, b)    / 100.0
    jw  = distance.JaroWinkler.normalized_similarity(a, b)
    composite = (
        _W_TOKEN_SORT   * ts  +
        _W_TOKEN_SET    * tse +
        _W_PARTIAL      * pr  +
        _W_JARO_WINKLER * jw
    )
    return {
        "score_token_sort":   round(ts,  4),
        "score_token_set":    round(tse, 4),
        "score_partial":      round(pr,  4),
        "score_jaro_winkler": round(jw,  4),
        "composite_score":    round(composite, 4),
    }


def match(input_name: str, candidate_row: pd.Series) -> MatchResult:
    """
    Score input_name against a single candidate row (primary name + all aliases).

    Parameters
    ----------
    input_name     : Normalised input query name
    candidate_row  : A row from the sanctions DataFrame

    Returns
    -------
    MatchResult with best scores and which name produced the best match
    """
    input_upper = input_name.upper().strip()

    # Build list of all names to test: primary + aliases
    names_to_test: list[tuple[str, bool]] = [
        (candidate_row["primary_name"], False)
    ]
    aliases_str = candidate_row.get("aliases", "")
    if aliases_str:
        for alias in str(aliases_str).split("|"):
            a = alias.strip()
            if a:
                names_to_test.append((a, True))

    # Exact match fast-path
    for name, is_alias in names_to_test:
        if input_upper == name.upper():
            return MatchResult(
                entity_id=candidate_row["entity_id"],
                primary_name=candidate_row["primary_name"],
                best_match_name=name,
                score_token_sort=1.0,
                score_token_set=1.0,
                score_partial=1.0,
                score_jaro_winkler=1.0,
                composite_score=1.0,
                matched_on_alias=is_alias,
            )

    # Score all names; keep the one with the highest composite
    best_scores: dict[str, float] = {}
    best_name   = candidate_row["primary_name"]
    best_alias  = False
    best_composite = -1.0

    for name, is_alias in names_to_test:
        scores = _score_pair(input_upper, name)
        if scores["composite_score"] > best_composite:
            best_composite = scores["composite_score"]
            best_scores    = scores
            best_name      = name
            best_alias     = is_alias

    return MatchResult(
        entity_id=candidate_row["entity_id"],
        primary_name=candidate_row["primary_name"],
        best_match_name=best_name,
        matched_on_alias=best_alias,
        **best_scores,
    )


def batch_match(
    input_name: str,
    candidates: pd.DataFrame,
    score_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Match input_name against all rows in candidates DataFrame.

    Parameters
    ----------
    input_name       : Normalised query name
    candidates       : Blocked candidate DataFrame
    score_threshold  : Only return rows with composite_score >= threshold

    Returns
    -------
    pd.DataFrame of MatchResult dicts, sorted by composite_score descending
    """
    results = [asdict(match(input_name, row)) for _, row in candidates.iterrows()]
    result_df = pd.DataFrame(results)
    if score_threshold > 0 and not result_df.empty:
        result_df = result_df[result_df["composite_score"] >= score_threshold]
    return result_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
