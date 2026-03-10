"""
matching_engine/rules/atomic_rules.py
═══════════════════════════════════════
OWS-aligned atomic (discrete, named) match rule functions.

Each function is independently callable, independently reportable, and
independently togglable via rule_config.py.  They map to OWS rule codes:

  I010O  Exact name match
  I020O  Fuzzy composite match  (token_sort + token_set + partial + jaro-winkler)
  I030O  Reversed token order
  I040O  Given name / family name split match
  I050O  Abbreviated given name  (e.g. "M. ALI" → "MOHAMMED ALI")
  I060O  Phonetic match          (double-metaphone per token)
  I070O  Transliteration match   (Arabic / Cyrillic / CJK variants)
  I080O  DOB cluster             (name must reach minimum band + DOB corroborates)

All functions return a float in [0.0, 1.0].
0.0 means the rule did not fire / produced no match.
"""
from __future__ import annotations

# ── optional rapid-fuzz ────────────────────────────────────────────────────
try:
    from rapidfuzz import fuzz, distance as _rfdist
    _RAPIDFUZZ = True
except ImportError:
    _RAPIDFUZZ = False
    fuzz = None
    _rfdist = None


# ══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _prep(s: str) -> str:
    return s.upper().strip()


def _fuzzy_composite(a: str, b: str) -> float:
    """
    Core 4-algorithm composite score used by I020O, I030O, I040O, I050O.
    Weights: token_sort 30 % | token_set 30 % | partial 20 % | jaro-winkler 20 %
    """
    a, b = _prep(a), _prep(b)
    if not a or not b:
        return 0.0
    if not _RAPIDFUZZ:
        if a == b:        return 1.0
        if a in b or b in a: return 0.75
        return 0.0
    ts  = fuzz.token_sort_ratio(a, b)  / 100.0
    tse = fuzz.token_set_ratio(a, b)   / 100.0
    pr  = fuzz.partial_ratio(a, b)     / 100.0
    jw  = _rfdist.JaroWinkler.normalized_similarity(a, b)
    return round(0.30*ts + 0.30*tse + 0.20*pr + 0.20*jw, 4)


def _dob_match_level(dob_inp: str, dob_cand: str) -> int:
    """
    Return DOB match granularity:
      0 = no match / missing
      1 = year only
      2 = year + month
      3 = exact (year + month + day)
    """
    d1, d2 = str(dob_inp).strip(), str(dob_cand).strip()
    if not d1 or not d2:
        return 0
    if d1[:4] != d2[:4] or not d1[:4].isdigit():
        return 0
    if len(d1) >= 7 and len(d2) >= 7 and d1[:7] == d2[:7]:
        if d1 == d2:
            return 3
        return 2
    return 1


# ══════════════════════════════════════════════════════════════════════════════
#  [I010O]  Exact name match
# ══════════════════════════════════════════════════════════════════════════════

def rule_I010O_exact(inp_name: str, cand_name: str) -> float:
    """
    Returns 1.0 only when the normalised strings are character-for-character
    identical.  Any deviation returns 0.0.
    """
    a, b = _prep(inp_name), _prep(cand_name)
    return 1.0 if (a and b and a == b) else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  [I020O]  Fuzzy composite match
# ══════════════════════════════════════════════════════════════════════════════

def rule_I020O_fuzzy(inp_name: str, cand_name: str) -> float:
    """
    Standard weighted fuzzy composite.  This is what the legacy _composite()
    function computed — now surfaced as a named, auditable rule.
    """
    return _fuzzy_composite(inp_name, cand_name)


# ══════════════════════════════════════════════════════════════════════════════
#  [I030O]  Reversed token order
# ══════════════════════════════════════════════════════════════════════════════

def rule_I030O_reversed(inp_name: str, cand_name: str) -> float:
    """
    "HUSSEIN MOHAMMED" matching "MOHAMMED HUSSEIN".

    Sorts both token lists alphabetically, then runs the fuzzy composite on
    the sorted strings.  Returns 0.0 if the result is no better than the
    direct composite (i.e. the reversal was not the key factor).

    A slight confidence penalty of 0.97× is applied because reversed-order
    matches carry marginally less certainty than canonical-order matches.
    """
    a_toks = sorted(_prep(inp_name).split())
    b_toks = sorted(_prep(cand_name).split())
    if not a_toks or not b_toks:
        return 0.0
    s_sorted  = _fuzzy_composite(" ".join(a_toks), " ".join(b_toks))
    s_direct  = _fuzzy_composite(inp_name, cand_name)
    if s_sorted > s_direct:
        return round(s_sorted * 0.97, 4)
    return s_direct


# ══════════════════════════════════════════════════════════════════════════════
#  [I040O]  Given name / family name split match
# ══════════════════════════════════════════════════════════════════════════════

def _split_name(full: str) -> tuple[str, str]:
    """
    Split 'JOHN MICHAEL SMITH' → given='JOHN MICHAEL', family='SMITH'.
    Single-token name → given='', family=token.
    """
    toks = _prep(full).split()
    if len(toks) == 0:
        return "", ""
    if len(toks) == 1:
        return "", toks[0]
    return " ".join(toks[:-1]), toks[-1]


def rule_I040O_given_family(
    inp_name: str,
    cand_name: str,
    inp_given: str = "",
    inp_family: str = "",
    cand_given: str = "",
    cand_family: str = "",
) -> float:
    """
    Match given name and family name as separate fields.

    If explicit split fields are supplied (inp_given / inp_family / cand_given /
    cand_family) those are used directly; otherwise the full name strings are
    auto-split on the last token.

    Family name is weighted higher (60 %) than given name (40 %) because family
    names are more discriminating in sanctions lists.
    """
    ig = _prep(inp_given)  or _split_name(inp_name)[0]
    if_= _prep(inp_family) or _split_name(inp_name)[1]
    cg = _prep(cand_given) or _split_name(cand_name)[0]
    cf = _prep(cand_family)or _split_name(cand_name)[1]

    if not if_ or not cf:
        return 0.0

    f_score = _fuzzy_composite(if_, cf)
    g_score = _fuzzy_composite(ig, cg) if (ig and cg) else f_score

    return round(0.40 * g_score + 0.60 * f_score, 4)


# ══════════════════════════════════════════════════════════════════════════════
#  [I050O]  Abbreviated given name
# ══════════════════════════════════════════════════════════════════════════════

def rule_I050O_abbreviated(inp_name: str, cand_name: str) -> float:
    """
    Detects when the query contains an initial ("M. ALI", "J SMITH") and checks
    whether the initial matches the first letter of the corresponding token in
    the candidate, then scores the remainder of the name normally.

    Rules:
      • First token of INPUT must be 1 character (+ optional period).
      • That character must match the first character of the first token of CAND.
      • The remaining tokens are scored by fuzzy composite.
      • A small confidence penalty (0.90×) is applied for abbreviated evidence.

    Returns 0.0 if the first token is not an initial.
    """
    inp_toks = _prep(inp_name).split()
    cnd_toks = _prep(cand_name).split()
    if not inp_toks or not cnd_toks:
        return 0.0

    initial = inp_toks[0].rstrip(".")
    if len(initial) != 1:
        return 0.0                          # not an initial

    if not cnd_toks[0].startswith(initial):
        return 0.0                          # initial doesn't match

    inp_rest = " ".join(inp_toks[1:])
    cnd_rest = " ".join(cnd_toks[1:])

    if not inp_rest:
        # Only an initial was provided — weak match on initial alone
        return round(0.60, 4)

    rest_score = _fuzzy_composite(inp_rest, cnd_rest) if cnd_rest else 0.0
    return round(rest_score * 0.90, 4)


# ══════════════════════════════════════════════════════════════════════════════
#  [I060O]  Phonetic match (Double Metaphone per token)
# ══════════════════════════════════════════════════════════════════════════════

def _metaphone_codes(token: str) -> set[str]:
    """Return the set of non-empty double-metaphone codes for a token."""
    try:
        from metaphone import doublemetaphone  # type: ignore
        return {c for c in doublemetaphone(token) if c}
    except ImportError:
        return set()


def rule_I060O_phonetic(inp_name: str, cand_name: str) -> float:
    """
    Token-level Double Metaphone phonetic matching.

    For each token in the input, check whether any candidate token shares a
    metaphone code.  The proportion of matching token pairs drives the score.

    Confidence cap: 0.82 — phonetic evidence is weaker than string evidence.
    Returns 0.0 gracefully when the `metaphone` library is not installed.
    """
    inp_toks = _prep(inp_name).split()
    cnd_toks = _prep(cand_name).split()
    if not inp_toks or not cnd_toks:
        return 0.0

    # Guard: if metaphone is not available, return 0.0 silently
    try:
        from metaphone import doublemetaphone  # noqa: F401 — just check import
    except ImportError:
        return 0.0

    matched = 0
    for it in inp_toks:
        ic = _metaphone_codes(it)
        if not ic:
            continue
        for ct in cnd_toks:
            if ic & _metaphone_codes(ct):
                matched += 1
                break

    if matched == 0:
        return 0.0

    ratio = matched / max(len(inp_toks), len(cnd_toks))
    return round(min(0.82, ratio * 0.82), 4)


# ══════════════════════════════════════════════════════════════════════════════
#  [I070O]  Transliteration match
# ══════════════════════════════════════════════════════════════════════════════

def rule_I070O_transliteration(inp_name: str, cand_name: str) -> float:
    """
    Generate romanised / transliterated variants of both names via the existing
    transliteration_engine, then take the best composite score across all
    (input-variant × candidate-variant) pairs.

    The existing engine covers Arabic, Cyrillic, Persian, and CJK tokens via
    static maps plus the optional `transliterate` library.

    A confidence cap of 0.90 is applied — transliteration matches carry some
    ambiguity compared to direct string matches.
    """
    try:
        from data_layer.utils.transliteration_engine import transliterate
    except ImportError:
        return 0.0

    inp_upper = _prep(inp_name)
    cnd_upper = _prep(cand_name)

    inp_variants = [inp_upper] + [_prep(v) for v in transliterate(inp_upper)]
    cnd_variants = [cnd_upper] + [_prep(v) for v in transliterate(cnd_upper)]

    best = 0.0
    for iv in inp_variants:
        for cv in cnd_variants:
            if not iv or not cv:
                continue
            s = _fuzzy_composite(iv, cv)
            if s > best:
                best = s
                if best >= 1.0:
                    return 1.0  # short-circuit on perfect match

    return round(min(0.90, best), 4)


# ══════════════════════════════════════════════════════════════════════════════
#  [I080O]  DOB cluster (name + DOB together as a distinct rule)
# ══════════════════════════════════════════════════════════════════════════════

# DOB boost values per match granularity
_DOB_BOOST = {1: 0.05, 2: 0.08, 3: 0.12}

# Minimum name score required for DOB to "activate" this cluster rule
_DOB_CLUSTER_MIN_NAME = 0.45


def rule_I080O_dob_cluster(
    inp_name: str,
    cand_name: str,
    inp_dob: str,
    cand_dob: str,
) -> float:
    """
    Name + DOB corroboration cluster rule.

    This is NOT simply a boost on top of another rule.  It fires as a distinct
    named rule when BOTH conditions hold:
      1. Name score reaches the minimum cluster threshold (0.45)
      2. DOB matches at year level or better

    Output = min(1.0, name_score + dob_boost)

    If either condition fails, returns 0.0 — the rule does not fire.
    """
    level = _dob_match_level(inp_dob, cand_dob)
    if level == 0:
        return 0.0

    name_score = _fuzzy_composite(inp_name, cand_name)
    if name_score < _DOB_CLUSTER_MIN_NAME:
        return 0.0

    boost = _DOB_BOOST[level]
    return round(min(1.0, name_score + boost), 4)


# ══════════════════════════════════════════════════════════════════════════════
#  Alias-aware wrapper: run any atomic rule across primary_name + all aliases
# ══════════════════════════════════════════════════════════════════════════════

import pandas as pd


def best_alias_score(
    rule_fn,
    inp_name: str,
    row: pd.Series,
    **kwargs,
) -> tuple[float, str, bool]:
    """
    Apply `rule_fn(inp_name, candidate_name, **kwargs)` against the primary
    name and all pipe-delimited aliases in `row`.

    Returns (best_score, best_matched_name, matched_on_alias).

    For rules that accept extra arguments (e.g. I080O needs DOBs), pass them
    as **kwargs — they are forwarded to every call.
    """
    pname = str(row.get("primary_name", ""))
    candidates: list[tuple[str, bool]] = [(pname, False)]

    aliases_str = row.get("aliases", "")
    if aliases_str:
        for a in str(aliases_str).split("|"):
            a = a.strip()
            if a:
                candidates.append((a, True))

    best_score = 0.0
    best_name  = pname
    best_alias = False

    for cname, is_alias in candidates:
        s = rule_fn(inp_name, cname, **kwargs)
        if s > best_score:
            best_score = s
            best_name  = cname
            best_alias = is_alias

    return best_score, best_name, best_alias
