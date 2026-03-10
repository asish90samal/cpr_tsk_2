"""
blocking.py
───────────
Candidate blocking / pre-filtering before expensive matching.

Fixes applied vs original:
  1. Alias-aware blocking — candidates whose aliases share the input initial are included
  2. Soundex / metaphone blocking strategy added (phonetic)
  3. N-gram token blocking added for short names
  4. Multi-strategy combinator that unions all strategies
  5. Returns deduplicated candidate DataFrame with blocking_reason column
"""

from __future__ import annotations
import re

import pandas as pd

try:
    from metaphone import doublemetaphone  # pip install metaphone
    _METAPHONE_AVAILABLE = True
except ImportError:
    _METAPHONE_AVAILABLE = False


# ── Strategy 1 : First-letter (original — extended to aliases) ─────────────

def _first_letter_candidates(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Return rows where primary_name OR any alias begins with the same
    first character as the input name.
    """
    if not name:
        return df.iloc[0:0]
    initial = name[0].upper()

    primary_match = df["primary_name"].str.startswith(initial, na=False)

    # aliases column is a pipe-delimited string — match if any alias starts with initial
    alias_match = df["aliases"].str.contains(
        rf"(^|\|){re.escape(initial)}", regex=True, na=False
    )

    mask = primary_match | alias_match
    result = df[mask].copy()
    result["blocking_reason"] = "first_letter"
    return result


# ── Strategy 2 : Phonetic (Double Metaphone) ──────────────────────────────

def _metaphone_key(name: str) -> set[str]:
    """Return the set of non-empty Double Metaphone codes for all name tokens."""
    keys: set[str] = set()
    for token in name.upper().split():
        codes = doublemetaphone(token)
        keys.update(c for c in codes if c)
    return keys


def _phonetic_candidates(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Return rows whose primary_name shares at least one Double Metaphone
    code with the input name.
    Skipped if `metaphone` library is not installed.
    """
    if not _METAPHONE_AVAILABLE:
        return df.iloc[0:0]

    input_keys = _metaphone_key(name)
    if not input_keys:
        return df.iloc[0:0]

    def _shares_key(candidate_name: str) -> bool:
        return bool(_metaphone_key(candidate_name) & input_keys)

    mask = df["primary_name"].apply(_shares_key)
    result = df[mask].copy()
    result["blocking_reason"] = "phonetic"
    return result


# ── Strategy 3 : Token overlap ────────────────────────────────────────────

def _token_candidates(df: pd.DataFrame, name: str, min_shared: int = 1) -> pd.DataFrame:
    """
    Return rows that share at least `min_shared` tokens with the input name.
    Useful for compound names like "OSAMA BIN LADEN".
    """
    input_tokens = set(name.upper().split())
    if not input_tokens:
        return df.iloc[0:0]

    def _overlap(candidate_name: str) -> bool:
        return len(set(candidate_name.upper().split()) & input_tokens) >= min_shared

    mask = df["primary_name"].apply(_overlap)
    result = df[mask].copy()
    result["blocking_reason"] = "token_overlap"
    return result


# ── Combined entry-point ───────────────────────────────────────────────────

def block_candidates(
    df: pd.DataFrame,
    name: str,
    strategies: list[str] | None = None,
) -> pd.DataFrame:
    """
    Apply one or more blocking strategies and return a deduplicated
    candidate DataFrame.

    Parameters
    ----------
    df         : Full sanctioned-entities DataFrame
    name       : Input name to screen (already normalised)
    strategies : List of strategy names to apply.
                 Defaults to ['first_letter', 'phonetic', 'token'].
                 Union of all strategies is returned.

    Returns
    -------
    pd.DataFrame with blocking_reason column added.
    """
    if strategies is None:
        strategies = ["first_letter", "phonetic", "token"]

    frames: list[pd.DataFrame] = []

    if "first_letter" in strategies:
        frames.append(_first_letter_candidates(df, name))
    if "phonetic" in strategies:
        frames.append(_phonetic_candidates(df, name))
    if "token" in strategies:
        frames.append(_token_candidates(df, name))

    if not frames:
        return df.iloc[0:0]

    combined = pd.concat(frames).drop_duplicates(subset=["entity_id"])
    return combined.reset_index(drop=True)


# ── Convenience alias (backwards compatible with original API) ─────────────

def first_letter_block(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Backwards-compatible wrapper — now also includes alias-based blocking."""
    return _first_letter_candidates(df, name)
