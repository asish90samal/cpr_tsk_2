"""
transliteration_engine.py
──────────────────────────
Generates transliterated / romanised variants of a name.

Fixes applied vs original:
  1. Original name no longer returned as its own transliteration
  2. Arabic, Cyrillic, Persian, and CJK transliteration tables added
  3. Returns deduplicated list excluding the input name
  4. Falls back gracefully when 'transliterate' library is unavailable
"""

from __future__ import annotations
import re

# ── Manual transliteration tables ─────────────────────────────────────────

# Arabic/Persian → Latin common variants
_ARABIC_MAP: dict[str, list[str]] = {
    "ABDALLAH":   ["ABDULLAH", "ABD ALLAH", "ABDALLA"],
    "MOHAMMED":   ["MUHAMMAD", "MEHMET", "MAHOMET"],
    "MOHAMMAD":   ["MOHAMMED", "MUHAMMAD"],
    "HASSAN":     ["HASAN", "HUSSEIN", "HUSSAIN"],
    "HUSSEIN":    ["HUSSAIN", "HOSAIN", "HOSSEIN"],
    "OMAR":       ["UMAR", "UMER"],
    "OSAMA":      ["USAMA", "OUSSAMA"],
    "MUSTAFA":    ["MUSTAPHA", "MOUSTAFA"],
    "KHALID":     ["KHALED", "CALLID"],
    "TARIQ":      ["TAREK", "TARECK"],
    "FATIMA":     ["FATEMEH", "FATIMAH"],
    "AISHA":      ["AYESHA", "AICHA", "AISCHA"],
}

# Cyrillic → Latin variants
_CYRILLIC_MAP: dict[str, list[str]] = {
    "IVANOV":     ["IWANOV", "IVANOFF", "IVANOW"],
    "PETROV":     ["PETROFF", "PETROW", "PETROF"],
    "VOLKOV":     ["WOLKOW", "WOLKOV"],
    "MEDVEDEV":   ["MEDWEDEW", "MEDVEDEV"],
    "PUTIN":      ["POUTINE", "PUTYIN"],
    "KOZLOV":     ["KOSLOV", "KOZLOFF"],
    "MOROZOV":    ["MOROSOW", "MOROZOFF"],
    "SERGEI":     ["SERGEY", "SERGUEI", "SERGEJ"],
    "MIKHAIL":    ["MICHAEL", "MIKHAYIL", "MIKHAEL"],
    "NATALIA":    ["NATALYA", "NATHALIA"],
    "YURI":       ["JURI", "YURIY", "IOURI"],
    "ALEXEI":     ["ALEXEY", "ALEKSEI", "ALEKSEY"],
    "NIKOLAI":    ["NIKOLAY", "NIKOLAJ", "NICKOLAI"],
}

# CJK → Latin variants
_CJK_MAP: dict[str, list[str]] = {
    "ZHANG":      ["CHANG", "ZANG"],
    "WANG":       ["VANG", "WONG"],
    "LI":         ["LEE", "LY", "LYI"],
    "LIU":        ["LYU", "LU"],
    "CHEN":       ["CHAN", "TAN"],
    "HUANG":      ["WONG", "HWANG"],
    "WEI":        ["WAY", "WEY"],
    "YANG":       ["YOUNG", "YEUNG"],
    "KIM":        ["GIM", "GEEM"],
    "PARK":       ["PAK", "BAK"],
    "LEE":        ["LI", "LY", "RHEE"],
    "CHOI":       ["CHOE", "CHEY"],
}

_ALL_MAPS = [_ARABIC_MAP, _CYRILLIC_MAP, _CJK_MAP]


def _map_variants(name: str) -> list[str]:
    """Look up all known transliteration variants from static maps."""
    tokens = name.split()
    variants: list[str] = []

    for mapping in _ALL_MAPS:
        for i, token in enumerate(tokens):
            if token in mapping:
                for replacement in mapping[token]:
                    new_tokens = tokens[:i] + [replacement] + tokens[i + 1:]
                    variants.append(" ".join(new_tokens))

    return variants


def _library_transliterate(name: str) -> list[str]:
    """
    Attempt to use the `transliterate` library for automatic
    script → Latin conversion (Cyrillic, Greek, Armenian, etc.).
    Returns empty list if the library is not installed.
    """
    try:
        from transliterate import translit, get_available_language_codes  # type: ignore
        variants = []
        for lang_code in get_available_language_codes():
            try:
                result = translit(name, lang_code, reversed=True).upper()
                if result != name:
                    variants.append(result)
            except Exception:
                continue
        return variants
    except ImportError:
        return []


def transliterate(name: str) -> list[str]:
    """
    Return a deduplicated list of romanised / transliterated variants.
    The original name itself is NOT included in the returned list.

    Parameters
    ----------
    name : Primary name string (uppercase recommended)
    """
    name = name.strip().upper()
    variants: list[str] = []

    variants += _map_variants(name)
    variants += _library_transliterate(name)

    return list({v.strip() for v in variants if v.strip() and v.strip() != name})
