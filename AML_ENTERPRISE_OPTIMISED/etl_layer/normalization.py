"""
normalization.py
────────────────
ETL-layer name and record normalisation.

Fixes applied vs original:
  1. Script-aware normalisation — Arabic / CJK preserved; only Latin decomposed
  2. Punctuation and noise stripping
  3. Alias string → list helper
  4. DOB parsing with partial-date support (YYYY, YYYY-MM, YYYY-MM-DD)
  5. Country code normalisation lookup
"""

from __future__ import annotations
import re
import unicodedata
from datetime import date

# ── Latin-only NFKD decomposition ─────────────────────────────────────────
_LATIN_RANGE = re.compile(r"[\u0000-\u024F]")   # Basic Latin + Latin Extended


def _is_mostly_latin(name: str) -> bool:
    latin_chars = sum(1 for c in name if _LATIN_RANGE.match(c))
    return latin_chars / max(len(name), 1) >= 0.6


def normalize_name(name: str | None) -> str:
    """
    Normalise a name string for comparison.
    - Strips leading/trailing whitespace
    - Upper-cases
    - For Latin-dominant strings: NFKD-decomposes to remove diacritics
    - For Arabic / CJK strings: preserves original script (no decomposition)
    - Collapses internal whitespace
    - Removes punctuation except hyphens between word characters
    """
    if not name:
        return ""

    name = name.strip().upper()

    if _is_mostly_latin(name):
        # Decompose and strip combining marks (diacritics)
        name = unicodedata.normalize("NFKD", name)
        name = "".join(c for c in name if not unicodedata.combining(c))

    # Remove non-alphanumeric except spaces and intra-word hyphens
    name = re.sub(r"[^\w\s\-]", " ", name)
    name = re.sub(r"(?<!\w)-|-(?!\w)", " ", name)   # strip dangling hyphens
    name = re.sub(r"\s+", " ", name).strip()

    return name


def normalize_aliases(aliases_str: str | None) -> list[str]:
    """
    Parse a pipe-delimited alias string into a list of normalised names.
    Returns an empty list for null / blank inputs.
    """
    if not aliases_str:
        return []
    return [normalize_name(a) for a in aliases_str.split("|") if a.strip()]


def normalize_dob(dob_str: str | None) -> dict[str, int | None]:
    """
    Parse a date-of-birth string of variable precision.
    Supports: 'YYYY', 'YYYY-MM', 'YYYY-MM-DD'

    Returns a dict: {'year': int|None, 'month': int|None, 'day': int|None}
    """
    result: dict[str, int | None] = {"year": None, "month": None, "day": None}
    if not dob_str:
        return result

    parts = dob_str.strip().split("-")
    try:
        if len(parts) >= 1 and parts[0]:
            result["year"] = int(parts[0])
        if len(parts) >= 2 and parts[1]:
            result["month"] = int(parts[1])
        if len(parts) >= 3 and parts[2]:
            result["day"] = int(parts[2])
    except ValueError:
        pass

    return result


def normalize_country(country: str | None) -> str:
    """
    Upper-case and strip country string.
    Expand common abbreviations to full names for consistent matching.
    """
    _COUNTRY_MAP = {
        "KP":   "NORTH KOREA",
        "DPRK": "NORTH KOREA",
        "IR":   "IRAN",
        "SY":   "SYRIA",
        "RU":   "RUSSIA",
        "BY":   "BELARUS",
        "VE":   "VENEZUELA",
        "CU":   "CUBA",
        "MM":   "MYANMAR",
        "SO":   "SOMALIA",
        "SD":   "SUDAN",
        "LY":   "LIBYA",
        "AF":   "AFGHANISTAN",
    }
    if not country:
        return "UNKNOWN"
    upper = country.strip().upper()
    return _COUNTRY_MAP.get(upper, upper)


def split_name_parts(full_name: str | None) -> dict:
    """
    Split a full name into given name and family name.
    Last token = family name; all preceding tokens = given name.

    Returns dict: {'given_name': str, 'family_name': str}

    Examples:
      "JOHN MICHAEL SMITH" -> {'given_name': 'JOHN MICHAEL', 'family_name': 'SMITH'}
      "MOHAMMED"           -> {'given_name': '',              'family_name': 'MOHAMMED'}
    """
    if not full_name:
        return {"given_name": "", "family_name": ""}
    normed = normalize_name(full_name)
    tokens = normed.split()
    if not tokens:
        return {"given_name": "", "family_name": ""}
    if len(tokens) == 1:
        return {"given_name": "", "family_name": tokens[0]}
    return {"given_name": " ".join(tokens[:-1]), "family_name": tokens[-1]}


def detect_script(name: str | None) -> str:
    """
    Detect the dominant script of a name string.
    Returns one of: 'LATIN', 'ARABIC', 'CYRILLIC', 'CJK', 'MIXED', 'UNKNOWN'.
    """
    if not name or not name.strip():
        return "UNKNOWN"
    counts = {"LATIN": 0, "ARABIC": 0, "CYRILLIC": 0, "CJK": 0}
    for ch in name:
        cp = ord(ch)
        if 0x0000 <= cp <= 0x024F:
            counts["LATIN"] += 1
        elif 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
            counts["ARABIC"] += 1
        elif 0x0400 <= cp <= 0x04FF:
            counts["CYRILLIC"] += 1
        elif (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
              0xAC00 <= cp <= 0xD7AF):
            counts["CJK"] += 1
    total = sum(counts.values())
    if total == 0:
        return "UNKNOWN"
    dominant = max(counts, key=lambda k: counts[k])
    if counts[dominant] / total >= 0.80:
        return dominant
    return "MIXED"
