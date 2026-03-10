"""
alias_engine.py
───────────────
Generates name aliases for screening candidates.

Fixes applied vs original:
  1. Original name no longer included as its own alias
  2. Entity-type-aware logic — COMPANY/LIMITED substitutions only for non-individuals
  3. Common name abbreviations expanded (not just MOHAMMED)
  4. Particle handling (von, van, de, al-, bin, binte, etc.)
  5. Returns deduplicated list excluding the input name itself
"""

from __future__ import annotations

# ── Individual name abbreviation map ──────────────────────────────────────
_INDIVIDUAL_SUBS: dict[str, list[str]] = {
    "MOHAMMED":  ["MOHAMMAD", "MUHAMMED", "MUHAMED", "MOHAMAD", "MOHD", "MD", "MHD"],
    "MUHAMMAD":  ["MOHAMMED", "MOHAMMAD", "MUHAMMED", "MOHD"],
    "ABDALLAH":  ["ABDULLAH", "ABD ALLAH"],
    "AHMAD":     ["AHMED", "AHMET"],
    "AHMED":     ["AHMAD", "AHMET"],
    "IBRAHIM":   ["EBRAHIM", "BRAHIM"],
    "YUSUF":     ["YUSEF", "YOUSSEF", "JOSEPH"],
    "ALI":       ["ALIE", "ALEE"],
    "HASSAN":    ["HASAN", "HUSSAIN", "HUSSEIN"],
    "HUSSAIN":   ["HASSAN", "HASAN", "HUSSEIN"],
    "IVAN":      ["IVAN"],
    "IVANOV":    ["IWANOV", "IVANOFF"],
    "ZHANG":     ["CHANG", "ZANG"],
    "WANG":      ["VANG"],
    "LI":        ["LEE", "LY"],
    "PETROV":    ["PETROFF", "PETROW"],
}

# ── Corporate entity abbreviation map ─────────────────────────────────────
_ENTITY_SUBS: dict[str, list[str]] = {
    "LIMITED":  ["LTD", "LTD.", "L.T.D"],
    "COMPANY":  ["CO", "CO.", "CORP", "CORPORATION"],
    "TRADING":  ["TRD", "TRG"],
    "BROTHERS": ["BROS", "BRO"],
    "IMPORT":   ["IMP"],
    "EXPORT":   ["EXP"],
    "INTERNATIONAL": ["INTL", "INT'L", "INT"],
    "GROUP":    ["GRP", "GP"],
}

# ── Name particles to strip / permute ─────────────────────────────────────
_PARTICLES = {"AL", "AL-", "EL", "EL-", "BIN", "BIN-", "BINTE", "VAN", "VON", "DE", "DEL", "DI", "LE", "LA"}


def _apply_subs(name: str, sub_map: dict[str, list[str]]) -> list[str]:
    """Return all variants produced by substituting tokens in sub_map."""
    tokens = name.split()
    variants: list[str] = []
    for i, token in enumerate(tokens):
        for src, replacements in sub_map.items():
            if token == src:
                for rep in replacements:
                    new_tokens = tokens[:i] + [rep] + tokens[i + 1:]
                    variants.append(" ".join(new_tokens))
    return variants


def _particle_variants(name: str) -> list[str]:
    """Generate name variants by dropping or retaining name particles."""
    tokens = name.split()
    upper_particles = {p.upper().rstrip("-") for p in _PARTICLES}
    filtered = [t for t in tokens if t.rstrip("-") not in upper_particles]
    variants = []
    if len(filtered) < len(tokens):
        variants.append(" ".join(filtered))
    # also try hyphen-joined double-barrel surnames
    if len(tokens) >= 2:
        variants.append("-".join(tokens))
    return variants


def _initials_variant(name: str) -> list[str]:
    """Produce first-initial-dot surname variant, e.g. M. KHAN."""
    tokens = name.split()
    if len(tokens) >= 2:
        return [f"{tokens[0][0]}. {' '.join(tokens[1:])}"]
    return []


def generate_aliases(name: str, entity_type: str = "INDIVIDUAL") -> list[str]:
    """
    Return a deduplicated list of alias variants for the given name.
    The original name itself is NOT included in the returned list.

    Parameters
    ----------
    name        : The primary name (uppercase recommended)
    entity_type : 'INDIVIDUAL' or 'ENTITY' — controls which substitution map is used
    """
    name = name.strip().upper()
    variants: list[str] = []

    # Apply appropriate substitution map
    if entity_type == "INDIVIDUAL":
        variants += _apply_subs(name, _INDIVIDUAL_SUBS)
        variants += _particle_variants(name)
        variants += _initials_variant(name)
    else:
        variants += _apply_subs(name, _ENTITY_SUBS)
        # Also try individual subs in case entity name contains a personal name
        variants += _apply_subs(name, _INDIVIDUAL_SUBS)

    # Remove original, blanks, duplicates
    return list({v.strip() for v in variants if v.strip() and v.strip() != name})
