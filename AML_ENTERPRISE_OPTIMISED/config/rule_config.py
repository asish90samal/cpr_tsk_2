"""
config/rule_config.py
══════════════════════
Per-job rule enable/disable configuration.
Equivalent to OWS Run Profile properties:
  phase.*.process.*.[I010O] Exact name only.san_rule_enabled = true/false

Each job code maps to a RuleConfig object whose .enabled(key) method
tells the RuleChain whether to execute a given atomic rule.

Default policy:
  • Exact + Fuzzy + Reversed + Given/Family + Abbreviated + Transliteration
    are ON for all name-match jobs.
  • Phonetic (I060O) is OFF for SAN/SCION (high false-positive rate on
    sanction names) but ON for PEP jobs (higher sensitivity acceptable).
  • DOB cluster (I080O) is ON only for INDIVIDUAL jobs that carry a DOB field.
  • NNS jobs disable DOB cluster (NNS records don't carry DOB).

To change flags at runtime:
    from config.rule_config import get_rule_config
    cfg = get_rule_config("SAN_IND")
    cfg.set("I060O_PHONETIC", True)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict


# ── Default ON/OFF for every rule key ─────────────────────────────────────
_DEFAULTS: Dict[str, bool] = {
    "I010O_EXACT":           True,
    "I020O_FUZZY":           True,
    "I030O_REVERSED":        True,
    "I040O_GIVEN_FAMILY":    True,
    "I050O_ABBREVIATED":     True,
    "I060O_PHONETIC":        False,   # OFF by default — high FP rate
    "I070O_TRANSLITERATION": True,
    "I080O_DOB_CLUSTER":     True,
}


@dataclass
class RuleConfig:
    """
    Holds rule flags for a single job.  Inherits from defaults; overrides
    are applied per-job below.
    """
    job_code: str
    _flags: Dict[str, bool] = field(default_factory=dict)

    def __post_init__(self):
        # Start from defaults, then apply any job-specific overrides
        merged = dict(_DEFAULTS)
        merged.update(self._flags)
        self._flags = merged

    def enabled(self, key: str) -> bool:
        """Return True if the rule identified by `key` is enabled for this job."""
        return self._flags.get(key, True)

    def set(self, key: str, value: bool) -> None:
        """Override a rule flag at runtime (e.g. for A/B testing)."""
        self._flags[key] = value

    def as_dict(self) -> Dict[str, bool]:
        """Return a copy of all flags."""
        return dict(self._flags)


# ══════════════════════════════════════════════════════════════════════════════
#  Per-job configurations
# ══════════════════════════════════════════════════════════════════════════════

_JOB_CONFIGS: Dict[str, RuleConfig] = {

    # ── SAN_IND ───────────────────────────────────────────────────────────
    # Sanctions Individual: strict name + DOB, phonetic OFF (high FP)
    "SAN_IND": RuleConfig(
        job_code="SAN_IND",
        _flags={
            "I060O_PHONETIC":     False,
            "I080O_DOB_CLUSTER":  True,
        },
    ),

    # ── SAN_ENT ───────────────────────────────────────────────────────────
    # Sanctions Entity: no DOB (entities don't have DOB), phonetic OFF
    "SAN_ENT": RuleConfig(
        job_code="SAN_ENT",
        _flags={
            "I060O_PHONETIC":     False,
            "I080O_DOB_CLUSTER":  False,   # entities have no DOB
        },
    ),

    # ── SCION_IND ─────────────────────────────────────────────────────────
    # Internal watchlist individual: all standard rules, phonetic OFF
    "SCION_IND": RuleConfig(
        job_code="SCION_IND",
        _flags={
            "I060O_PHONETIC":     False,
            "I080O_DOB_CLUSTER":  True,
        },
    ),

    # ── SCION_ENT ─────────────────────────────────────────────────────────
    "SCION_ENT": RuleConfig(
        job_code="SCION_ENT",
        _flags={
            "I060O_PHONETIC":     False,
            "I080O_DOB_CLUSTER":  False,
        },
    ),

    # ── PEP_IND ───────────────────────────────────────────────────────────
    # PEP Individual: phonetic ON (higher sensitivity acceptable for PEP),
    # DOB cluster ON
    "PEP_IND": RuleConfig(
        job_code="PEP_IND",
        _flags={
            "I060O_PHONETIC":     True,    # PEP jobs can afford higher sensitivity
            "I080O_DOB_CLUSTER":  True,
        },
    ),

    # ── PEP_ENT ───────────────────────────────────────────────────────────
    "PEP_ENT": RuleConfig(
        job_code="PEP_ENT",
        _flags={
            "I060O_PHONETIC":     True,
            "I080O_DOB_CLUSTER":  False,   # entities have no DOB
        },
    ),

    # ── NNS_IND ───────────────────────────────────────────────────────────
    # Negative News Individual: phonetic ON, DOB cluster OFF (NNS lacks DOB)
    "NNS_IND": RuleConfig(
        job_code="NNS_IND",
        _flags={
            "I060O_PHONETIC":     True,
            "I080O_DOB_CLUSTER":  False,   # NNS records don't carry DOB
        },
    ),

    # ── NNS_ENT ───────────────────────────────────────────────────────────
    "NNS_ENT": RuleConfig(
        job_code="NNS_ENT",
        _flags={
            "I060O_PHONETIC":     True,
            "I080O_DOB_CLUSTER":  False,
        },
    ),

    # ── CTRY jobs ─────────────────────────────────────────────────────────
    # Country jobs don't use name matching — all rules OFF
    "CTRY_IND": RuleConfig(
        job_code="CTRY_IND",
        _flags={k: False for k in _DEFAULTS},
    ),
    "CTRY_ENT": RuleConfig(
        job_code="CTRY_ENT",
        _flags={k: False for k in _DEFAULTS},
    ),
}


def get_rule_config(job_code: str) -> RuleConfig:
    """
    Return the RuleConfig for the given job_code.
    Falls back to the default config (all defaults) for unknown job codes.
    """
    code = job_code.strip().upper()
    if code in _JOB_CONFIGS:
        return _JOB_CONFIGS[code]
    # Unknown job → default all-on config
    return RuleConfig(job_code=code, _flags={})


def list_all_configs() -> Dict[str, Dict[str, bool]]:
    """Return a human-readable dict of all job configs (for reporting / debugging)."""
    return {jc: cfg.as_dict() for jc, cfg in sorted(_JOB_CONFIGS.items())}
