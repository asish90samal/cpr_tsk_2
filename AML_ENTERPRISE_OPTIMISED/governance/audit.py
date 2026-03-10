"""
governance/audit.py
────────────────────
Audit trail and decision logging for AML screening.

Fixes applied vs original:
  1. log_decision now accepts and stores the full AlertDecision object
  2. Timestamp added to every log entry
  3. Analyst / system attribution field
  4. Batch logging helper
  5. Export to DataFrame / CSV for review
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any
import pandas as pd

from policy_engine.thresholds import AlertDecision


@dataclass
class AuditEntry:
    entity_id:      int | str
    input_name:     str
    matched_name:   str
    score:          float
    threshold:      float
    decision:       str
    escalate:       bool
    reason:         str
    risk_tier:      str
    analyst:        str               = "SYSTEM"
    timestamp:      str               = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata:       dict[str, Any]    = field(default_factory=dict)


# ── In-memory audit store (swap for DB in production) ─────────────────────
_AUDIT_LOG: list[AuditEntry] = []


def log_decision(
    entity_id:    int | str,
    input_name:   str,
    matched_name: str,
    decision_obj: AlertDecision,
    risk_tier:    str = "MEDIUM",
    analyst:      str = "SYSTEM",
    metadata:     dict[str, Any] | None = None,
) -> AuditEntry:
    """
    Create and store an audit log entry for a screening decision.

    Parameters
    ----------
    entity_id     : The candidate entity's ID
    input_name    : The name that was screened
    matched_name  : The name / alias that produced the match
    decision_obj  : AlertDecision from policy_engine.thresholds.apply_threshold
    risk_tier     : Risk classification of the entity
    analyst       : ID / name of analyst or 'SYSTEM'
    metadata      : Any additional key-value context

    Returns
    -------
    AuditEntry (also appended to in-memory log)
    """
    entry = AuditEntry(
        entity_id=entity_id,
        input_name=input_name,
        matched_name=matched_name,
        score=decision_obj.score,
        threshold=decision_obj.threshold,
        decision=decision_obj.decision,
        escalate=decision_obj.escalate,
        reason=decision_obj.reason,
        risk_tier=risk_tier,
        analyst=analyst,
        metadata=metadata or {},
    )
    _AUDIT_LOG.append(entry)
    return entry


def log_batch(entries: list[AuditEntry]) -> None:
    """Append a list of AuditEntry objects to the log."""
    _AUDIT_LOG.extend(entries)


def get_audit_log() -> list[AuditEntry]:
    """Return all logged audit entries."""
    return list(_AUDIT_LOG)


def clear_audit_log() -> None:
    """Clear the in-memory audit log (use in tests)."""
    _AUDIT_LOG.clear()


def audit_log_to_df() -> pd.DataFrame:
    """Export the full audit log as a pandas DataFrame."""
    return pd.DataFrame([asdict(e) for e in _AUDIT_LOG])


def export_audit_csv(path: str) -> None:
    """Write the audit log to a CSV file."""
    df = audit_log_to_df()
    df.to_csv(path, index=False)
    print(f"[Audit] Exported {len(df)} entries to {path}")
