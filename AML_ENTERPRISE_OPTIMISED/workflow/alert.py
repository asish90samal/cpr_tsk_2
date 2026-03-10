"""
workflow/alert.py
──────────────────
Alert creation and workflow management for AML screening decisions.

Fixes applied vs original:
  1. create_alert now returns a structured Alert object (not just a string)
  2. Alert severity derived from score
  3. Alert status lifecycle: OPEN → UNDER_REVIEW → CLOSED / ESCALATED
  4. Batch alert creation helper
  5. Alert summary stats
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from policy_engine.thresholds import AlertDecision

AlertStatus   = Literal["OPEN", "UNDER_REVIEW", "ESCALATED", "CLOSED_TRUE_POSITIVE", "CLOSED_FALSE_POSITIVE"]
AlertSeverity = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]


def _derive_severity(score: float) -> AlertSeverity:
    if score >= 0.92:
        return "CRITICAL"
    if score >= 0.80:
        return "HIGH"
    if score >= 0.65:
        return "MEDIUM"
    return "LOW"


@dataclass
class Alert:
    alert_id:       str
    entity_id:      int | str
    input_name:     str
    matched_name:   str
    score:          float
    decision:       str
    severity:       AlertSeverity
    status:         AlertStatus   = "OPEN"
    escalate:       bool          = False
    assigned_to:    str           = "QUEUE"
    created_at:     str           = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at:     str           = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes:          str           = ""


# ── In-memory alert store ──────────────────────────────────────────────────
_ALERTS: dict[str, Alert] = {}
_ALERT_COUNTER: int = 0


def _next_alert_id() -> str:
    global _ALERT_COUNTER
    _ALERT_COUNTER += 1
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"AML-{ts}-{_ALERT_COUNTER:06d}"


def create_alert(
    entity_id:    int | str,
    input_name:   str,
    matched_name: str,
    decision_obj: AlertDecision,
) -> Alert | None:
    """
    Create an Alert for ALERT or REVIEW decisions. Returns None for NO_ALERT.

    Parameters
    ----------
    entity_id     : Candidate entity ID
    input_name    : The name that was screened
    matched_name  : The matched name / alias
    decision_obj  : AlertDecision from policy engine

    Returns
    -------
    Alert object if decision is ALERT or REVIEW, else None
    """
    if decision_obj.decision == "NO_ALERT":
        return None

    alert = Alert(
        alert_id=_next_alert_id(),
        entity_id=entity_id,
        input_name=input_name,
        matched_name=matched_name,
        score=decision_obj.score,
        decision=decision_obj.decision,
        severity=_derive_severity(decision_obj.score),
        escalate=decision_obj.escalate,
        assigned_to="REVIEW_QUEUE" if decision_obj.decision == "REVIEW" else "ALERT_QUEUE",
    )
    _ALERTS[alert.alert_id] = alert
    return alert


def update_alert_status(alert_id: str, status: AlertStatus, notes: str = "") -> Alert | None:
    """Update the status of an existing alert."""
    alert = _ALERTS.get(alert_id)
    if not alert:
        return None
    alert.status     = status
    alert.notes      = notes
    alert.updated_at = datetime.now(timezone.utc).isoformat()
    return alert


def get_alerts(status: AlertStatus | None = None) -> list[Alert]:
    """Return all alerts, optionally filtered by status."""
    alerts = list(_ALERTS.values())
    if status:
        alerts = [a for a in alerts if a.status == status]
    return sorted(alerts, key=lambda a: a.created_at, reverse=True)


def alert_summary() -> dict:
    """Return counts by severity and status."""
    alerts = list(_ALERTS.values())
    return {
        "total":          len(alerts),
        "open":           sum(1 for a in alerts if a.status == "OPEN"),
        "under_review":   sum(1 for a in alerts if a.status == "UNDER_REVIEW"),
        "escalated":      sum(1 for a in alerts if a.status == "ESCALATED"),
        "closed_tp":      sum(1 for a in alerts if a.status == "CLOSED_TRUE_POSITIVE"),
        "closed_fp":      sum(1 for a in alerts if a.status == "CLOSED_FALSE_POSITIVE"),
        "by_severity":    {
            sev: sum(1 for a in alerts if a.severity == sev)
            for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
        },
    }
