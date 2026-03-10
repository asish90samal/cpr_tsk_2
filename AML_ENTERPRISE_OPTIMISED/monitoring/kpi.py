"""
monitoring/kpi.py
──────────────────
KPI and performance monitoring for the AML screening system.

Fixes applied vs original:
  1. calculate_fp_rate formula corrected: FP / (FP + TN)
  2. False Discovery Rate added as a separate metric (original formula)
  3. Full confusion-matrix-based metrics suite
  4. Throughput and latency tracking
  5. Drift detection helper (score distribution shift)
"""

from __future__ import annotations
from collections import deque
from datetime import datetime, timezone

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Classification metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_fp_rate(fp: int, tn: int) -> float:
    """
    False Positive Rate (FPR) = FP / (FP + TN)
    Also called Fall-out. Measures how often a clear record is incorrectly alerted.

    NOTE: The original code computed FP/(TP+FP) which is the False Discovery Rate —
    a different metric. This function now computes the correct FPR.
    """
    return fp / (fp + tn + 1e-9)


def calculate_fdr(tp: int, fp: int) -> float:
    """
    False Discovery Rate (FDR) = FP / (TP + FP)
    Proportion of raised alerts that are actually false alarms.
    """
    return fp / (tp + fp + 1e-9)


def calculate_recall(tp: int, fn: int) -> float:
    """Recall (Sensitivity) = TP / (TP + FN). Proportion of true matches caught."""
    return tp / (tp + fn + 1e-9)


def calculate_precision(tp: int, fp: int) -> float:
    """Precision = TP / (TP + FP). Proportion of alerts that are true matches."""
    return tp / (tp + fp + 1e-9)


def calculate_f1(tp: int, fp: int, fn: int) -> float:
    """F1 Score — harmonic mean of precision and recall."""
    p = calculate_precision(tp, fp)
    r = calculate_recall(tp, fn)
    return 2 * p * r / (p + r + 1e-9)


def confusion_matrix_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    """
    Compute the full set of binary classification metrics from a confusion matrix.

    Returns
    -------
    Dict with: accuracy, precision, recall, f1, fpr, fdr, specificity, npv
    """
    total = tp + fp + tn + fn + 1e-9
    return {
        "accuracy":    round((tp + tn) / total, 4),
        "precision":   round(calculate_precision(tp, fp), 4),
        "recall":      round(calculate_recall(tp, fn), 4),
        "f1":          round(calculate_f1(tp, fp, fn), 4),
        "fpr":         round(calculate_fp_rate(fp, tn), 4),     # corrected
        "fdr":         round(calculate_fdr(tp, fp), 4),
        "specificity": round(tn / (tn + fp + 1e-9), 4),
        "npv":         round(tn / (tn + fn + 1e-9), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Throughput / latency tracking
# ─────────────────────────────────────────────────────────────────────────────

class ThroughputTracker:
    """
    Track screening throughput and latency using a rolling window.

    Usage
    -----
    tracker = ThroughputTracker(window=1000)
    t0 = time.perf_counter()
    ... run screening ...
    tracker.record(time.perf_counter() - t0)
    print(tracker.summary())
    """

    def __init__(self, window: int = 500) -> None:
        self._latencies: deque[float] = deque(maxlen=window)
        self._window = window

    def record(self, latency_seconds: float) -> None:
        self._latencies.append(latency_seconds)

    def summary(self) -> dict[str, float]:
        if not self._latencies:
            return {}
        arr = np.array(self._latencies)
        return {
            "samples":          len(arr),
            "mean_ms":          round(arr.mean() * 1000, 2),
            "p50_ms":           round(float(np.percentile(arr, 50)) * 1000, 2),
            "p95_ms":           round(float(np.percentile(arr, 95)) * 1000, 2),
            "p99_ms":           round(float(np.percentile(arr, 99)) * 1000, 2),
            "throughput_per_s": round(1.0 / (arr.mean() + 1e-9), 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Score distribution drift detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_score_drift(
    baseline_scores: list[float],
    current_scores:  list[float],
    threshold: float = 0.05,
) -> dict[str, float | bool]:
    """
    Detect distributional drift in model scores using the KS statistic.
    Alerts if the KS statistic exceeds `threshold` (default 0.05).

    Returns
    -------
    Dict with ks_statistic, p_value, drift_detected flag.
    """
    from scipy.stats import ks_2samp   # lazy import to avoid hard dependency
    ks_stat, p_value = ks_2samp(baseline_scores, current_scores)
    return {
        "ks_statistic":   round(float(ks_stat), 4),
        "p_value":        round(float(p_value), 4),
        "drift_detected": bool(ks_stat > threshold),
    }
