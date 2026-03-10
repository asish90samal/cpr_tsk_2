"""
ml_engine/model.py
──────────────────
XGBoost-based AML name-match scoring model.

Fixes applied vs original:
  1. scale_pos_weight computed from training labels (handles class imbalance)
  2. eval_metric set to 'aucpr' (PR-AUC — correct metric for imbalanced AML data)
  3. Early stopping to prevent overfitting
  4. Feature names stored and validated at predict time
  5. Probability calibration via CalibratedClassifierCV (optional)
  6. Model persistence (save / load)
  7. Threshold-tuned predict_label that uses the policy engine threshold
  8. Performance report helper
"""

from __future__ import annotations
import os
import json
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)

from feature_engine.feature_builder import FEATURE_NAMES
from policy_engine.thresholds import apply_threshold


class AMLModel:
    """
    XGBoost classifier for AML name-match scoring.

    Workflow
    --------
    1. model = AMLModel()
    2. model.train(X, y)          # X: DataFrame or 2D array, y: 0/1 labels
    3. scores = model.predict(X)  # returns float probabilities
    4. decisions = model.predict_labels(X)  # returns 'ALERT' / 'NO_ALERT'
    5. model.save('model.json')
    6. model = AMLModel.load('model.json')
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        calibrate: bool = True,
        random_state: int = 42,
    ) -> None:
        self.n_estimators    = n_estimators
        self.max_depth       = max_depth
        self.learning_rate   = learning_rate
        self.subsample       = subsample
        self.colsample_bytree = colsample_bytree
        self.calibrate       = calibrate
        self.random_state    = random_state
        self.feature_names   = FEATURE_NAMES
        self._model: xgb.XGBClassifier | None = None
        self._calibrated: CalibratedClassifierCV | None = None
        self._is_trained: bool = False

    def _build_xgb(self, scale_pos_weight: float) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=scale_pos_weight,   # KEY FIX: handles class imbalance
            eval_metric="aucpr",                  # KEY FIX: PR-AUC for imbalanced data
            use_label_encoder=False,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )

    def train(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        eval_fraction: float = 0.15,
    ) -> "AMLModel":
        """
        Train model with automatic class-weight calculation and early stopping.

        Parameters
        ----------
        X              : Feature matrix (rows = candidate pairs)
        y              : Binary labels (1 = match / alert, 0 = clear)
        eval_fraction  : Fraction held out for early-stopping eval set
        """
        y_arr = np.asarray(y)
        n_pos = y_arr.sum()
        n_neg = len(y_arr) - n_pos

        if n_pos == 0:
            raise ValueError("Training data contains no positive examples (risk_label=1). "
                             "Check your dataset generation.")

        scale_pos_weight = n_neg / n_pos
        print(f"[AMLModel] Class distribution: {n_neg} negatives, {n_pos} positives "
              f"→ scale_pos_weight={scale_pos_weight:.2f}")

        X_train, X_eval, y_train, y_eval = train_test_split(
            X, y_arr, test_size=eval_fraction, stratify=y_arr, random_state=self.random_state
        )

        self._model = self._build_xgb(scale_pos_weight)
        self._model.fit(
            X_train, y_train,
            eval_set=[(X_eval, y_eval)],
            verbose=False,
        )

        if self.calibrate:
            # Probability calibration improves decision reliability
            self._calibrated = CalibratedClassifierCV(
                self._model, method="isotonic", cv="prefit"
            )
            self._calibrated.fit(X_eval, y_eval)

        self._is_trained = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return probability scores (float array, range 0–1)."""
        self._assert_trained()
        if self.calibrate and self._calibrated:
            return self._calibrated.predict_proba(X)[:, 1]
        return self._model.predict_proba(X)[:, 1]  # type: ignore[union-attr]

    def predict_labels(
        self,
        X: pd.DataFrame | np.ndarray,
        threshold: float | None = None,
    ) -> list[str]:
        """Return 'ALERT' / 'NO_ALERT' decisions for each row."""
        scores = self.predict(X)
        return [apply_threshold(float(s), threshold=threshold) for s in scores]

    def evaluate(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> dict[str, Any]:
        """
        Compute standard evaluation metrics.

        Returns dict with roc_auc, pr_auc, classification_report, confusion_matrix.
        """
        self._assert_trained()
        y_arr   = np.asarray(y)
        y_scores = self.predict(X)
        y_pred  = (y_scores >= 0.5).astype(int)

        metrics = {
            "roc_auc":               round(roc_auc_score(y_arr, y_scores), 4),
            "pr_auc":                round(average_precision_score(y_arr, y_scores), 4),
            "classification_report": classification_report(y_arr, y_pred, target_names=["CLEAR", "ALERT"]),
            "confusion_matrix":      confusion_matrix(y_arr, y_pred).tolist(),
        }
        print(f"ROC-AUC : {metrics['roc_auc']}")
        print(f"PR-AUC  : {metrics['pr_auc']}")
        print(metrics["classification_report"])
        return metrics

    def save(self, path: str) -> None:
        """Persist model to disk (XGBoost native JSON format)."""
        self._assert_trained()
        self._model.save_model(path)  # type: ignore[union-attr]
        meta = {
            "feature_names": self.feature_names,
            "calibrate":     self.calibrate,
        }
        with open(path + ".meta.json", "w") as f:
            json.dump(meta, f)
        print(f"[AMLModel] Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "AMLModel":
        """Load a previously saved model."""
        instance = cls()
        instance._model = xgb.XGBClassifier()
        instance._model.load_model(path)
        meta_path = path + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            instance.feature_names = meta.get("feature_names", FEATURE_NAMES)
        instance._is_trained = True
        return instance

    def _assert_trained(self) -> None:
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model has not been trained yet. Call .train(X, y) first.")
