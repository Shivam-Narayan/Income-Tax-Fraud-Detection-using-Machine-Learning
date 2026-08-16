"""Enterprise-grade model evaluation with comprehensive diagnostics.

Changes from original:
- Full metrics saved to metadata (not just roc_auc + average_precision)
- ROC curve + Precision-Recall curve plots
- Calibration curve
- Feature importance chart (for tree-based models)
- Confusion matrix values in metadata (not just image)
- Classification report as structured dict in metadata
"""
from __future__ import annotations

import json
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from .config import ARTIFACT_DIR, POSITIVE_LABEL, logger


def _positive_class_index(fitted_pipeline: Pipeline) -> int:
    """Find the column index for POSITIVE_LABEL in predict_proba output."""
    return list(fitted_pipeline.classes_).index(POSITIVE_LABEL)


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Evaluate the model comprehensively on the test set.

    Saves plots to ARTIFACT_DIR and returns a dict of structured metrics
    suitable for model_metadata.json.
    """
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError("Pipeline must support predict_proba for probabilistic evaluation.")
    if len(X_test) == 0 or len(y_test) == 0:
        raise ValueError("Test data cannot be empty.")
    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must have the same number of rows.")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Evaluating model on test set (%s samples)…", f"{len(X_test):,}")

    y_pred = pipeline.predict(X_test)
    pos_idx = _positive_class_index(pipeline)
    y_proba = pipeline.predict_proba(X_test)[:, pos_idx]
    y_test_bin = (y_test == POSITIVE_LABEL).astype(int)

    # --- Core metrics ---
    test_f1 = float(f1_score(y_test, y_pred, pos_label=POSITIVE_LABEL, zero_division=0))
    test_acc = float(accuracy_score(y_test, y_pred))
    test_prec = float(precision_score(y_test, y_pred, pos_label=POSITIVE_LABEL, zero_division=0))
    test_rec = float(recall_score(y_test, y_pred, pos_label=POSITIVE_LABEL, zero_division=0))
    roc_auc = float(roc_auc_score(y_test_bin, y_proba))
    avg_prec = float(average_precision_score(y_test_bin, y_proba))

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred)
    logger.info("\n%s", report_text)
    logger.info("ROC-AUC: %.4f | PR-AUC: %.4f | F1: %.4f", roc_auc, avg_prec, test_f1)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred, labels=list(pipeline.classes_))
    _plot_confusion_matrix(cm, list(pipeline.classes_))

    # --- ROC curve ---
    _plot_roc_curve(y_test_bin, y_proba, roc_auc)

    # --- Precision-Recall curve ---
    _plot_pr_curve(y_test_bin, y_proba, avg_prec)

    # --- Calibration curve ---
    _plot_calibration_curve(y_test_bin, y_proba)

    # --- Feature importance (tree-based models) ---
    importances = _extract_feature_importance(pipeline)
    if importances is not None:
        _plot_feature_importance(importances)

    # --- Build structured metrics dict ---
    metrics: Dict[str, Any] = {
        "f1": test_f1,
        "accuracy": test_acc,
        "precision": test_prec,
        "recall": test_rec,
        "roc_auc": roc_auc,
        "average_precision": avg_prec,
        "confusion_matrix": cm.tolist(),
        "classification_report": _clean_report(report_dict),
    }

    return metrics


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(cm: np.ndarray, labels: list) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix — Test Set")
    path = ARTIFACT_DIR / "confusion_matrix.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved confusion matrix → %s", path)


def _plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, auc_val: float) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Test Set")
    plt.legend(loc="lower right")
    path = ARTIFACT_DIR / "roc_curve.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved ROC curve → %s", path)


def _plot_pr_curve(y_true: np.ndarray, y_proba: np.ndarray, ap_val: float) -> None:
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(rec_arr, prec_arr, label=f"PR (AP = {ap_val:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Test Set")
    plt.legend(loc="upper right")
    path = ARTIFACT_DIR / "pr_curve.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved PR curve → %s", path)


def _plot_calibration_curve(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve — Test Set")
    plt.legend(loc="lower right")
    path = ARTIFACT_DIR / "calibration_curve.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved calibration curve → %s", path)


def _extract_feature_importance(pipeline: Pipeline) -> pd.Series | None:
    """Try to extract feature importances from tree-based models."""
    classifier = pipeline.named_steps.get("classifier")
    if classifier is None or not hasattr(classifier, "feature_importances_"):
        return None

    importances = classifier.feature_importances_

    # Try to get feature names from the preprocessor
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            names = preprocessor.get_feature_names_out()
        except Exception:
            names = [f"feature_{i}" for i in range(len(importances))]
    else:
        names = [f"feature_{i}" for i in range(len(importances))]

    return pd.Series(importances, index=names).sort_values(ascending=False)


def _plot_feature_importance(importances: pd.Series, top_n: int = 20) -> None:
    top = importances.head(top_n)
    plt.figure(figsize=(8, 6))
    top.plot(kind="barh")
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.gca().invert_yaxis()
    path = ARTIFACT_DIR / "feature_importance.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved feature importance → %s", path)


def _clean_report(report_dict: dict) -> dict:
    """Convert numpy values in classification_report dict to native Python types."""
    cleaned = {}
    for k, v in report_dict.items():
        if isinstance(v, dict):
            cleaned[k] = {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv for kk, vv in v.items()}
        elif isinstance(v, (float, np.floating)):
            cleaned[k] = float(v)
        else:
            cleaned[k] = v
    return cleaned
