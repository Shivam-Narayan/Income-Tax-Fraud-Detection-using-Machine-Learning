import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from .config import (
    POSITIVE_LABEL,
    logger,
    ARTIFACT_DIR
)


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate the model on the test set, generate metrics and plots.
    Saves plots to the artifacts directory.
    """
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError("The provided pipeline must support predict_proba for probabilistic evaluation.")

    if len(X_test) == 0 or len(y_test) == 0:
        raise ValueError("Test data cannot be empty.")

    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must have the same number of rows.")

    logger.info("Evaluating model on test set...")

    y_pred = pipeline.predict(X_test)

    if not hasattr(pipeline, "classes_"):
        raise AttributeError("The fitted pipeline does not expose classes_.")

    pos_idx = list(pipeline.classes_).index(POSITIVE_LABEL)
    y_proba = pipeline.predict_proba(X_test)[:, pos_idx]

    y_test_bin = (y_test == POSITIVE_LABEL).astype(int)

    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test_bin, y_proba)
    avg_prec = average_precision_score(y_test_bin, y_proba)

    logger.info("\n" + report)
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"Average Precision (PR-AUC): {avg_prec:.4f}")

    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix on Test Set')

    cm_path = ARTIFACT_DIR / 'confusion_matrix.png'
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()

    logger.info(f"Confusion matrix plot saved to {cm_path}")

    return {
        'roc_auc': roc_auc,
        'average_precision': avg_prec
    }
