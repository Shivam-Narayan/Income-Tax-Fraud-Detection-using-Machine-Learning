import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import NEGATIVE_LABEL, POSITIVE_LABEL
from src import evaluate as evaluate_module


def test_evaluate_model_creates_artifact_directory(tmp_path, monkeypatch):
    """Evaluation should create the output directory before saving plots."""
    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setattr(evaluate_module, "ARTIFACT_DIR", artifact_dir)

    X_test = pd.DataFrame({"feature": [0, 1, 0, 1]})
    y_test = pd.Series([POSITIVE_LABEL, NEGATIVE_LABEL, POSITIVE_LABEL, NEGATIVE_LABEL])

    pipeline = Pipeline([("classifier", LogisticRegression(max_iter=1000))])
    pipeline.fit(X_test, y_test)

    metrics = evaluate_module.evaluate_model(pipeline, X_test, y_test)

    assert metrics["roc_auc"] >= 0.0
    assert metrics["average_precision"] >= 0.0
    assert artifact_dir.exists()
    assert (artifact_dir / "confusion_matrix.png").exists()
