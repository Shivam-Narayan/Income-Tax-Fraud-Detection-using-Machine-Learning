import pandas as pd
import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from src.preprocessing import build_preprocessor
from src.model import build_pipeline, compare_models
from src.config import FEATURE_COLUMNS, POSITIVE_LABEL, NEGATIVE_LABEL

@pytest.fixture
def sample_data():
    """Create a minimal dataframe matching the census-income schema for testing."""
    np.random.seed(42)
    
    # 10 samples
    data = {}
    for col in FEATURE_COLUMNS:
        if col in ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]:
            data[col] = np.random.randint(18, 65, 10)
        else:
            data[col] = ["Test"] * 10
            
    df = pd.DataFrame(data)
    y = pd.Series([POSITIVE_LABEL] * 5 + [NEGATIVE_LABEL] * 5)
    
    return df, y

def test_pipeline_fit_predict(sample_data):
    """Test that the pipeline can successfully fit and predict without errors."""
    X, y = sample_data
    
    preprocessor = build_preprocessor()
    pipeline = build_pipeline(preprocessor)
    
    # Fit
    pipeline.fit(X, y)
    
    # Predict
    preds = pipeline.predict(X)
    
    assert len(preds) == len(X)
    assert set(preds).issubset({POSITIVE_LABEL, NEGATIVE_LABEL})


def test_compare_models_selects_best_candidate(sample_data):
    """Model comparison should return a best candidate and validation metrics for the selected model."""
    X, y = sample_data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    comparison, best_pipeline = compare_models(
        X_train,
        y_train,
        X_val,
        y_val,
        preprocessor,
        model_names=["Logistic Regression", "Decision Tree"],
    )

    assert comparison["best_model_name"] in {"Logistic Regression", "Decision Tree"}
    assert comparison["best_metrics"]["f1"] >= 0.0
    assert best_pipeline is not None
    assert best_pipeline.named_steps["classifier"] is not None


def test_compare_models_runs_all_default_candidates(sample_data):
    """The default comparison should evaluate the full candidate roster from the notebook workflow."""
    X, y = sample_data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    comparison, _ = compare_models(X_train, y_train, X_val, y_val, preprocessor)

    assert len(comparison["all_results"]) == 9
