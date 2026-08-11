from typing import Dict, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV

from .config import (
    RANDOM_STATE,
    CV_FOLDS,
    N_JOBS,
    POSITIVE_LABEL,
    logger,
)


def build_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """Build a reusable scikit-learn pipeline with a balanced Logistic Regression classifier."""
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=2000,
            class_weight='balanced'
        ))
    ])


def _get_candidate_models() -> List[Tuple[str, Pipeline]]:
    """Return a broad set of candidate models aligned with the notebook workflow."""
    return [
        ("Logistic Regression", build_pipeline(ColumnTransformer([], remainder='passthrough'))),
        (
            "Decision Tree",
            Pipeline([("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced'))]),
        ),
        (
            "Random Forest",
            Pipeline([("classifier", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS, class_weight='balanced'))]),
        ),
        (
            "Extra Trees",
            Pipeline([("classifier", ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS, class_weight='balanced'))]),
        ),
        (
            "Feed Forward Neural Network",
            Pipeline([("classifier", MLPClassifier(random_state=RANDOM_STATE, max_iter=1000))]),
        ),
        (
            "k-Nearest Neighbors",
            Pipeline([("classifier", KNeighborsClassifier())]),
        ),
        (
            "Naive Bayes",
            Pipeline([("classifier", GaussianNB())]),
        ),
        (
            "SVM",
            Pipeline([("classifier", CalibratedClassifierCV(SVC(random_state=RANDOM_STATE), cv=3, method='sigmoid'))]),
        ),
        (
            "Gradient Boosting",
            Pipeline([("classifier", GradientBoostingClassifier(random_state=RANDOM_STATE))]),
        ),
    ]


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    preprocessor: ColumnTransformer,
    model_names: List[str] | None = None,
) -> Tuple[Dict[str, object], Pipeline]:
    """Train and compare multiple model families on a validation split and return the best pipeline."""
    if X_train.empty or y_train.empty or X_val.empty or y_val.empty:
        raise ValueError("Training and validation data cannot be empty.")

    if len(X_train) != len(y_train) or len(X_val) != len(y_val):
        raise ValueError("Training and validation inputs must have the same number of rows.")

    candidate_models = _get_candidate_models()
    if model_names is not None:
        candidate_models = [item for item in candidate_models if item[0] in model_names]

    if not candidate_models:
        raise ValueError("No candidate models were provided for comparison.")

    results = []
    best_pipeline = None
    best_metrics = None
    best_name = None

    for name, base_pipeline in candidate_models:
        logger.info(f"Training candidate model: {name}")
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", base_pipeline.named_steps["classifier"])
        ])

        if name == "Logistic Regression":
            param_grid = {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__solver': ['lbfgs', 'liblinear']
            }
            min_class_count = int(y_train.value_counts().min())
            n_splits = min(CV_FOLDS, min_class_count)
            if n_splits < 2:
                raise ValueError("Not enough samples per class to perform cross-validation.")
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring='f1_macro',
                n_jobs=N_JOBS,
                verbose=0,
            )
            search.fit(X_train, y_train)
            fitted_pipeline = search.best_estimator_
        else:
            fitted_pipeline = pipeline.fit(X_train, y_train)

        y_pred = fitted_pipeline.predict(X_val)
        val_f1 = f1_score(y_val, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        val_accuracy = accuracy_score(y_val, y_pred)
        val_precision = precision_score(y_val, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        val_recall = recall_score(y_val, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        metrics = {
            "f1": float(val_f1),
            "accuracy": float(val_accuracy),
            "precision": float(val_precision),
            "recall": float(val_recall),
            "predictions": y_pred,
        }
        results.append((name, metrics, fitted_pipeline))

        if best_metrics is None or val_f1 > best_metrics["f1"]:
            best_metrics = metrics
            best_name = name
            best_pipeline = fitted_pipeline

    ranked_results = sorted(results, key=lambda item: item[1]["f1"], reverse=True)
    comparison = {
        "best_model_name": best_name,
        "best_metrics": best_metrics,
        "all_results": [
            {"model_name": name, "metrics": metrics} for name, metrics, _ in ranked_results
        ],
    }

    logger.info(f"Best validation model: {best_name} with F1={best_metrics['f1']:.4f}")
    return comparison, best_pipeline


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> Tuple[Dict[str, object], Pipeline]:
    """Train and select the best model using a validation split, mirroring the notebook workflow."""
    if X_train.empty or y_train.empty:
        raise ValueError("Training data cannot be empty.")

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same number of rows.")

    if X_val is None or y_val is None:
        raise ValueError("Validation data is required to select the best model.")

    return compare_models(X_train, y_train, X_val, y_val, preprocessor)
