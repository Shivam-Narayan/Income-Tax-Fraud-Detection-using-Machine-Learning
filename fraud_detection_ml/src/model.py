"""Model training with hyperparameter tuning and multi-model comparison.

Enterprise-grade changes vs. the original:
- RandomizedSearchCV for ALL candidate models (not just Logistic Regression)
- sample_weight for models that don't support class_weight natively
- Centralized F1 scorer from config (avoids the pos_label=1 bug)
- Cross-validation confidence intervals on key metrics
- Synced model list with the notebook
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from .config import (
    CV_FOLDS,
    F1_SCORER,
    N_ITER_SEARCH,
    N_JOBS,
    POSITIVE_LABEL,
    RANDOM_STATE,
    logger,
)


# ---------------------------------------------------------------------------
# Candidate model definitions + search spaces
# ---------------------------------------------------------------------------

# Models that require sample_weight because they lack class_weight support
SAMPLE_WEIGHTED_MODELS = {"Gradient Boosting", "Naive Bayes"}
# Models with no rebalancing lever at all — noted explicitly
UNWEIGHTED_MODELS = {"Feed Forward Neural Network", "k-Nearest Neighbors"}


def _get_candidate_specs() -> List[Dict[str, Any]]:
    """Return candidate model specs aligned with the notebook workflow.

    Each entry has:
      name        — display name
      estimator   — sklearn estimator instance
      param_dist  — search space for RandomizedSearchCV (empty → skip tuning)
    """
    return [
        {
            "name": "Logistic Regression",
            "estimator": LogisticRegression(
                random_state=RANDOM_STATE, max_iter=2000, class_weight="balanced"
            ),
            "param_dist": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__solver": ["lbfgs", "saga"],
                "classifier__penalty": ["l2"],
            },
        },
        {
            "name": "Decision Tree",
            "estimator": DecisionTreeClassifier(
                random_state=RANDOM_STATE, class_weight="balanced"
            ),
            "param_dist": {
                "classifier__max_depth": [5, 10, 20, 30, None],
                "classifier__min_samples_split": [2, 5, 10, 20],
                "classifier__min_samples_leaf": [1, 2, 4, 8],
            },
        },
        {
            "name": "Random Forest",
            "estimator": RandomForestClassifier(
                random_state=RANDOM_STATE, n_jobs=N_JOBS, class_weight="balanced"
            ),
            "param_dist": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [10, 20, 30, None],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
            },
        },
        {
            "name": "Feed Forward Neural Network",
            "estimator": MLPClassifier(random_state=RANDOM_STATE, max_iter=1000),
            "param_dist": {
                "classifier__hidden_layer_sizes": [(100,), (100, 50), (64, 32)],
                "classifier__alpha": [0.0001, 0.001, 0.01],
                "classifier__learning_rate_init": [0.001, 0.01],
            },
        },

        {
            "name": "Naive Bayes",
            "estimator": GaussianNB(),
            "param_dist": {
                "classifier__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
            },
        },
        {
            "name": "Gradient Boosting",
            "estimator": GradientBoostingClassifier(
                random_state=RANDOM_STATE, n_iter_no_change=10, validation_fraction=0.1,
            ),
            "param_dist": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "classifier__max_depth": [3, 5, 7],
                "classifier__subsample": [0.8, 0.9, 1.0],
                "classifier__min_samples_leaf": [1, 2, 4],
            },
        },
    ]


# ---------------------------------------------------------------------------
# Confidence interval helper
# ---------------------------------------------------------------------------

def _confidence_interval_95(scores: np.ndarray) -> Tuple[float, float]:
    """Return 95% CI for an array of CV fold scores."""
    n = len(scores)
    if n < 2:
        return (float(scores[0]), float(scores[0]))
    mean = np.mean(scores)
    se = sp_stats.sem(scores)
    ci = sp_stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
    return (float(ci[0]), float(ci[1]))


# ---------------------------------------------------------------------------
# Core training + comparison
# ---------------------------------------------------------------------------

def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    preprocessor,
    model_names: List[str] | None = None,
) -> Tuple[Dict[str, Any], Pipeline]:
    """Train, tune, and compare multiple model families.

    Returns (comparison_dict, best_pipeline).
    """
    if X_train.empty or y_train.empty or X_val.empty or y_val.empty:
        raise ValueError("Training and validation data cannot be empty.")
    if len(X_train) != len(y_train) or len(X_val) != len(y_val):
        raise ValueError("Training and validation inputs must have the same number of rows.")

    specs = _get_candidate_specs()
    if model_names is not None:
        specs = [s for s in specs if s["name"] in model_names]
    if not specs:
        raise ValueError("No candidate models were provided for comparison.")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    results: List[Tuple[str, Dict[str, Any], Pipeline]] = []
    best_pipeline: Pipeline | None = None
    best_f1 = -1.0
    best_name = ""

    for spec in specs:
        name = spec["name"]
        estimator = spec["estimator"]
        param_dist = spec["param_dist"]

        logger.info("Training candidate: %s", name)

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])

        # --- Hyperparameter search (if search space defined) ---
        if param_dist:
            n_iter = min(N_ITER_SEARCH, _count_combos(param_dist))
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_dist,
                n_iter=n_iter,
                cv=cv,
                scoring=F1_SCORER,
                n_jobs=N_JOBS,
                random_state=RANDOM_STATE,
                verbose=0,
                error_score="raise",
            )
            fit_params = {}
            if name in SAMPLE_WEIGHTED_MODELS:
                fit_params["classifier__sample_weight"] = sample_weight

            search.fit(X_train, y_train, **fit_params)
            fitted_pipeline = search.best_estimator_
            logger.info("  Best params for %s: %s", name, search.best_params_)
        else:
            fit_params = {}
            if name in SAMPLE_WEIGHTED_MODELS:
                fit_params["classifier__sample_weight"] = sample_weight
            fitted_pipeline = pipeline.fit(X_train, y_train, **fit_params)

        # --- Cross-validation score with confidence interval ---
        cv_scores = cross_val_score(
            fitted_pipeline, X_train, y_train, cv=cv, scoring=F1_SCORER, n_jobs=N_JOBS,
        )
        cv_mean = float(np.mean(cv_scores))
        cv_ci = _confidence_interval_95(cv_scores)

        # --- Validation-set metrics ---
        y_pred = fitted_pipeline.predict(X_val)
        val_f1 = float(f1_score(y_val, y_pred, pos_label=POSITIVE_LABEL, zero_division=0))
        val_acc = float(accuracy_score(y_val, y_pred))
        val_prec = float(precision_score(y_val, y_pred, pos_label=POSITIVE_LABEL, zero_division=0))
        val_rec = float(recall_score(y_val, y_pred, pos_label=POSITIVE_LABEL, zero_division=0))

        metrics = {
            "f1": val_f1,
            "accuracy": val_acc,
            "precision": val_prec,
            "recall": val_rec,
            "cv_f1_mean": cv_mean,
            "cv_f1_ci_95": cv_ci,
        }
        results.append((name, metrics, fitted_pipeline))

        logger.info(
            "  %s — val F1=%.4f  acc=%.4f  prec=%.4f  rec=%.4f  cv_F1=%.4f (95%% CI [%.4f, %.4f])",
            name, val_f1, val_acc, val_prec, val_rec, cv_mean, cv_ci[0], cv_ci[1],
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_name = name
            best_pipeline = fitted_pipeline

    # --- Build comparison report ---
    ranked = sorted(results, key=lambda r: r[1]["f1"], reverse=True)
    comparison = {
        "best_model_name": best_name,
        "best_metrics": next(m for n, m, _ in ranked if n == best_name),
        "all_results": [
            {"model_name": n, "metrics": m} for n, m, _ in ranked
        ],
    }
    logger.info("Best validation model: %s with F1=%.4f", best_name, best_f1)
    return comparison, best_pipeline


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> Tuple[Dict[str, Any], Pipeline]:
    """Train and select the best model using a validation split."""
    if X_train.empty or y_train.empty:
        raise ValueError("Training data cannot be empty.")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same number of rows.")
    if X_val is None or y_val is None:
        raise ValueError("Validation data is required to select the best model.")

    return compare_models(X_train, y_train, X_val, y_val, preprocessor)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _count_combos(param_dist: Dict) -> int:
    """Upper-bound estimate of the grid size for a param distribution."""
    total = 1
    for v in param_dist.values():
        if hasattr(v, "__len__"):
            total *= len(v)
        else:
            total *= 10  # continuous distribution — use n_iter cap
    return total
