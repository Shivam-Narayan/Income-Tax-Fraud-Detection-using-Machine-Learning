import hashlib

import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path

from .config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    POSITIVE_LABEL,
    NEGATIVE_LABEL,
    TEST_SIZE,
    RANDOM_STATE,
    logger,
    validate_dataframe,
)
from sklearn.model_selection import train_test_split


def clean_census_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize whitespace and standardize missing tokens."""
    df = raw.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"?": np.nan, "": np.nan})
    return df


def _validate_dataset(df: pd.DataFrame) -> None:
    """Validate that the dataset contains expected columns and valid targets."""
    expected_cols = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {sorted(missing_cols)}")
    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")

    observed = sorted(df[TARGET_COLUMN].unique())
    expected_targets = sorted({POSITIVE_LABEL, NEGATIVE_LABEL})
    if observed != expected_targets:
        raise ValueError(f"Target values do not match expected labels: {observed}")


def compute_data_hash(df: pd.DataFrame) -> str:
    """Compute a SHA-256 hash of the dataframe for reproducibility tracking."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def load_data(data_path: Path) -> pd.DataFrame:
    """Load, clean, validate data from CSV, and log data quality summary."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path.resolve()}")

    logger.info("Loading data from %s", data_path)
    raw_df = pd.read_csv(data_path)
    df = clean_census_frame(raw_df)

    # Schema-level validation
    _validate_dataset(df)
    # Enterprise-grade quality checks from config
    validate_dataframe(df, context="load_data")

    # Log missing-value summary for feature columns
    feature_df = df[FEATURE_COLUMNS + [TARGET_COLUMN]]
    missing = feature_df.isna().sum()
    missing_report = missing[missing > 0].sort_values(ascending=False)
    if not missing_report.empty:
        logger.info("Missing values after cleaning:\n%s", missing_report.to_string())
    else:
        logger.info("No missing values detected in feature columns.")

    logger.info("Successfully loaded %s rows from %s", f"{len(df):,}", data_path.name)
    return df


def get_train_test_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and test sets with stratification."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    logger.info(
        "Splitting data with TEST_SIZE=%s and RANDOM_STATE=%s", TEST_SIZE, RANDOM_STATE
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(
        "Train: %s | Test: %s", f"{len(X_train):,}", f"{len(X_test):,}"
    )
    return X_train, X_test, y_train, y_test

