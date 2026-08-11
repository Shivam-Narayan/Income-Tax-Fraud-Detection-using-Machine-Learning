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
    logger
)
from sklearn.model_selection import train_test_split

def clean_census_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize whitespace and standardize missing tokens."""
    df = raw.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"?": np.nan, "": np.nan})
    return df

def validate_dataset(df: pd.DataFrame) -> None:
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

def load_data(data_path: Path) -> pd.DataFrame:
    """Load, clean and validate data from CSV."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path.resolve()}")
        
    logger.info(f"Loading data from {data_path}")
    raw_df = pd.read_csv(data_path)
    df = clean_census_frame(raw_df)
    validate_dataset(df)
    logger.info(f"Successfully loaded {len(df)} rows.")
    return df

def get_train_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and test sets."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    logger.info(f"Splitting data with TEST_SIZE={TEST_SIZE} and RANDOM_STATE={RANDOM_STATE}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test
