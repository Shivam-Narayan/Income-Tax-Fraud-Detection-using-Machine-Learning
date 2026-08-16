import pandas as pd
import numpy as np
import pytest

from src.data_loader import clean_census_frame, _validate_dataset

def test_clean_census_frame():
    """Test that missing values are correctly replaced with np.nan and whitespace is stripped."""
    data = pd.DataFrame({
        "age": [25, 38],
        "workclass": [" Private ", "?"],
        "occupation": ["", "Sales"]
    })
    
    cleaned_df = clean_census_frame(data)
    
    # Check whitespace stripping
    assert cleaned_df.loc[0, "workclass"] == "Private"
    
    # Check "?" replacement
    assert pd.isna(cleaned_df.loc[1, "workclass"])
    
    # Check "" replacement
    assert pd.isna(cleaned_df.loc[0, "occupation"])

def test_validate_dataset_missing_columns():
    """Test that validating a dataset with missing columns raises ValueError."""
    # Create an empty dataframe
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="Missing expected columns"):
        _validate_dataset(df)

def test_validate_dataset_invalid_target():
    """Test that validating a dataset with invalid targets raises ValueError."""
    # From config
    from src.config import FEATURE_COLUMNS, TARGET_COLUMN
    
    # Create dummy data with all required columns
    data = {col: [1] for col in FEATURE_COLUMNS}
    data[TARGET_COLUMN] = ["INVALID_LABEL"]
    
    df = pd.DataFrame(data)
    
    with pytest.raises(ValueError, match="Target values do not match expected labels"):
        _validate_dataset(df)
