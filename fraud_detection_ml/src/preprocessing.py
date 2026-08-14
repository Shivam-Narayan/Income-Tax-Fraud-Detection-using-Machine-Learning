import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, logger


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add derived features before the main preprocessing step.

    Engineered features:
    - net_capital: capital-gain minus capital-loss (captures net investment outcome)
    - log_capital_gain: log1p(capital-gain) to reduce right-skew
    """

    _NEW_FEATURES = ["net_capital", "log_capital_gain"]

    def fit(self, X, y=None):  # noqa: ARG002
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES) if not isinstance(X, pd.DataFrame) else X.copy()
        df["net_capital"] = df["capital-gain"] - df["capital-loss"]
        df["log_capital_gain"] = np.log1p(df["capital-gain"].clip(lower=0))
        return df

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else (NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        return base + self._NEW_FEATURES


# Extended numeric features list after engineering
EXTENDED_NUMERIC = NUMERIC_FEATURES + FeatureEngineer._NEW_FEATURES


def build_preprocessor() -> Pipeline:
    """Build a full preprocessing pipeline: engineer → impute → scale/encode.

    Returns a Pipeline (not just a ColumnTransformer) so the feature
    engineering step runs first and feeds into the column transformer.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, EXTENDED_NUMERIC),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder='drop',
    )

    full_preprocessor = Pipeline(steps=[
        ('feature_engineer', FeatureEngineer()),
        ('column_transform', column_transformer),
    ])

    logger.info(
        "Preprocessor built — numeric features: %s, categorical features: %s",
        len(EXTENDED_NUMERIC), len(CATEGORICAL_FEATURES),
    )
    return full_preprocessor
