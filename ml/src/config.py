import logging
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("fraud-detection")


def resolve_project_root() -> Path:
    """Find the root folder that contains the data directory (works in Jupyter and CLI)."""
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "data" / "census-income.csv").exists():
            return candidate
    return Path.cwd()


def _load_env_file(project_root: Path) -> None:
    """Load a simple .env file if present so local runs can override defaults."""
    env_file = project_root / ".env"
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", str(resolve_project_root()))).expanduser().resolve()
_load_env_file(PROJECT_ROOT)


def _get_env(name: str, default, cast=None):
    value = os.getenv(name)
    if value is None:
        return default
    if cast is None:
        return value
    if cast is bool:
        return value.lower() in {"1", "true", "yes", "on"}
    return cast(value)


# --- Reproducibility & paths ---
RANDOM_STATE = _get_env("RANDOM_STATE", 42, int)

DATA_PATH = Path(_get_env("DATA_PATH", str(PROJECT_ROOT / "data" / "census-income.csv")))
if not DATA_PATH.is_absolute():
    DATA_PATH = (PROJECT_ROOT / DATA_PATH).resolve()

ARTIFACT_DIR = Path(_get_env("ARTIFACT_DIR", str(PROJECT_ROOT / "artifacts")))
if not ARTIFACT_DIR.is_absolute():
    ARTIFACT_DIR = (PROJECT_ROOT / ARTIFACT_DIR).resolve()

MODEL_PATH = Path(_get_env("MODEL_PATH", str(ARTIFACT_DIR / "model.pkl")))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = (PROJECT_ROOT / MODEL_PATH).resolve()

METADATA_PATH = Path(_get_env("METADATA_PATH", str(ARTIFACT_DIR / "model_metadata.json")))
if not METADATA_PATH.is_absolute():
    METADATA_PATH = (PROJECT_ROOT / METADATA_PATH).resolve()

from sklearn.metrics import f1_score, make_scorer

POSITIVE_LABEL = ">50K"
NEGATIVE_LABEL = "<=50K"
TEST_SIZE = _get_env("TEST_SIZE", 0.20, float)
# 25% of the remaining 80% → 20% of the full dataset, matching the notebook
VALIDATION_SIZE = _get_env("VALIDATION_SIZE", 0.25, float)
CV_FOLDS = _get_env("CV_FOLDS", 5, int)
N_JOBS = _get_env("N_JOBS", -1, int)
# Number of random parameter combinations to try per model during tuning
N_ITER_SEARCH = _get_env("N_ITER_SEARCH", 20, int)

FEATURE_COLUMNS = [
    "age",
    "workclass",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]
TARGET_COLUMN = "income"

CATEGORICAL_FEATURES = [
    "workclass",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

NUMERIC_FEATURES = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES]

# Centralized F1 scorer — avoids the `scoring="f1"` bug where pos_label=1 doesn't
# match string labels like "<=50K" / ">50K".
F1_SCORER = make_scorer(f1_score, pos_label=POSITIVE_LABEL, zero_division=0)

# Expected schema for runtime validation of the input dataframe.
EXPECTED_SCHEMA = {
    "age": "int64",
    "workclass": "object",
    "education-num": "int64",
    "marital-status": "object",
    "occupation": "object",
    "relationship": "object",
    "race": "object",
    "sex": "object",
    "capital-gain": "int64",
    "capital-loss": "int64",
    "hours-per-week": "int64",
    "native-country": "object",
    "income": "object",
}


def validate_dataframe(df, context: str = "input") -> None:
    """Validate that a dataframe matches the expected schema and basic quality checks."""
    missing_cols = set(EXPECTED_SCHEMA.keys()) - set(df.columns)
    if missing_cols:
        raise ValueError(f"[{context}] Missing required columns: {missing_cols}")

    # Check target column is not all one class
    target_counts = df[TARGET_COLUMN].value_counts(normalize=True)
    if len(target_counts) < 2:
        raise ValueError(f"[{context}] Target column '{TARGET_COLUMN}' has only one class")
    minority_ratio = target_counts.min()
    if minority_ratio < 0.01:
        logger.warning(
            "[%s] Extreme class imbalance detected: minority class is %.2f%% of data",
            context, minority_ratio * 100,
        )

    # Log data summary
    logger.info("[%s] Shape: %s, Target distribution: %s", context, df.shape,
                target_counts.to_dict())
