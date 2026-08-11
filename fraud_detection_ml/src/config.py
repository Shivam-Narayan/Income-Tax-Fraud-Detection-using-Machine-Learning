import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("fraud-detection")

def resolve_project_root() -> Path:
    """Find the folder that contains census-income.csv (works in Jupyter and CLI)."""
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "census-income.csv").exists():
            return candidate
    return Path.cwd()

# --- Reproducibility & paths ---
RANDOM_STATE = 42

PROJECT_ROOT = resolve_project_root()
DATA_PATH = PROJECT_ROOT / "census-income.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = PROJECT_ROOT / "fraud_app" / "model.pkl"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

POSITIVE_LABEL = ">50K"
NEGATIVE_LABEL = "<=50K"
TEST_SIZE = 0.20
CV_FOLDS = 5
N_JOBS = -1

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
