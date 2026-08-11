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
    """Find the folder that contains census-income.csv (works in Jupyter and CLI)."""
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "census-income.csv").exists():
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

DATA_PATH = Path(_get_env("DATA_PATH", str(PROJECT_ROOT / "census-income.csv")))
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

POSITIVE_LABEL = ">50K"
NEGATIVE_LABEL = "<=50K"
TEST_SIZE = _get_env("TEST_SIZE", 0.20, float)
CV_FOLDS = _get_env("CV_FOLDS", 5, int)
N_JOBS = _get_env("N_JOBS", -1, int)

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
