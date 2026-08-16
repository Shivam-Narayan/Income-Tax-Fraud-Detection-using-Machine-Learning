import json
import joblib
import datetime
import platform
import pandas as pd
import numpy as np
import sklearn
from pathlib import Path

from src.config import (
    DATA_PATH, 
    MODEL_PATH, 
    METADATA_PATH,
    ARTIFACT_DIR,
    logger,
    VALIDATION_SIZE,
    RANDOM_STATE,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from sklearn.model_selection import train_test_split

from src.data_loader import load_data, get_train_test_split, compute_data_hash
from src.preprocessing import build_preprocessor
from src.model import train_model
from src.evaluate import evaluate_model
from src.tflite_export import export_to_tflite

def main():
    logger.info("Starting Enterprise-Grade ML Pipeline...")
    start_time = datetime.datetime.now(datetime.timezone.utc)

    try:
        # 1. Load Data
        logger.info("Loading dataset from %s", DATA_PATH)
        df = load_data(DATA_PATH)
        data_hash = compute_data_hash(df)

        # 2. Split Data
        logger.info("Splitting dataset into train/validation/test sets")
        X_train, X_test, y_train, y_test = get_train_test_split(df)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )

        # 3. Preprocessing
        logger.info("Building preprocessing pipeline")
        preprocessor = build_preprocessor()

        # 4. Train and Select Model
        logger.info("Comparing candidate models on validation data")
        comparison, best_pipeline = train_model(X_train, y_train, preprocessor, X_val, y_val)

        # 5. Evaluate Model
        logger.info("Evaluating selected model on test data")
        metrics = evaluate_model(best_pipeline, X_test, y_test)

        # 6. Save Artifacts
        logger.info("Saving model to %s", MODEL_PATH)
        joblib.dump(best_pipeline, MODEL_PATH)

        end_time = datetime.datetime.now(datetime.timezone.utc)
        
        metadata = {
            "model_info": {
                "version": "1.1.0",
                "selected_model": comparison.get("best_model_name", "unknown"),
                "description": f"Model selected by validation F1 from the comparison workflow: {comparison.get('best_model_name', 'unknown')}.",
            },
            "training_info": {
                "start_time_utc": start_time.isoformat(),
                "end_time_utc": end_time.isoformat(),
                "duration_seconds": (end_time - start_time).total_seconds(),
                "data_hash_sha256": data_hash,
            },
            "environment": {
                "python_version": platform.python_version(),
                "scikit_learn_version": sklearn.__version__,
                "pandas_version": pd.__version__,
                "numpy_version": np.__version__,
                "joblib_version": joblib.__version__,
            },
            "features": {
                "feature_columns": FEATURE_COLUMNS,
                "target_column": TARGET_COLUMN,
            },
            "metrics": metrics,
        }

        with open(METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
        logger.info("Metadata saved to %s", METADATA_PATH)

        # 7. TFLite Export
        logger.info("Exporting model to TensorFlow Lite format")
        export_to_tflite(best_pipeline)

        logger.info("Pipeline completed successfully.")
    except Exception as exc:
        logger.exception("Training pipeline failed")
        raise

if __name__ == "__main__":
    main()
