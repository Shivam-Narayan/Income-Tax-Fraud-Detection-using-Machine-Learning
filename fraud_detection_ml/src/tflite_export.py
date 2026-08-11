import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from pathlib import Path

from .config import logger, ARTIFACT_DIR

def export_to_tflite(pipeline: Pipeline) -> None:
    """
    Extracts the weights from the trained Logistic Regression model
    and exports an equivalent TensorFlow Lite model.
    """
    try:
        import tensorflow as tf
    except ImportError:
        logger.error("TensorFlow is not installed. Skipping TFLite export.")
        return

    logger.info("Starting TFLite export process...")
    
    clf = pipeline.named_steps.get('classifier')
    
    if not isinstance(clf, LogisticRegression):
        logger.warning("TFLite export currently only supports LogisticRegression. Skipping export.")
        return
        
    # Get coefficients and intercept
    weights = clf.coef_.T  # Transpose to match TF Dense layer shape (n_features, 1)
    intercept = clf.intercept_
    n_features = weights.shape[0]
    
    logger.info(f"Model has {n_features} processed features.")
    
    # Build equivalent Keras model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Set weights
    model.layers[0].set_weights([weights, intercept])
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the model
    tflite_path = ARTIFACT_DIR / 'model.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    logger.info(f"Successfully exported TFLite model to {tflite_path}")
    logger.info("NOTE: The TFLite model expects preprocessed features (imputed, scaled, encoded) as input.")
