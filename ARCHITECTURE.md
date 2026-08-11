# Application Architecture

## Overview

This Django app is a lightweight prediction API for income tax/fraud classification. It provides a single POST endpoint, `POST /predict/`, and a root health check at `GET /`.

The architecture is intentionally simple:
- `fraud_project/` contains the Django project configuration and URL routing
- `fraud_app/` contains the prediction logic, serializers, and view endpoints
- `artifacts/` contains the saved model metadata and supporting assets
- `fraud_detection_ml/` contains the enterprise MLOps pipeline for training, evaluating, and exporting models
- `census-income.csv` provides the raw training dataset

## High-level Flow

1. User sends a request to `POST /predict/` with Census Income feature values as JSON.
2. The API normalizes hyphenated request fields into Python-friendly names.
3. Validation runs using DRF serializers.
4. Validated payload is converted into a pandas DataFrame.
5. The saved model in `fraud_app/model.pkl` predicts a probability.
6. The app returns a structured JSON response with the result.

## Request-to-Response Flow Diagram

```text
Client                        Django App                       Model
  |                               |                              |
  | POST /predict/                |                              |
  |------------------------------>|                              |
  |                               | parse JSON                   |
  |                               | normalize field names        |
  |                               | validate with serializer     |
  |                               |----------------------------->|
  |                               |                              | load model artifact
  |                               |                              | prepare DataFrame
  |                               |                              | predict probability
  |                               |<-----------------------------|
  |                               | map probability to label     |
  |                               | build response JSON          |
  |<------------------------------|                              |
  | response with prediction      |                              |
```

## Components

### `fraud_project/`

- `settings.py`
  - Registers installed apps: `fraud_app`, `rest_framework`, and `drf_spectacular`
  - Configures `DEFAULT_SCHEMA_CLASS` for DRF to use `drf_spectacular` OpenAPI generation
  - Uses SQLite database for Django admin and test support

- `urls.py`
  - Routes `''` to `fraud_app.urls`
  - Exposes OpenAPI schema at `/api/schema/`
  - Exposes Swagger UI at `/swagger/`
  - Keeps admin available at `/admin/`

### `fraud_app/`

- `views.py`
  - `home(request)`
    - A simple health endpoint returning `{ "message": "App is running", "status": "ok" }`
  - `PredictionInputSerializer`
    - Validates incoming request fields using DRF serializers
    - Expects snake_case fields internally, but request payload may include hyphenated names
  - `PredictionResponseSerializer`
    - Defines the response structure for OpenAPI schema generation
  - `predict(request)`
    - Accepts POST input
    - Copies and normalizes request fields
    - Validates input with `PredictionInputSerializer`
    - Builds a payload with hyphenated feature names for the model
    - Converts payload to `pandas.DataFrame`
    - Runs model prediction or fallback logic if the saved model is missing
    - Returns JSON with `prediction`, `probability`, `threshold`, `model_loaded`, and `data`

- `urls.py`
  - `path('', views.home, name='home')`
  - `path('predict/', views.predict, name='predict')`

### `fraud_detection_ml/`

- `src/`
  - `config.py`: Centralizes constants, feature sets, and paths.
  - `data_loader.py`: Handles dataset ingestion, cleaning, and train/test splits.
  - `preprocessing.py`: Defines the Scikit-Learn `ColumnTransformer` for feature scaling and encoding.
  - `model.py`: Implements model training and hyperparameter tuning (`GridSearchCV`).
  - `evaluate.py`: Calculates classification metrics (ROC AUC, F1) and visualizes the confusion matrix.
  - `tflite_export.py`: Converts the trained classifier weights to a TensorFlow Lite model.
- `tests/`
  - `pytest` suite validating data integrity and pipeline execution.
- `train_pipeline.py`
  - End-to-end orchestration script that loads data, trains the model, and outputs artifacts to `artifacts/` and `fraud_app/`.

### `artifacts/`

- `model_metadata.json`
  - Contains metadata such as `decision_threshold`
  - Loaded at startup and used to decide the classification threshold

- `model.pkl`
  - The trained model artifact used for real inference
  - If missing, the app currently uses fallback rule logic for testing

## Request Validation and Normalization

The API accepts JSON fields like:
- `education-num`
- `marital-status`
- `capital-gain`
- `capital-loss`
- `hours-per-week`
- `native-country`

These are normalized internally to:
- `education_num`
- `marital_status`
- `capital_gain`
- `capital_loss`
- `hours_per_week`
- `native_country`

This normalization ensures DRF serializers validate clean snake_case field names while still accepting the original dataset-style input keys.

## Prediction Logic

- The model is loaded once at module import time from `fraud_app/model.pkl`.
- `METADATA` is loaded from `artifacts/model_metadata.json`.
- `DECISION_THRESHOLD` defaults to `0.5` but can be overridden by metadata.
- The app computes probability using `model.predict_proba(df)[0, 1]`.
- It maps the raw probability to a human-friendly label:
  - `Likely fraud` if the probability is above threshold
  - `Likely safe` otherwise

If the model file is missing, the service uses a fallback heuristic:
- `probability = 0.58` if `capital-gain + capital-loss >= 5000`
- otherwise `probability = 0.42`

## Swagger / OpenAPI Integration

- `drf_spectacular` is used to generate a schema automatically.
- Schema endpoint: `/api/schema/`
- Swagger UI: `/swagger/`

The API schema is generated from DRF serializers and view annotations in `fraud_app/views.py`.

## Example Request

```json
{
  "age": 35,
  "workclass": "Private",
  "education-num": 10,
  "marital-status": "Never-married",
  "occupation": "Other-service",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Male",
  "capital-gain": 1000,
  "capital-loss": 0,
  "hours-per-week": 40,
  "native-country": "United-States"
}
```

## Example Response

```json
{
  "prediction": "Likely safe",
  "probability": 0.054,
  "threshold": 0.5,
  "model_loaded": true,
  "data": {
    "age": 35,
    "workclass": "Private",
    "education-num": 10,
    "marital-status": "Never-married",
    "occupation": "Other-service",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 1000,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
  }
}
```

## Running the App

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Apply database migrations:
   ```bash
   python manage.py migrate
   ```
3. Start the server:
   ```bash
   python manage.py runserver
   ```
4. View Swagger UI at `http://127.0.0.1:8000/swagger/`

## Notes

- The service is built for testing/demo purposes.
- It supports a minimal API surface with only prediction and health endpoints.
- The enterprise ML training pipeline (`fraud_detection_ml/`) is modular and separate from the API runtime, ensuring production readiness and ease of testing.
