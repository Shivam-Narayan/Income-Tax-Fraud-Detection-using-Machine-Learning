# Application Architecture

## Status Summary

This repository currently implements a working machine learning inference API and a modular training pipeline for income-tax fraud detection. Recent enterprise-grade updates have elevated the ML training pipeline into a robust, production-ready system. 

The architecture below reflects the current implementation accurately and highlights the remaining production-hardening steps for the API layer before going live.

## 1. System Purpose

The system performs two related tasks:

1. Training and evaluation of fraud-detection models from a tabular census-style dataset.
2. Serving predictions through a Django REST API using a serialized model artifact.

The solution combines:
- a Django web/API layer for inference,
- a Python-based enterprise ML training pipeline for schema validation, preprocessing, model comparison, hyperparameter tuning, validation, and artifact export,
- a model artifact, TFLite edge model, and comprehensive metadata stored for runtime use.

## 2. High-Level Architecture

```text
Client / API Consumer
        |
        v
Django REST API (fraud_app)
        |
        |-- Request validation and normalization
        |
        v
Model Artifact + Threshold Metadata
        |
        v
Prediction Response
```

Training flow:

```text
Raw Dataset
  |
  v
Data Loader / Schema Validator / Quality Asserter
  |
  v
Preprocessing Pipeline (Feature Engineering, Target Encoding, Scaling)
  |
  v
Model Comparison and Hyperparameter Tuning (RandomizedSearchCV)
  |
  v
Selected Model Evaluation (ROC, PR, Calibration, Feature Importance)
  |
  v
Selected Model + TFLite Model + Metadata + Artifacts
```

## 3. Core Components

### 3.1 Django Project Layer

Location: [fraud_project](fraud_project)

Responsibilities:
- project configuration and Django settings,
- route registration,
- OpenAPI/Swagger schema exposure,
- application startup configuration.

Key files:
- [fraud_project/settings.py](fraud_project/settings.py)
- [fraud_project/urls.py](fraud_project/urls.py)

Current characteristics:
- uses SQLite for local development and testing,
- exposes a simple health endpoint and a prediction endpoint,
- includes DRF and drf-spectacular for API documentation.

### 3.2 Inference API Layer

Location: [fraud_app](fraud_app)

Responsibilities:
- accept prediction requests,
- normalize input field names,
- validate request payloads,
- prepare a DataFrame for the model,
- invoke the trained model artifact,
- return prediction probability and label.

Key files:
- [fraud_app/views.py](fraud_app/views.py)
- [fraud_app/urls.py](fraud_app/urls.py)

Current behavior:
- the health route returns a simple status payload,
- the prediction route accepts JSON payloads and validates them with DRF serializers,
- the model is loaded from a pickled artifact at startup,
- the decision threshold is read from metadata.

### 3.3 Enterprise Machine Learning Training Layer

Location: [fraud_detection_ml](fraud_detection_ml)

Responsibilities:
- ingest raw tabular data,
- perform strict runtime schema validation and data quality assertions,
- compute SHA-256 data hashes for reproducibility,
- split data into target-stratified train/validation/test subsets,
- engineer new features (e.g., `net_capital`, `log_capital_gain`) and preprocess columns,
- compare multiple candidate model families using automated `RandomizedSearchCV`,
- handle class imbalance automatically via `class_weight` and `sample_weight`,
- select the best model using validation F1 performance with 95% confidence intervals,
- evaluate the selected model on a holdout test set (ROC, PR, Calibration, Confusion Matrix),
- export the selected model, TFLite equivalent, and rich JSON metadata.

Key files:
- [fraud_detection_ml/src/data_loader.py](fraud_detection_ml/src/data_loader.py)
- [fraud_detection_ml/src/preprocessing.py](fraud_detection_ml/src/preprocessing.py)
- [fraud_detection_ml/src/model.py](fraud_detection_ml/src/model.py)
- [fraud_detection_ml/src/evaluate.py](fraud_detection_ml/src/evaluate.py)
- [fraud_detection_ml/src/tflite_export.py](fraud_detection_ml/src/tflite_export.py)
- [fraud_detection_ml/train_pipeline.py](fraud_detection_ml/train_pipeline.py)

### 3.4 Artifacts and Metadata

Location: [artifacts](artifacts)

Responsibilities:
- persist trained model binaries (`model.pkl` and `model.tflite`),
- persist deep diagnostic plots,
- persist comprehensive model metadata tracking versions, features, and metrics,
- support versioned and reproducible inference.

Current artifacts generated:
- [artifacts/model_metadata.json](artifacts/model_metadata.json)
- [artifacts/model.tflite](artifacts/model.tflite)
- [fraud_app/model.pkl](fraud_app/model.pkl)

## 4. Request-to-Response Flow

### Inference Flow

1. A client sends a POST request to the prediction endpoint.
2. The API normalizes hyphenated input names to the internal snake_case structure.
3. DRF validates the payload.
4. The validated payload is converted into a pandas DataFrame.
5. The runtime model artifact is loaded.
6. The model predicts a probability.
7. The system maps the probability to a label using the configured threshold.
8. The API returns a JSON response with prediction, probability, threshold, and input data.

### Training Flow

1. The training pipeline loads the dataset from the configured CSV path.
2. Data is strictly validated against the expected schema (`EXPECTED_SCHEMA`) and hashed.
3. Feature columns and the target column are separated.
4. The data is stratified and split into train, validation, and test sets.
5. A pipeline featuring automated engineering and scaling is constructed.
6. Candidate models are tuned via `RandomizedSearchCV` and compared on the validation set.
7. The best pipeline is extensively evaluated on the holdout test set to generate diagnostic plots.
8. The selected model is serialized to `.pkl` and `.tflite` alongside its rich metadata.

## 5. Data Contract

The API currently expects fields such as:
- age
- workclass
- education_num
- marital_status
- occupation
- relationship
- race
- sex
- capital_gain
- capital_loss
- hours_per_week
- native_country

The request layer also supports dataset-style hyphenated keys such as:
- education-num
- marital-status
- capital-gain
- capital-loss
- hours-per-week
- native-country

This makes the API compatible with both modern snake_case payloads and the original CSV-style field names.

## 6. Design Strengths

The current architecture displays excellent ML engineering practices:
- completely decoupled training and inference stages,
- modular, highly maintainable ML source code,
- strict runtime schema validation and data hashing for reproducibility,
- dynamic model selection backed by hyperparameter tuning and cross-validation,
- automated extraction of feature importances, ROC, and PR curves,
- edge-ready TFLite compilation,
- robust API schema generation.

## 7. Production Gaps to Address

While the ML pipeline is now enterprise-grade, the Django API layer still requires some traditional software-engineering enhancements for full production readiness:

### Security
- secret keys are currently hard-coded in settings,
- authentication and authorization are not yet implemented for the API,
- request throttling and input abuse protection are not present.

### Reliability and Operations
- there is no centralized logging and metrics pipeline,
- there is no model monitoring for drift or performance regression in production.

### Deployment and Scale
- the current setup is designed for local development and simple hosting,
- containerization, orchestration, and CI/CD are not yet part of the architecture,
- artifact storage should be migrated to a proper model registry (like MLflow) or object store (like S3/GCS).

## 8. Production-Ready Target Architecture

A production-ready version of this solution should evolve toward the following model:

```text
Client
  |
  v
API Gateway / Load Balancer
  |
  v
Django API Services
  |
  +--> Input validation
  +--> Authentication / authorization
  +--> Logging / tracing / metrics
  +--> Model inference service
  |
  v
Model Registry / Artifact Store (MLflow)
  |
  v
Training Pipeline / Retraining Job
  |
  v
Monitoring and Alerting
```

Recommended production additions:
- environment-based configuration instead of hard-coded values,
- containerized deployment with Docker,
- application monitoring with Prometheus/Grafana or equivalent,
- structured logging and request tracing.

## 9. Conclusion

The current architecture is a highly robust solution for an income-tax fraud detection MVP. The ML training pipeline operates at an enterprise standard, providing deep evaluation metrics, extensive hyperparameter tuning, and cross-platform artifact exports.

In short:
- Current state: Outstanding prototype and internal deployment candidate with a production-ready ML pipeline.
- Next Steps: Containerize the API, add authentication, and connect the ML pipeline to a CI/CD orchestrator and Model Registry.
