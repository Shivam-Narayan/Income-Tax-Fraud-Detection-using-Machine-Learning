# Application Architecture

## Status Summary

This repository currently implements a working machine learning inference API and a modular training pipeline for income-tax fraud detection. The codebase is well structured for an internal MVP or a strong prototype, but it is not yet a fully production-grade deployment.

The architecture below reflects the current implementation accurately and also highlights the remaining production-hardening steps that should be addressed before going live.

## 1. System Purpose

The system performs two related tasks:

1. Training and evaluation of fraud-detection models from a tabular census-style dataset.
2. Serving predictions through a Django REST API using a serialized model artifact.

The solution combines:
- a Django web/API layer for inference,
- a Python-based ML training pipeline for preprocessing, model comparison, validation, and artifact export,
- a model artifact and metadata stored for runtime use.

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
Data Loader / Validator
  |
  v
Preprocessing Pipeline
  |
  v
Model Comparison and Validation
  |
  v
Selected Model + Metadata + Artifacts
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

### 3.3 Machine Learning Training Layer

Location: [fraud_detection_ml](fraud_detection_ml)

Responsibilities:
- ingest and validate raw tabular data,
- split data into train/validation/test subsets,
- preprocess feature columns,
- compare several candidate model families,
- select the best model using validation performance,
- export the selected model and metadata.

Key files:
- [fraud_detection_ml/src/data_loader.py](fraud_detection_ml/src/data_loader.py)
- [fraud_detection_ml/src/preprocessing.py](fraud_detection_ml/src/preprocessing.py)
- [fraud_detection_ml/src/model.py](fraud_detection_ml/src/model.py)
- [fraud_detection_ml/src/evaluate.py](fraud_detection_ml/src/evaluate.py)
- [fraud_detection_ml/train_pipeline.py](fraud_detection_ml/train_pipeline.py)

Current behavior:
- multiple model types are compared,
- validation is used to pick the best model,
- the best model is then evaluated on the test split,
- the final pipeline is serialized for inference.

### 3.4 Artifacts and Metadata

Location: [artifacts](artifacts)

Responsibilities:
- persist trained model binaries,
- persist model metadata such as threshold information,
- support versioned and reproducible inference.

Current artifacts:
- [artifacts/model_metadata.json](artifacts/model_metadata.json)
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
2. Data is cleaned and validated.
3. Feature columns and the target column are separated.
4. The data is split into train, validation, and test sets.
5. Candidate models are fitted and compared on the validation set.
6. The best pipeline is evaluated on the holdout test set.
7. The selected model is serialized and exported for inference.

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

The current architecture already shows several good engineering practices:
- clear separation between training and inference,
- modular ML source code under a dedicated package,
- reusable preprocessing and model-building components,
- explicit evaluation metrics and validation-based model selection,
- test coverage for data and model logic,
- API schema generation for documentation.

## 7. Production Gaps to Address

The architecture is strong for an MVP, but it is not yet fully production-grade. The main gaps are:

### Security
- secret keys are currently hard-coded in settings,
- authentication and authorization are not yet implemented for the API,
- request throttling and input abuse protection are not present.

### Reliability and Operations
- there is no centralized logging and metrics pipeline,
- there is no model monitoring for drift or performance regression,
- the fallback logic should not be relied on in production deployments.

### Deployment and Scale
- the current setup is designed for local development and simple hosting,
- containerization, orchestration, and CI/CD are not yet part of the architecture,
- artifact storage and model versioning should be managed through a proper registry or object store.

### Data Governance
- there is no formal feature store or schema registry,
- input validation should be expanded for stricter contract enforcement,
- retraining and rollback policies should be defined.

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
Model Registry / Artifact Store
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
- structured logging and request tracing,
- automated CI/CD pipelines,
- model versioning and rollback support,
- stricter authentication and rate limiting.

## 9. Conclusion

The current architecture is a solid foundation for a machine-learning-powered API and is reasonably well aligned with the code that exists today. It is not yet “perfectly production-level,” but it is close enough to serve as a strong MVP and a good base for production hardening.

In short:
- Current state: good prototype and internal deployment candidate.
- Production target: requires security, observability, deployment, and governance improvements.
