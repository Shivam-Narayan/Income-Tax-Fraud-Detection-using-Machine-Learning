# 📌 Income Tax Fraud Detection Project

## 🚀 Overview
This repository contains a complete end-to-end enterprise-grade machine learning workflow for training, validating, evaluating, and exporting a model from a CSV dataset. It also includes a Django API that can load the exported model for prediction.

This project implements modern ML best practices:
1. **Data Loader:** Schema validation, data quality assertions, and SHA-256 data hashing.
2. **Preprocessing:** Configurable pipelines, automated feature engineering (`net_capital`, `log_capital_gain`), and leakage-safe scaling/encoding.
3. **Model Selection:** Automated cross-validation with `RandomizedSearchCV` across multiple model architectures, handling class imbalance natively or via sample weights, and 95% CI metrics.
4. **Diagnostics:** Full evaluation suite including ROC, Precision-Recall, Calibration curves, and structured feature importance.
5. **Serialization:** Exports to both standard Scikit-Learn `.pkl` and edge-ready TensorFlow Lite (`.tflite`) formats, alongside a rich JSON metadata artifact tracking execution environments and evaluation metrics.

---

## 🧠 Project structure
- `fraud_detection_ml/` — Modular training pipeline (`src/`), preprocessing, model comparison, evaluation, and export
- `fraud_app/` — Django app that loads the exported model for prediction
- `fraud_project/` — Django project configuration
- `census-income.csv` — sample dataset used by the training pipeline
- `Income_tax_fraud_detection_updated.ipynb` — Interactive notebook demonstrating the modeling workflow
- `artifacts/` — Generated directory containing trained models (`model.pkl`, `model.tflite`), diagnostics plots, and `model_metadata.json`

---

## ⚙️ Setup from scratch
1. Open the project folder:
   ```powershell
   cd "C:\Projects\fraud-detection\Income-Tax-Fraud-Detection-using-Machine-Learning"
   ```

2. Create or activate the Python environment:
   ```powershell
   c:/Projects/fraud-detection/.venv-1/Scripts/python.exe -m pip install -r requirements.txt
   ```

3. Install the ML package dependencies:
   ```powershell
   cd fraud_detection_ml
   c:/Projects/fraud-detection/.venv-1/Scripts/python.exe -m pip install -r requirements.txt
   ```

---

## 🏋️ Train, validate, evaluate, and export the model
From the `fraud_detection_ml` folder, run:

```powershell
c:/Projects/fraud-detection/.venv-1/Scripts/python.exe train_pipeline.py
```

This will automatically:
- Load the dataset and run schema validation
- Generate target-stratified train, validation, and test splits
- Perform automated feature engineering
- Compare multiple models using randomized hyperparameter search inside cross-validation
- Evaluate the best model on the holdout test set, generating diagnostic plots
- Export the final model to `.pkl` and `.tflite`
- Write comprehensive tracking data to `model_metadata.json`

### Exported model files
The trained artifacts are saved to `artifacts/`:
- `model.pkl` (Scikit-Learn Pipeline)
- `model.tflite` (TensorFlow Lite Model)
- `model_metadata.json`
- Diagnostics: `roc_curve.png`, `pr_curve.png`, `calibration_curve.png`, `feature_importance.png`, `confusion_matrix.png`

The `.pkl` is also automatically copied to `fraud_app/model.pkl` for immediate Django integration.

---

## 🧪 Run tests
Run the test suite with:

```powershell
c:/Projects/fraud-detection/.venv-1/Scripts/python.exe -m pytest -q
```

---

## 🔄 Use a new dataset
To use a different dataset from scratch:

1. Replace or add your CSV file in the project folder.
2. Update the expected columns in `fraud_detection_ml/src/config.py`:
   - `FEATURE_COLUMNS`
   - `TARGET_COLUMN`
   - `EXPECTED_SCHEMA`
3. Make sure the dataset has the same feature names expected by the pipeline.
4. Run:
   ```powershell
   c:/Projects/fraud-detection/.venv-1/Scripts/python.exe train_pipeline.py
   ```

---

## ▶️ Run the Django API
If you want to use the exported model inside the API:

1. Apply migrations:
   ```powershell
   python manage.py migrate
   ```

2. Start the server:
   ```powershell
   python manage.py runserver
   ```

3. Send a prediction request:
   ```powershell
   curl -X POST http://127.0.0.1:8000/predict/ -H "Content-Type: application/json" -d '{"age": 35, "workclass": "Private", "education-num": 10, "marital-status": "Never-married", "occupation": "Other-service", "relationship": "Not-in-family", "race": "White", "sex": "Male", "capital-gain": 1000, "capital-loss": 0, "hours-per-week": 40, "native-country": "United-States"}'
   ```

---

## 💡 Notes
- The ML workflow is designed to be highly reproducible and easy to extend via the `src/` modules.
- The model comparison step evaluates multiple classifier families, handling imbalanced datasets systematically through `class_weight` and `sample_weight`.
- For production use, consider tracking the rich `model_metadata.json` output via an experiment tracker (like MLflow) and integrating the pipeline directly into CI/CD.
