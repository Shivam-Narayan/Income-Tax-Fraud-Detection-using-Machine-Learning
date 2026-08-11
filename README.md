# 📌 Income Tax Fraud Detection Project

## 🚀 Overview
This repository contains a complete end-to-end machine learning workflow for training, validating, evaluating, and exporting a model from a CSV dataset. It also includes a Django API that can load the exported model for prediction.

This project is designed so that anyone can start from scratch with a new dataset and follow the same process:
1. load data
2. clean and validate it
3. split into train / validation / test sets
4. compare multiple models
5. select the best one
6. evaluate it
7. export the final model as a `.pkl` file

---

## 🧠 Project structure
- `fraud_detection_ml/` — training pipeline, preprocessing, model comparison, evaluation, and export
- `fraud_app/` — Django app that loads the exported model for prediction
- `fraud_project/` — Django project configuration
- `census-income.csv` — sample dataset used by the training pipeline
- `Income_tax_fraud_detection_updated.ipynb` — notebook reference for the modeling workflow

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

This will:
- load the dataset
- split it into train, validation, and test sets
- compare multiple models
- select the best-performing one
- evaluate it on the test set
- export the final model as `.pkl`

### Exported model files
The trained model is saved to:
- `fraud_app/model.pkl`
- `artifacts/model.pkl`

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
- The ML workflow is designed to be reproducible and easy to extend.
- The model comparison step evaluates multiple classifier families and selects the best one by validation performance.
- For production use, you should later add experiment tracking, model monitoring, and better deployment practices.

