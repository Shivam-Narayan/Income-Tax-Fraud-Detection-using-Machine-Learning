# 📌 Income Tax Fraud Detection using Django

## 🚀 Overview
This project provides a minimal Django API for predicting income class based on Census Income features. The current implementation exposes a single POST endpoint for testing predictions and does not include a web form or Swagger UI.

---

## 🔧 What changed
- Uses a Django app with a single API endpoint: `POST /predict/`
- Removed extra routes and Swagger/OpenAPI support for a simpler testable flow
- Keeps the census-income dataset and training notebook as the model foundation
- Includes Django test coverage for the prediction endpoint

---

## 🧠 Project structure
- `fraud_app/` — Django app containing prediction logic and tests
- `fraud_project/` — Django project configuration and URL routing
- `census-income.csv` — dataset used by the training notebook
- `Income_tax_fraud_detection_updated.ipynb` — updated notebook with the training pipeline

---

## ▶️ Run locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the development server:
   ```bash
   python manage.py runserver
   ```
3. Send a POST request to the prediction endpoint:
   ```http
   POST http://127.0.0.1:8000/predict/
   Content-Type: application/json
   ```

   Example request body:
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

---

## 🧪 Verification
Run Django tests with:
```bash
python manage.py test
```

---

## 💡 Notes
The API is intentionally simple for test purposes. Future improvements can include a trained model artifact, validation of request inputs, and richer result metadata.

