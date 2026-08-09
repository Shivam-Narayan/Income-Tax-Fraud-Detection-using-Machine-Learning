# 📌 Income Tax Fraud Detection using Django

## 🚀 Overview
This repository contains a minimal Django API that exposes a single POST endpoint for predicting income class from Census Income features. It is built for testing and demonstration, not production use.

---

## 🔧 Current behavior
- The app serves one POST-only endpoint: `POST /predict/`
- Requests are accepted as JSON and mapped to Census Income input fields
- No frontend, no Swagger/OpenAPI UI, and no homepage endpoint are provided
- The prediction endpoint returns a simple JSON response with probability and decision text

---

## 🧠 Project structure
- `fraud_app/` — Django app with prediction logic and tests
- `fraud_project/` — Django project config and URL routing
- `census-income.csv` — dataset used by the training notebook
- `Income_tax_fraud_detection_updated.ipynb` — updated notebook with training and evaluation

---

## ▶️ Run locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Apply migrations before running the server:
   ```bash
   python manage.py migrate
   ```
3. Start the server:
   ```bash
   python manage.py runserver
   ```
4. Test the API with a POST request:
   ```bash
   curl -X POST http://127.0.0.1:8000/predict/ \
     -H "Content-Type: application/json" \
     -d '{
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
     }'
   ```

---

## 🔍 Request payload
The API accepts these fields in JSON:
- `age`
- `workclass`
- `education-num`
- `marital-status`
- `occupation`
- `relationship`
- `race`
- `sex`
- `capital-gain`
- `capital-loss`
- `hours-per-week`
- `native-country`

---

## ✅ Example response
```json
{
  "prediction": "Likely fraud",
  "probability": 0.732,
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

---

## 🧪 Verification
Run the Django test suite:
```bash
python manage.py test
```

---

## 💡 Notes
This repo is designed for a lightweight API test of the prediction flow. The training notebook remains available for model development and dataset exploration.

