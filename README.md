# 📌 Income Tax Fraud Detection using Django

## 🚀 Overview
This project now uses a Django web app to demonstrate tax fraud detection with a machine learning model trained from the existing census-income dataset. The app lets users submit basic taxpayer attributes and receive a fraud-risk prediction through a simple web form.

---

## 🔧 What changed
- Converted the project from an Android/Gradle app to a Django web application.
- Removed the Java/Android-specific project structure.
- Added a web form and prediction view for fraud-risk scoring.
- Kept the original dataset and notebook as the project’s foundation.

---

## 🧠 Project structure
- [fraud_app](fraud_app) — Django app with views, templates, and prediction logic.
- [fraud_project](fraud_project) — Django project settings and URL routing.
- [census-income.csv](census-income.csv) — dataset used as the basis for the demo model.
- [Income_tax_fraud_detection.ipynb](Income_tax_fraud_detection.ipynb) — original training notebook.

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
3. Open the app in your browser at:
   ```text
   http://127.0.0.1:8000/
   ```

---

## 🧪 Verification
The project’s Django tests are included in [fraud_app/tests.py](fraud_app/tests.py) and can be run with:
```bash
python manage.py test
```

---

## 💡 Notes
The current demo uses a lightweight prediction flow and can be expanded later with a fully trained model file and a richer UI.

