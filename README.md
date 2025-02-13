# 📌 Income Tax Fraud Detection using Machine Learning

## 🚀 Overview
This project leverages **machine learning** to detect **income tax fraud** by analyzing taxpayers' financial data. We evaluate six machine learning algorithms to determine the most effective model, ultimately selecting **Logistic Regression** for its superior performance in capturing fraudulent patterns. The model is trained on an [OpenML dataset](https://www.openml.org) and deployed as an **Android app** using TensorFlow Lite, making fraud detection accessible and efficient.

---

## 🔥 Key Features

### 📊 Algorithm Comparison
- Evaluates six machine learning algorithms:
  - **Logistic Regression** ✅ (Best Performer)
  - **Decision Tree**
  - **Random Forest**
  - **Naive Bayes**
  - **k-Nearest Neighbors (k-NN)**
  - **Feedforward Neural Network**

### 📂 OpenML Dataset Integration
- Uses publicly available financial datasets for **transparency** and **reproducibility**.

### 📱 Android App Deployment
- Built with **Android Studio** and **TensorFlow Lite**.
- Allows users to **input financial data** and receive **fraud risk scores**.

### 🔍 Fraud Pattern Recognition
- Detects both **linear & non-linear relationships** in financial data, improving accuracy.

---

## 🛠️ Methodology

### 📁 Dataset & Preprocessing
- **Data Source**: OpenML financial datasets.
- **Preprocessing**:
  - Handling missing values
  - Feature scaling
  - Train-test split

### ⚙️ Machine Learning Models
| Algorithm | Accuracy (%) |
|-----------|------------|
| Logistic Regression | **95.2** ✅ |
| Decision Tree | 91.4 |
| Random Forest | 92.8 |
| Naive Bayes | 88.6 |
| k-Nearest Neighbors | 90.3 |
| Feedforward Neural Network | 93.1 |

**Evaluation Metrics**:

✔️ Accuracy  
✔️ Precision  
✔️ Recall  
✔️ F1-Score  

### 📲 Model Deployment
- The best-performing **Logistic Regression** model was **converted to TensorFlow Lite**.
- Integrated into an **Android app**.

#### 📌 User Flow:
1. **Input taxpayer data** (income, deductions, expenses, etc.).
2. **Predict fraud risk** (fraud probability score).
3. **Review flagged cases** for further investigation.

---

## 📈 Results
- **Logistic Regression** achieved the highest accuracy in detecting tax fraud.
- The model successfully identified fraudulent tax patterns.
- Detailed metrics are available in the [Project Report](Report_shivam.pdf).

---

## 💡 Impact

✅ **Reduces Revenue Losses**: Helps governments identify tax fraud, minimizing financial losses.  
✅ **Promotes Fairness**: Ensures equitable tax compliance.  
✅ **Cost-Effective**: Scalable and affordable solution via mobile app deployment.  

---

## 📥 Installation Guide

### 🔧 Prerequisites
- **Android Studio** (latest version)
- **TensorFlow Lite** dependency
- **Python** (for model training, optional)

### 📌 Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Shivam-Narayan/Uncovered_Income_Tax_Fraud_Detection.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt  # For Python model training
   ```
3. Open the Android project in **Android Studio**:
   - Import the `android-app` folder.
   - Sync Gradle dependencies.
4. Run the app on an **emulator or physical device**.

---

## 📝 Usage

1. **Input Financial Data**: Enter taxpayer details (e.g., income, deductions, business expenses).
2. **Predict Fraud Risk**: The app processes data and provides a **fraud probability score**.
3. **Review Results**: Authorities can analyze flagged cases for fraud confirmation.

---

## 🔮 Future Improvements

🔄 **Real-Time Data Processing**: Enhance dynamic fraud detection.  
🌐 **Expand Dataset**: Include more financial profiles for better generalization.  
🔍 **Improve Interpretability**: Enhance model explainability for policymakers and users.  

---

## 👥 Contributing
Contributions are welcome! Follow these steps:
1. **Fork** the repository.
2. **Create a feature branch**.
3. **Submit a pull request**.
4. Open an issue for major changes before implementation.

---

## 📜 License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 📬 Contact
📧 Email: sshivam6495@gmail.com  
🔗 GitHub: [Shivam-Narayan](https://github.com/Shivam-Narayan)  

---

## 📷 Visual Overview

### 📌 Workflow Diagram
![Workflow Diagram](https://via.placeholder.com/800x400?text=Workflow+Diagram)

### 📊 Model Comparison Graph
![Model Comparison](https://via.placeholder.com/800x400?text=Model+Comparison+Graph)

### 📱 Android App Screenshot
![Android App](https://via.placeholder.com/800x400?text=Android+App+Screenshot)

