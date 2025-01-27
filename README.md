# Income Tax Fraud Detection using Machine Learning

## Overview
This project focuses on building a machine learning model to identify income tax fraud by analyzing taxpayers' financial data. The model compares six algorithms—**Logistic Regression, Decision Tree, Random Forest, Naive Bayes, k-Nearest Neighbors (k-NN), and Feedforward Neural Network**—to detect fraudulent patterns in tax returns. Among these, **Logistic Regression** emerged as the most effective algorithm, capturing both linear and non-linear relationships among variables for accurate fraud detection. The model is trained on an [OpenML dataset](https://www.openml.org) and deployed as an Android app using TensorFlow Lite, making fraud detection accessible for end-users.

---

## Key Features

### 🔢 Algorithm Comparison
- Evaluates six machine learning algorithms:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Naive Bayes
  - k-Nearest Neighbors (k-NN)
  - Feedforward Neural Network

### 🌐 OpenML Dataset Integration
- Uses publicly available financial datasets to ensure transparency and reproducibility.

### 🛠️ Deployment as an Android App
- Developed with Android Studio and TensorFlow Lite for real-world predictions.
- Allows users to input financial data and receive fraud risk scores directly on their devices.

### 🔅 Linear & Non-Linear Pattern Detection
- Captures complex relationships in financial data, improving the model's accuracy.

---

## Methodology

### Data & Algorithms
- **Dataset**: Financial data sourced from OpenML, split into training and testing subsets.
- **Algorithms Evaluated**:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. Naive Bayes
  5. k-Nearest Neighbors (k-NN)
  6. Feedforward Neural Network
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score

### Model Deployment
- The best-performing model (Logistic Regression) was converted to TensorFlow Lite.
- Integrated into an Android app for accessibility.
- **User Flow**:
  1. Input taxpayer data (e.g., income, deductions, expenses).
  2. Predict fraud risk with a probability score.
  3. Flagged results can be further reviewed by authorities.

---

## Results
- **Logistic Regression** achieved the highest accuracy in fraud detection.
- Outperformed existing methods by effectively identifying fraudulent patterns in financial data.
- Detailed metrics for all algorithms are provided in the [project report](Report_shivam.pdf).

---

## Impact

### 📈 Reduces Revenue Losses
- Helps governments identify fraudulent tax returns, minimizing revenue losses.

### ⚖️ Promotes Fairness
- Ensures equitable tax compliance among taxpayers.

### 📊 Cost-Effective Solution
- Affordable and scalable via mobile app deployment.

---

## Installation

### Prerequisites
- Android Studio (latest version)
- TensorFlow Lite dependency
- Python (optional, for model training)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Shivam-Narayan/Uncovered_Income_Tax_Fraud_Detection.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt  # For Python model training
   ```
3. Open the Android project in Android Studio:
   - Import the `android-app` folder.
   - Sync Gradle dependencies.
4. Run the app on an emulator or a physical device.

---

## Usage

### Input Financial Data
- Enter taxpayer information such as income, deductions, and business expenses.

### Predict Fraud Risk
- The app processes the data and returns a fraud probability score.

### Review Results
- Flagged cases can be further investigated by authorities to confirm fraudulent activity.

---

## Future Work

### 🌐 Expand Dataset
- Include more diverse financial profiles for better generalization.

### 🔄 Real-Time Data Processing
- Implement real-time data processing to enhance dynamic fraud detection.

### 🔍 Improve Interpretability
- Enhance model explainability for policymakers and end-users.

---

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.
4. For major changes, open an issue first to discuss the proposed updates.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact
For questions or collaboration inquiries, feel free to reach out:
- **Email**: sshivam6495@gmail.com
- **GitHub**: [Shivam-Narayan](https://github.com/Shivam-Narayan)

---

## Visual Overview

### Workflow Diagram
![Workflow Diagram](https://via.placeholder.com/800x400?text=Workflow+Diagram)

### Model Comparison
![Model Comparison](https://via.placeholder.com/800x400?text=Model+Comparison+Graph)

### Android App Screenshot
![Android App](https://via.placeholder.com/800x400?text=Android+App+Screenshot)

