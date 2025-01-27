# Income Tax Fraud Detection using Machine Learning

## Overview
This project focuses on developing a machine learning model to identify income tax fraud by analyzing taxpayers' financial data. The model compares six algorithms—**Logistic Regression, Decision Tree, Random Forest, Naive Bayes, k-Nearest Neighbors (k-NN), and Feedforward Neural Network**—to detect fraudulent patterns in tax returns. Logistic Regression emerged as the most effective algorithm, capturing both linear and non-linear relationships among variables for accurate fraud detection. The model is trained on an [OpenML dataset](https://www.openml.org) and deployed as an Android app using TensorFlow, enhancing accessibility for end-users.

## Key Features
- **Comparison of 6 ML Algorithms**: Evaluates Logistic Regression, Decision Tree, Random Forest, Naive Bayes, k-NN, and a Neural Network.
- **Linear & Non-Linear Pattern Detection**: Captures complex relationships in financial data for higher accuracy.
- **OpenML Dataset Integration**: Trained and tested on a publicly available dataset for transparency.
- **Android App Deployment**: A user-friendly app built with Android Studio and TensorFlow Lite for real-world predictions.

## Methodology
### Data & Algorithms
- **Dataset**: Financial data from OpenML, split into training and test sets.
- **Algorithms**:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Naive Bayes
  - k-Nearest Neighbors (k-NN)
  - Feedforward Neural Network
- **Evaluation**: Accuracy, precision, recall, and F1-score metrics on test data.

### Model Deployment
- The best-performing model (Logistic Regression) is converted to TensorFlow Lite and integrated into an Android app.
- Users can input financial data and receive fraud predictions directly on their devices.

## Results
- **Logistic Regression** achieved the highest accuracy in detecting tax fraud.
- Outperformed existing methods by capturing complex patterns in financial data.
- Detailed comparison metrics for all algorithms are provided in the [project report](link_to_report.md).

## Impact
- **Reduces Revenue Losses**: Helps governments identify fraudulent tax returns efficiently.
- **Promotes Fairness**: Ensures equitable tax compliance across individuals and businesses.
- **Cost-Effective Solution**: Affordable deployment via mobile apps for widespread use.

## Installation
### Prerequisites
- Android Studio (latest version)
- TensorFlow Lite dependency
- Python (for model training, optional)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Shivam-Narayan/Uncovered_Income_Tax_Fraud_Detection.git

2. Install dependendies:
'''bash
pip install -r requirements.txt  # For Python model training

3. Open the Android project in Android Studio:

-Import the android-app folder.

-Sync Gradle dependencies.

4. Run the app on an emulator or physical device.

## Usage
Input Financial Data: Enter taxpayer information (e.g., income, deductions, business expenses).

Predict Fraud Risk: The app processes the data and returns a fraud probability score.

Review Results: Flagged cases can be further investigated by authorities.

## Future Work
Expand dataset to include more diverse financial profiles.

Integrate real-time data processing for dynamic fraud detection.

Enhance model interpretability for policymakers.

## Contributing
Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request. For major changes, open an issue first to discuss proposed updates.

## License
This project is licensed under the MIT License. See LICENSE for details.

## Contact
For questions or collaborations, reach out to sshivam6495@gmail.com.
