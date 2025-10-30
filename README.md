🕵️‍♀️ Fraud Detection Using Machine Learning

This project builds a Fraud Detection Model using a real-world financial transactions dataset.
The goal is to classify whether a transaction is fraudulent (1) or legitimate (0) using machine learning techniques such as Decision Tree Classifier, Feature Engineering, and Cross-Validation.

📁 Project Structure
Fraud_Detection/
│
├── Fraud_Detection.ipynb     # Main Jupyter notebook (preprocessing, modeling, evaluation)
├── upload/
│   └── Fraud.csv             # Dataset used for training and testing
└── README.md                 # Project documentation

⚙️ Setup Instructions
1. Clone the repository
git clone https://github.com/rohansaii/Fraud-Detection-Model-for-Financial-Transactions.git
cd cd Fraud-Detection-Model-for-Financial-Transactions

2. Install dependencies

Make sure you have Python 3.8+ and install required packages:

pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

3. Open the notebook

You can launch the Jupyter notebook using:

jupyter notebook Fraud_Detection.ipynb

📊 Workflow Overview

Data Loading & Preprocessing

Read Fraud.csv

Handle categorical variables (type using pd.get_dummies)

Encode high-cardinality columns (nameOrig, nameDest)

Feature engineering:

errorBalanceOrig

errorBalanceDest

Drop irrelevant or redundant columns

Scaling & Splitting

Standardize features using StandardScaler

Split into train/test sets using stratified sampling

Handling Class Imbalance

Use SMOTE to balance the training data

Model Training

Train a Decision Tree Classifier

Evaluate on the unbalanced test set

Model Evaluation

Accuracy, AUC, and cross-validation scores

Perform 5-Fold Cross-Validation using ROC-AUC

🧠 Algorithms Used

Decision Tree Classifier
Simple, interpretable model ideal for exploratory fraud detection.

SMOTE (Synthetic Minority Oversampling Technique)
Used to balance the dataset by generating synthetic examples of minority class.

PCA (Optional in prior stages)
Used to reduce dimensionality and capture variance.

📈 Performance Metrics

ROC-AUC Score: Measures model’s ability to distinguish between fraud and non-fraud.

Cross-Validation Mean AUC: Provides stable generalization performance estimate.

🧰 Technologies Used
Category	Tools
Programming	Python 3.x
Libraries	pandas, numpy, scikit-learn, imbalanced-learn
Visualization	matplotlib, seaborn
Environment	Jupyter Notebook
🚀 Future Improvements

Add Random Forest and XGBoost for better performance

Include SHAP/Feature Importance analysis

Deploy model as a Flask API or Streamlit dashboard

📜 License

This project is licensed under the MIT License – feel free to use and modify for educational or research purposes.
