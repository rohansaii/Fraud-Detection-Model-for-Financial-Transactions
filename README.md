# Fraud-Detection-Model-for-Financial-Transactions

# Fraud Detection Case Study

## Project Overview

This project focuses on building a robust fraud detection model for a large financial dataset. The primary goal is to identify fraudulent transactions accurately and provide actionable insights for the prevention and evaluation of anti-fraud measures. The dataset contains approximately 6.3 million rows of financial transaction data.

## Business Context

In the financial industry, detecting fraudulent transactions is crucial for minimizing losses and maintaining customer trust. This project aims to develop a machine learning model that can effectively flag suspicious transactions and offer insights into common fraud patterns, thereby enhancing the security infrastructure.

## Dataset

The dataset used for this project is `fraud.csv`, containing various features related to financial transactions, including:
- `step`: Maps a unit of time in the real world. Each step is 1 hour.
- `type`: Type of transaction (e.g., PAYMENT, TRANSFER, CASH_OUT).
- `amount`: The amount of the transaction.
- `nameOrig`: Customer who initiated the transaction.
- `oldbalanceOrg`: Original balance before the transaction.
- `newbalanceOrig`: New balance after the transaction.
- `nameDest`: Customer who is the recipient of the transaction.
- `oldbalanceDest`: Original balance of the recipient before the transaction.
- `newbalanceDest`: New balance of the recipient after the transaction.
- `isFraud`: This is the target variable, indicating if the transaction is fraudulent (1) or not (0).
- `isFlaggedFraud`: Indicates if the system flagged the transaction as fraudulent (1) or not (0).

## Problem Statement

Build a fraud detection model for a financial dataset (~6.3M rows). The goal is to detect fraudulent transactions and provide business insights for prevention and evaluation of anti-fraud measures. The target variable is `isFraud`.

## Key Findings & Insights

Through extensive Exploratory Data Analysis (EDA) and model training, several key insights were uncovered:

*   **Fraud Concentration:** Fraudulent activities are predominantly concentrated in `TRANSFER` and `CASH-OUT` transaction types.
*   **Balance Discrepancies in Fraud:** Many fraudulent cases exhibit a pattern where the original balance of the originator (`oldbalanceOrg`) is greater than 0, but their new balance (`newbalanceOrig`) becomes 0 after the transaction. This suggests a complete draining of the account.
*   **Limited `isFlaggedFraud` Utility:** The `isFlaggedFraud` feature, intended to flag fraud, rarely triggered in the dataset, indicating its limited practical usefulness as a direct indicator.
*   **Strong Predictive Features:** The transaction `amount` and the newly engineered `balanceDiffOrig` (difference between old and new balance for originator minus amount) and `balanceDiffDest` (difference between new and old balance for recipient minus amount) proved to be very strong predictors of fraudulent activity.

## Recommendations

Based on the insights gained, the following recommendations are proposed to enhance fraud prevention:

1.  **Real-time Anomaly Detection:** Implement real-time anomaly detection systems specifically for `TRANSFER` and `CASH-OUT` transactions, as these are high-risk areas.
2.  **Multi-Factor Authentication (MFA):** Require multi-factor authentication for transactions exceeding a certain threshold (e.g., > \$200,000) to add an extra layer of security.
3.  **Balance Consistency Checks:** Introduce automated checks to flag transactions where balance inconsistencies (e.g., `oldbalanceOrg` > 0 and `newbalanceOrig` = 0 for the originator) are observed, as these are strong indicators of fraud.
4.  **Geolocation/IP Monitoring:** Deploy geolocation and IP address monitoring to detect suspicious access patterns or transactions originating from unusual locations.

## Evaluating Effectiveness

To measure the success of implemented anti-fraud measures:

*   **Track Fraud Percentage:** Monitor the percentage of fraudulent transactions before and after the implementation of preventive measures.
*   **Measure Precision & Recall:** Continuously measure the Precision and Recall of the fraud detection system to ensure it effectively identifies fraud while minimizing false positives.
*   **Monitor False Positives:** Closely monitor the rate of false positives to avoid negatively impacting genuine customers and their transaction experience.

## Technical Details

### Data Cleaning & Feature Engineering

*   Dropped non-predictive columns: `nameOrig`, `nameDest`.
*   Engineered `balanceDiffOrig` and `balanceDiffDest` to capture balance inconsistencies.
*   One-hot encoded the `type` column to convert categorical transaction types into numerical format.
*   Checked for and handled missing values (though none were found in this dataset after initial cleaning).

### Exploratory Data Analysis (EDA)

*   Visualized the distribution of fraud vs. non-fraud transactions.
*   Analyzed the distribution of transaction types in fraudulent cases.
*   Examined the transaction amount distribution by fraud status.
*   Generated a correlation heatmap to understand relationships between features.

### Model Training

*   **Class Imbalance Handling:** Used `SMOTE` (Synthetic Minority Over-sampling Technique) to address the severe class imbalance in the target variable (`isFraud`).
*   **Data Splitting:** Split the dataset into training and testing sets using `train_test_split` with stratification to maintain class distribution.
*   **Feature Scaling:** Applied `StandardScaler` to normalize numerical features, which is crucial for many machine learning algorithms.
*   **Models Trained:**
    *   Logistic Regression
    *   Random Forest Classifier
    *   XGBoost Classifier

### Model Evaluation

*   Generated `classification_report` for all models.
*   Visualized `confusion_matrix` for the best-performing model (Random Forest).
*   Plotted `ROC Curve` and calculated `ROC-AUC score` to assess model discrimination ability.
*   Analyzed `Feature Importance` from the Random Forest model to understand key drivers of fraud.

## Tools and Libraries Used

*   **Python**
*   **Pandas**: For data manipulation and analysis.
*   **NumPy**: For numerical operations.
*   **Matplotlib**: For creating static, interactive, and animated visualizations.
*   **Seaborn**: For statistical data visualization.
*   **Scikit-learn**: For machine learning utilities (model selection, preprocessing, metrics).
    *   `train_test_split`
    *   `StandardScaler`
    *   `LogisticRegression`
    *   `RandomForestClassifier`
    *   `classification_report`, `confusion_matrix`, `roc_auc_score`, `roc_curve`
*   **XGBoost**: For gradient boosting.
    *   `XGBClassifier`
*   **Imblearn**: For handling imbalanced datasets.
    *   `SMOTE`

## How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
2.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    (You'll need to create a `requirements.txt` file based on the libraries listed above. You can generate one using `pip freeze > requirements.txt` after installing them.)
3.  **Download the dataset:**
    Ensure `fraud.csv` is placed in the root directory of the project. (If the dataset is too large for GitHub, you might need to provide instructions on where to download it from, e.g., Kaggle).
4.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook MultipleFiles/Fraud_Detection.ipynb
    ```
    Follow the cells in the notebook to execute the data cleaning, EDA, model training, and evaluation steps.

## Contact

For any questions or collaborations, feel free to reach out:

**Velagala Rohan Sai Kumar Reddy**
[Your LinkedIn Profile Link]
[Your Email Address]

---
