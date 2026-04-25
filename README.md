💳 Credit Card Transaction Anomaly Detection
🧠 Problem Statement
Credit card fraud is a major problem for financial institutions and consumers alike. Detecting fraudulent transactions accurately is incredibly challenging because fraud represents a tiny fraction of total transactions (often < 1%). This project aims to build a highly accurate, production-ready machine learning pipeline capable of detecting fraudulent transactions while minimizing false positives, leveraging advanced feature engineering, and evaluating powerful supervised tree ensembles.

📊 Dataset
The dataset (final_dataset.csv) contains transactional data including:

Features: Transaction amounts, dates, times, and anonymized merchant/customer data.
Engineered Features: Transaction velocity (1H/24H windows), Amount deviations from personal means, Geospatial Haversine distance, and Time-based cyclic features.
Imbalance: Highly imbalanced, with fraud accounting for less than 1% of transactions.
Note: Ensure the dataset is located in the data/ directory before running the pipeline.

🤖 Models Used
The pipeline utilizes multiple state-of-the-art models for comparison:

Random Forest Classifier (via Pipeline): A robust supervised ensemble model. We addressed the extreme class imbalance by using SMOTE (Synthetic Minority Over-sampling Technique) to synthetically upsample the training data, wrapped in an sklearn Pipeline alongside StandardScaler.
XGBoost: A gradient boosting classifier that natively handles class imbalance using scale_pos_weight.
📈 Results & Outputs
Note: Results may vary slightly depending on cross-validation splits.

Model	Accuracy	Precision	Recall	F1-Score	AUC-ROC
XGBoost	~0.996	~0.74	~0.71	~0.72	~0.99
Random Forest	~0.996	~0.73	~0.71	~0.72	~0.98
Threshold Tuning: Unlike standard implementations, we did not rely on the default 0.5 decision boundary. We optimized the threshold for Random Forest and XGBoost by maximizing the F1-score across a Precision-Recall Curve.

Model Visualizations
Receiver Operating Characteristic (ROC) Curve Demonstrating the strong discriminative ability of the Random Forest model across different thresholds.
ROC Curve
Review
ROC Curve

Confusion Matrix (Random Forest) Visualizing the trade-off between True Positives (caught fraud) and False Positives (false alarms) after applying the optimized decision threshold.
Confusion Matrix
Review
Confusion Matrix

Top Feature Importances Comparing which features the Random Forest and XGBoost models heavily relied on to identify anomalous transactions. Notice how the engineered features play a massive role.
Feature Importances
Review
Feature Importances

🚀 How to Run
Setup Environment: Ensure you have the required dependencies installed:

bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn joblib
Prepare Data: Create a data directory in the root folder and place your final_dataset.csv inside it.

bash
mkdir -p data
mv path/to/your/final_dataset.csv data/
Run the Notebook: You can run anomaly_detection.ipynb cell-by-cell in Jupyter, or execute the entire notebook from the terminal:

bash
jupyter nbconvert --to notebook --execute --inplace anomaly_detection.ipynb
Outputs:

The trained Random Forest Pipeline will be saved to models/random_forest.pkl.
Feature importances and Confusion Matrix / ROC Curves will be generated and displayed natively in the notebook.
