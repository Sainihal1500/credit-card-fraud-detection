# 💳 Credit Card Fraud Detection System

## 📌 Overview

This project builds a machine learning pipeline to detect fraudulent credit card transactions using an imbalanced dataset.

The system compares multiple models and identifies the best-performing approach.

---

## 🎯 Problem Statement

Fraud detection is challenging because fraudulent transactions are very rare compared to normal transactions.

The goal is to accurately detect fraud while minimizing false alarms.

---

## 🧠 Models Used

* Random Forest (baseline + tuned)
* Isolation Forest (anomaly detection)
* HistGradientBoosting (advanced model)

---

## ⚙️ Techniques Applied

* Missing value handling (SimpleImputer)
* Feature engineering (time-based features)
* Class imbalance handling (SMOTEENN)
* Hyperparameter tuning (RandomizedSearchCV)
* Model calibration and threshold tuning

---

## 📊 Evaluation Metrics

* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

---

## 🏆 Results

* Random Forest → Strong performance
* Isolation Forest → Poor performance
* HistGradientBoosting → ⭐ Best model

---

## ▶️ How to Run

```bash
pip install -r fraud_ml/requirements.txt
python fraud_ml/fraud_pipeline.py --data-path final_dataset.csv --target-col is_fraud
```

---

## 📁 Output

* `results_summary_real.json` → contains model performance results

---

## 🚀 Future Improvements

* Deploy as API using Flask/FastAPI
* Real-time fraud detection system
* Deep learning models for improved accuracy

---

## 👨‍💻 Author

Nihal
