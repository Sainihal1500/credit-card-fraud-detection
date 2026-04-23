# Credit Card Fraud Detection Pipeline

A production-style fraud detection workflow for highly imbalanced transaction data.
It keeps the existing project structure, but improves preprocessing, feature engineering,
modeling, threshold tuning, evaluation, and artifact saving.

## Assignment rationale

- **Supervised learning is chosen** because fraud is usually a label-driven pattern problem, not just random noise.
- **IsolationForest is kept only as a baseline comparison** because anomaly detection often over-flags legitimate transactions and misses structured fraud patterns.
- **Gradient boosting is the main model** because it learns non-linear interactions across amount, timing, geography, and merchant context better than a pure anomaly detector.

## What it does

- Automatically detects numeric and categorical columns
- Applies one-hot encoding to categorical features
- Imputes missing values properly
- Engineers fraud-focused features:
  - hour, day-of-week, night flag, weekend flag
  - customer age from DOB
  - log amount, squared amount, amount-to-population ratio
  - merchant distance from customer coordinates
- Compares multiple models:
  - Random Forest baseline
  - Random Forest tuned with class weights
  - Random Forest with controlled SMOTE
  - HistGradientBoosting baseline
  - HistGradientBoosting with class weights
  - HistGradientBoosting with controlled SMOTE
  - Optional XGBoost if installed
  - IsolationForest baseline as an unsupervised reference only
- Tunes thresholds using F1 or recall instead of default `0.5`
- Saves results to JSON and the best model to `joblib`
- Prints a clear performance analysis: what worked, what failed, and the precision/recall tradeoff

## Setup

```bash
cd /Users/asainihal/Desktop/code/fraud_ml
/opt/anaconda3/bin/python -m pip install -r requirements.txt
```

## Run on your dataset

Use your real file:

```bash
cd /Users/asainihal/Desktop/code
/opt/anaconda3/bin/python -m fraud_ml.fraud_pipeline \
  --data-path "/Users/asainihal/Downloads/final_dataset.csv" \
  --target-col is_fraud \
  --max-rows 20000 \
  --output-json fraud_ml/results_summary_real.json \
  --model-path fraud_ml/best_model.joblib
```

If your dataset uses `Class` as the target:

```bash
cd /Users/asainihal/Desktop/code
/opt/anaconda3/bin/python -m fraud_ml.fraud_pipeline \
  --data-path "/absolute/path/to/creditcard.csv" \
  --target-col Class
```

## Demo run

```bash
cd /Users/asainihal/Desktop/code
/opt/anaconda3/bin/python -m fraud_ml.fraud_pipeline --demo --target-col Class
```

## Useful CLI options

- `--max-rows`: cap dataset size for faster Mac runs
- `--imbalance-strategy`: choose `smote` or `class_weight` for the preferred balancing setup
- `--threshold-metric`: choose `f1` or `recall`
- `--smote-ratio`: control the minority/majority ratio after SMOTE
- `--quiet`: reduce console logging

## Output

The pipeline prints a comparison table with:

- Precision
- Recall
- F1-score
- Accuracy
- ROC-AUC
- Confusion matrix counts
- Best threshold for each model

It also saves:

- JSON summary: `fraud_ml/results_summary.json`
- Best fitted model: `fraud_ml/best_model.joblib`

## Interpreting the results

- **High recall** means the model catches more fraud, which is usually the key business goal.
- **High precision** means fewer false alarms, which matters for customer friction and manual review cost.
- **Best F1** indicates the strongest balance between detecting fraud and limiting false positives.
- **False negatives** are usually more expensive than false positives in fraud detection, so the threshold is tuned accordingly.
- **Feature importance** should highlight transaction amount, time features, and location/merchant context.

## Optional advanced model

If you want to try XGBoost on top of the existing pipeline, install it separately:

```bash
/opt/anaconda3/bin/python -m pip install xgboost
```

The pipeline will detect it automatically and use it as the optional advanced model.

## Predicting new transactions

After training, you can score new records with the saved joblib artifact:

```python
from fraud_ml.fraud_pipeline import predict_new_transactions

records = ...  # pandas DataFrame with the same columns as training data
scored = predict_new_transactions("fraud_ml/best_model.joblib", records)
print(scored[["fraud_probability", "fraud_prediction"]].head())
```

## Notes

- IsolationForest is kept only as an unsupervised baseline and is not the main production model.
- For fraud detection, the most important metrics are usually recall and F1 for the fraud class.
- The best model is selected by F1 after threshold tuning with a minimum accuracy constraint.
