# Credit Card Fraud Detection Pipeline

A structured machine-learning workflow for highly imbalanced fraud datasets, with direct comparison of:

- Random Forest baseline (no balancing)
- Random Forest tuned + `class_weight='balanced'`
- Random Forest tuned + `SMOTEENN` balancing + `class_weight='balanced'`
- Random Forest tuned + `SMOTEENN` + probability calibration
- Isolation Forest baseline (scaled + threshold tuning)
- Isolation Forest tuned contamination (scaled + threshold tuning)
- Advanced boosting model (`HistGradientBoosting` for a Mac-friendly advanced baseline)

## What this pipeline covers

1. Data preprocessing
   - Missing value handling (`SimpleImputer`)
   - Temporal feature engineering (`trans_hour`, `trans_dayofweek`, `customer_age`)
   - Feature scaling where required (Isolation Forest path)
   - Feature/target split (`X`, `y`)
2. Class imbalance handling
   - No-balancing vs balancing comparison
   - `SMOTEENN` for combined over/under-sampling
3. Random Forest improvements
   - `RandomizedSearchCV` hyperparameter tuning
   - Tuning `n_estimators`, `max_depth`, `min_samples_split`, etc.
   - Uses `class_weight='balanced'`
4. Isolation Forest improvements
   - Input scaling before training
   - Baseline and tuned contamination comparison
   - Threshold tuning for better fraud F1 with accuracy floor
5. Further improvements implemented
   - Mac-friendly advanced boosting model (`HistGradientBoosting`)
   - Probability calibration (`CalibratedClassifierCV`)
   - Feature importance + permutation importance analysis
6. Evaluation metrics
   - Precision, Recall, F1-score, Confusion Matrix, ROC-AUC, Accuracy

## Setup

```bash
cd /Users/asainihal/Desktop/code/fraud_ml
/usr/local/bin/python3 -m pip install -r requirements.txt
```

## Run on your uploaded dataset

Your uploaded file uses target column `is_fraud`:

```bash
cd /Users/asainihal/Desktop/code
/usr/local/bin/python3 -m fraud_ml.fraud_pipeline --data-path "/Users/asainihal/Downloads/final_dataset.csv" --target-col is_fraud --output-json fraud_ml/results_summary_real.json
```

For parquet:

```bash
cd /Users/asainihal/Desktop/code
/usr/local/bin/python3 -m fraud_ml.fraud_pipeline --data-path "/absolute/path/to/your_dataset.parquet" --target-col Class
```

## Demo run (synthetic imbalanced data)

```bash
cd /Users/asainihal/Desktop/code
/usr/local/bin/python3 -m fraud_ml.fraud_pipeline --demo --target-col Class --output-json fraud_ml/results_summary_demo.json
```

## Run tests

```bash
cd /Users/asainihal/Desktop/code
/usr/local/bin/python3 -m pytest -q fraud_ml/tests/test_pipeline_smoke.py
```

## Notes

- The script enforces threshold search for Random Forest variants with an accuracy floor of 90% on training probabilities.
- Isolation Forest variants also use threshold tuning with an accuracy floor of 90%.
- Real-world fraud detection often prioritizes high recall/F1 for fraud class; acceptable false positives depend on business cost.
- Results are saved to `fraud_ml/results_summary.json`.
- The advanced model is intentionally lightweight for Mac runs.
