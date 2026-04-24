## 🏆 Final Models Selected

After evaluating multiple models, the following two were selected:

### 1️⃣ Random Forest (Primary Model)
- Precision: 0.87
- Recall: 0.46
- F1-score: 0.60
- Very low false positives

✅ Best overall balance → used as main model

---

### 2️⃣ HistGradientBoosting + SMOTE (Secondary Model)
- Precision: 0.74
- Recall: 0.50
- F1-score: 0.59

🚀 Better at detecting fraud (higher recall)

---

## 🧠 Key Insight

- Random Forest → safer (fewer false alarms)  
- HGB + SMOTE → more aggressive (catches more fraud)  

Final model choice depends on business needs.

## 📊 Final Results

| Model | Precision | Recall | F1 |
|------|----------|--------|----|
| Random Forest | 0.871 | 0.465 | 0.607 |
| HGB + SMOTE | 0.744 | 0.500 | 0.598 |
