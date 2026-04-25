# 🚨 Credit Card Fraud Detection using Machine Learning

## 📌 Overview

Credit card fraud is a critical issue in the financial industry, where fraudulent transactions represent only a very small fraction of total transactions but can cause significant losses. Detecting such transactions is challenging due to **extreme class imbalance** and evolving fraud patterns.

This project builds a **machine learning-based fraud detection system** that identifies suspicious transactions with high precision while minimizing false positives.

---

## 🎯 Objectives

* Detect fraudulent transactions accurately
* Handle highly imbalanced datasets
* Reduce false positives (important for real-world usability)
* Apply feature engineering to improve model performance

---

## 📊 Dataset

The dataset contains transaction-level information including:

* Transaction amount
* Timestamp and date
* Customer and merchant-related anonymized features
* Geographic information (latitude, longitude)

📌 **Note:** Fraud cases typically account for **<1% of total transactions**, making this a highly imbalanced classification problem.

---

## ⚙️ Approach

### 🔹 1. Data Preprocessing

* Handling missing values
* Sorting transactions by time
* Cleaning and structuring dataset

---

### 🔹 2. Feature Engineering (Key Strength of Project 🚀)

The following advanced features were created:

#### ⏱️ Time-Based Features

* Hour of transaction
* Day of week
* Time differences between transactions

#### ⚡ Transaction Velocity

* Number of transactions in:

  * Last 1 hour
  * Last 24 hours

#### 📈 Behavioral Features

* Deviation of transaction amount from user's average
* Spending pattern analysis

#### 🌍 Geospatial Feature

* Haversine distance between:

  * Customer location
  * Merchant location

👉 This helps detect abnormal location-based transactions.

---

### 🔹 3. Models Used

* 🌳 Random Forest (Primary Model)
* ⚡ XGBoost (Secondary Model)
* 🌲 Isolation Forest (for anomaly detection)

---

### 🔹 4. Handling Imbalanced Data

* SMOTE / SMOTEENN techniques
* Threshold tuning instead of default 0.5

---

### 🔹 5. Threshold Optimization (Important Insight 💡)

Instead of using the default classification threshold (0.5),
the model threshold was optimized (~0.9+) to:

✔ Reduce false positives
✔ Maintain good recall

👉 This is critical in fraud detection systems.

---

## 🏆 Results

| Model         | Precision | Recall                 | F1 Score | ROC-AUC |
| ------------- | --------- | ---------------------- | -------- | ------- |
| Random Forest | High      | Moderate               | Strong   | ~0.99   |
| XGBoost       | High      | Slightly better recall | Strong   | ~0.99   |

📌 The model achieves **high precision**, ensuring fewer legitimate transactions are flagged incorrectly.

---

## ▶️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/Sainihal1500/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebook:

```bash
jupyter notebook task-2.ipynb
```

---

## 📂 Project Structure

```bash
credit-card-fraud-detection/
│
├── task-2.ipynb      # Main notebook
├── README.md         # Project documentation
└── .gitignore
```

---

## 💡 Key Learnings

* Handling **imbalanced datasets** is critical in real-world ML
* Feature engineering can significantly outperform raw models
* Threshold tuning is more impactful than just changing models
* Fraud detection is more about **precision-recall tradeoff** than accuracy

---

## 🚀 Future Improvements

* Deploy as a **Streamlit web app**
* Build real-time fraud detection API
* Add deep learning models
* Improve feature engineering using sequence modeling

---

## 👨‍💻 Author

**Nihal**

---

## ⭐ If you found this useful

Give a ⭐ on GitHub and feel free to contribute!
