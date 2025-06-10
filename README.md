# startup-funding-prediction

This machine learning project predicts whether an Indian startup is likely to receive funding based on its features such as city, sector, subvertical, investment type, and founding date.

---

## 📌 Problem Statement

Startups play a vital role in economic development, but not all of them succeed in getting funded. Can we use machine learning to analyze patterns in funded vs. non-funded startups?

---

## 📊 Dataset Overview

- Contains 3,000+ Indian startup entries
- Fields include:
  - `Industry Vertical`
  - `SubVertical`
  - `City`
  - `Investment Type`
  - `Date`
  - `Funding Status` (target: Funded / Not Funded)

---

## 🧹 Data Cleaning & Preprocessing

- Cleaned inconsistent categorical values
- Extracted `Year` and `Month` from the `Date` column
- Handled missing data
- One-hot encoded categorical features
- Balanced dataset using **SMOTE** (Synthetic Minority Oversampling Technique)

---

## 🧠 Models Used

| Model | Key Details |
|-------|-------------|
| Logistic Regression | Baseline model |
| Random Forest | Tuned for better performance on imbalanced data |
| SMOTE | Applied before training to reduce class imbalance |

---

## 📈 Evaluation Metrics

- **Accuracy**
- **Precision, Recall, F1-Score**
- **ROC-AUC Score**
- **Confusion Matrix**

Due to data imbalance, ROC-AUC and F1-score were more reliable metrics than accuracy.

---

## 🖥️ Streamlit Web App

A user-friendly web app built with **Streamlit**, allowing users to:
- Select startup details (subvertical, city, industry, etc.)
- Predict whether the startup is likely to be funded
- Display prediction confidence

---

## ⚠️ Known Limitations

- The model currently predicts “Funded” more frequently due to:
  - Limited balanced training data
  - Fewer numerical/financial features
- Accuracy may vary — not meant for real-world investment decisions

---

## 💬 Why This Project Matters

- **End-to-End Pipeline**: From preprocessing to deployment
- **Deployment**: Bridging model development and real-world usage
- **Transparency**: Honest showcase of results and limitations
- **Mentorship Use of AI**: ChatGPT was used as a guide, not a solution

---

## 💡 Future Improvements

- Add financial metrics, team experience, and traction KPIs
- Try ensemble methods like XGBoost, CatBoost
- Hyperparameter tuning with GridSearch
- Use NLP on startup descriptions
- Improve UI with advanced filtering

---

## 🙋 Suggestions Welcome!

If you have ideas to improve the model, features to add, or want to collaborate, feel free to open an issue or connect with me!

---

## 📎 Project Files

- `startup.ipynb` — Jupyter notebook with complete preprocessing, training, evaluation
- `app_smote.py` — Streamlit application code
- `rf_model_smote.pkl` — Trained Random Forest model
- `encoded_columns.pkl` — Columns used during model training
- `startup_cleaned.csv` — Cleaned and preprocessed dataset

---

## 📌 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Streamlit
- Matplotlib, Seaborn

---

## 🔗 Connect

📧 Feel free to reach out via [LinkedIn](www.linkedin.com/in/madhumitha-s31) or raise an issue on GitHub!

