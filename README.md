# Credit Card Fraud Detection with Revenue Optimization Thresholding

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-orange?style=for-the-badge)](https://plotly.com/python/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**End-to-end fraud detection system** built on the classic Credit Card Fraud dataset (284,807 transactions, ~0.17% fraudulent).  
The project goes beyond standard accuracy metrics by implementing **cost-sensitive threshold optimization** to **maximize net revenue savings** — balancing fraud prevention against false-positive revenue friction.

**Live Interactive Dashboard** (Streamlit): https://credit-card-fraud-revenue-optimization.streamlit.app

## 🚀 Key Features

- **Business-Driven Optimization**  
  Cost-sensitive analysis with configurable FP cost ratio (default **0.01** – 1% revenue loss on false positives).  
  Optimized threshold ~0.5–0.6, achieving **~90% fraud recall** with strong net savings.

- **High Performance on Imbalanced Data**  
  - Average Precision: ~0.80–0.85  
  - Fraud Recall: **~90%** at balanced threshold  
  - False Positive Rate: <1% (controlled)

- **Root Cause Explainability**  
  SHAP summary plots identify top fraud drivers (V14, V17, V12, V10, V16).

- **Rich Visualizations**  
  - Net revenue savings vs. threshold curve  
  - Interactive Precision-Recall curve  
  - Annotated confusion matrix  
  - Dynamic estimated recall with threshold slider

- **Real-Time Scoring Dashboard**  
  Streamlit app with preset test cases, custom threshold slider, and live fraud probability + estimated detection rate.

## 📊 Results Summary

| Metric                              | Value                  | Interpretation                                   |
|-------------------------------------|------------------------|--------------------------------------------------|
| Average Precision (AP)              | ~0.80–0.85             | Excellent fraud ranking in imbalanced data       |
| Fraud Recall (at ~0.55 threshold)   | ~90%                   | High fraud capture with balanced FP cost         |
| Max Net Revenue Savings (Test Set)  | ~$20k–$40k             | Quantified business impact                       |

## 📁 Dataset
- **Source**: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- 284,807 anonymized transactions (492 fraud cases)  
- Features: `Time`, `Amount`, and 28 PCA-transformed variables (`V1`–`V28`)

## 🛠️ Tech Stack
- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, imbalanced-learn (SMOTE)
- XGBoost
- SHAP (explainability)
- Plotly (interactive plots)
- Streamlit (dashboard)
- joblib (model persistence)

## ⚙️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/s3achan/credit-card-fraud-revenue-optimization.git
   cd credit-card-fraud-revenue-optimization
