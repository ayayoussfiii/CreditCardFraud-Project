# CreditCardFraud-Project
Project Overview
This project applies a cluster-based modeling approach on the UCI Credit Card Default dataset to predict the likelihood of a client defaulting on their payment.
Instead of training a single global model, clients are first segmented into behavioral groups using HDBSCAN, then a dedicated prediction model is trained for each cluster. This approach yields more accurate predictions and cluster-specific financial risk insights.
 #Dataset
  Size: 30,000 clients, 23 features
Target: DEFAULT — whether a client will default next month (binary: 0/1)
Features: Credit limit, payment history (6 months), bill amounts, payment amounts, demographics
Raw Data (UCI_Credit_Card.csv)
        ↓
  clean_data.py           → Feature engineering, normalization
        ↓
  cluster_transactions.py → HDBSCAN segmentation (3 clusters)
        ↓
  train_models.py         → Gradient Boosting + Naive Bayes per cluster
        ↓
  shap_analysis.py        → SHAP feature importance per cluster
        ↓
  app.py                  → Flask web interface
