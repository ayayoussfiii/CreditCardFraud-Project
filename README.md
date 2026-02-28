# CreditCardFraud-Project HDBSCAN
Project Overview
This project applies a cluster-based modeling approach on the UCI Credit Card Default dataset to predict the likelihood of a client defaulting on their payment.
Instead of training a single global model, clients are first segmented into behavioral groups using HDBSCAN, then a dedicated prediction model is trained for each cluster. This approach yields more accurate predictions and cluster-specific financial risk insights.


  Dataset
  Size: 30,000 clients, 23 features
Target: DEFAULT — whether a client will default next month (binary: 0/1)
Features: Credit limit, payment history (6 months), bill amounts, payment amounts, demographics
<img width="251" height="194" alt="image" src="https://github.com/user-attachments/assets/217e5197-4f23-434f-9e84-50b4be87cf1d" />

<img width="491" height="207" alt="image" src="https://github.com/user-attachments/assets/8c864a0a-7c39-4658-a770-4c1f31a3e1f6" />

