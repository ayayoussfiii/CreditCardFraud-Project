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

<img width="659" height="290" alt="image" src="https://github.com/user-attachments/assets/167a9698-9083-4266-a880-8f7bcfdb0cd4" />

<img width="922" height="411" alt="image" src="https://github.com/user-attachments/assets/296e5880-fd83-4e9b-ab83-0220cddbde6d" />

<img width="900" height="491" alt="image" src="https://github.com/user-attachments/assets/a46ee536-9422-48b9-a135-b5e007f7eb15" />

<img width="877" height="311" alt="image" src="https://github.com/user-attachments/assets/0f546c78-591b-4d02-a77d-6a5568c0a300" />

<img width="877" height="464" alt="image" src="https://github.com/user-attachments/assets/50c83b47-6639-426a-bf2f-7ed930466eef" />

<img width="875" height="422" alt="image" src="https://github.com/user-attachments/assets/5aaeed07-8370-4358-85ac-9b0d6fe81f88" />






