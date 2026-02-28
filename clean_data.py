import pandas as pd
import numpy as np
import os

# -----------------------------
# 1. Charger le dataset
# -----------------------------
csv_file = os.path.join(os.path.dirname(__file__), "../data/UCI_Credit_Card.csv")
df = pd.read_csv(csv_file)

print("Shape initial:", df.shape)

# -----------------------------
# 2. Renommer la colonne cible
# -----------------------------
df.rename(columns={"default.payment.next.month": "DEFAULT"}, inplace=True)

# -----------------------------
# 3. Supprimer la colonne ID (inutile)
# -----------------------------
df.drop(columns=["ID"], inplace=True)

# -----------------------------
# 4. Nettoyer les valeurs aberrantes
# EDUCATION : valeurs 0, 5, 6 non documentées → regrouper en "Autre" (4)
# -----------------------------
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

# MARRIAGE : valeur 0 non documentée → regrouper en "Autre" (3)
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

# -----------------------------
# 5. Feature Engineering
# -----------------------------
# Moyenne des retards de paiement
df["AVG_PAY_DELAY"] = df[["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]].mean(axis=1)

# Moyenne des montants de facture
df["AVG_BILL_AMT"] = df[["BILL_AMT1","BILL_AMT2","BILL_AMT3",
                          "BILL_AMT4","BILL_AMT5","BILL_AMT6"]].mean(axis=1)

# Moyenne des montants payés
df["AVG_PAY_AMT"] = df[["PAY_AMT1","PAY_AMT2","PAY_AMT3",
                         "PAY_AMT4","PAY_AMT5","PAY_AMT6"]].mean(axis=1)

# Ratio paiement / facture (capacité de remboursement)
df["PAY_RATIO"] = df["AVG_PAY_AMT"] / (df["AVG_BILL_AMT"] + 1)

# Log du crédit limite (réduire skewness)
df["LIMIT_BAL_log"] = np.log1p(df["LIMIT_BAL"])

# -----------------------------
# 6. Vérification finale
# -----------------------------
print("Shape final:", df.shape)
print("Valeurs nulles:", df.isnull().sum().sum())
print("Distribution DEFAULT:\n", df["DEFAULT"].value_counts())

# -----------------------------
# 7. Sauvegarder
# -----------------------------
output_path = os.path.join(os.path.dirname(__file__), "../data/cleaned_data.csv")
df.to_csv(output_path, index=False)
print("\n✅ Données nettoyées sauvegardées dans data/cleaned_data.csv")