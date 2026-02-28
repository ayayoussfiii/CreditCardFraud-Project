import pandas as pd
import numpy as np
import pickle
import os

# ============================================
# 1. CHARGEMENT DES MODÈLES ET DONNÉES
# ============================================
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE_DIR, 'data', 'cleaned_data_with_clusters.csv')
MODELS_PATH  = os.path.join(BASE_DIR, 'results', 'models.pkl')

# Charger les modèles sauvegardés
with open(MODELS_PATH, 'rb') as f:
    models = pickle.load(f)

# Charger les données pour calculer les centroïdes
df = pd.read_csv(DATA_PATH)

CLUSTER_COL  = 'Cluster'
TARGET       = 'DEFAULT'
EXCLUDE_COLS = [CLUSTER_COL, TARGET]

print("✅ Modèles chargés")
print(f"✅ Clusters disponibles : {list(models.keys())}\n")

# ============================================
# 2. CALCUL DES CENTROÏDES PAR CLUSTER
# ============================================
# Le centroïde = point moyen de chaque cluster dans l'espace des features
features = df.drop(columns=EXCLUDE_COLS).columns.tolist()

centroids = {}
for cluster_id in sorted(df[CLUSTER_COL].unique()):
    df_c = df[df[CLUSTER_COL] == cluster_id]
    centroids[cluster_id] = df_c[features].mean().values

print("✅ Centroïdes calculés")
for cid, centroid in centroids.items():
    print(f"   Cluster {cid} : {len(df[df[CLUSTER_COL]==cid])} clients")

# ============================================
# 3. FONCTION D'ASSIGNATION AU CLUSTER
# ============================================
def assigner_cluster(client_features: np.ndarray) -> int:
    """
    Assigne un nouveau client au cluster le plus proche
    via la distance euclidienne aux centroïdes.
    """
    distances = {}
    for cluster_id, centroid in centroids.items():
        dist = np.linalg.norm(client_features - centroid)
        distances[cluster_id] = dist

    cluster_assigne = min(distances, key=distances.get)

    print(f"\n📍 Distances aux centroïdes :")
    for cid, dist in distances.items():
        marker = " ← assigné" if cid == cluster_assigne else ""
        print(f"   Cluster {cid} : {dist:.4f}{marker}")

    return cluster_assigne

# ============================================
# 4. FONCTION DE PRÉDICTION
# ============================================
def predire_client(client_dict: dict, seuil: float = 0.3) -> dict:
    """
    Prédit le risque de défaut d'un nouveau client.

    Args:
        client_dict : dictionnaire avec les features du client
        seuil       : seuil de décision (0.3 recommandé pour risque crédit)

    Returns:
        dict avec cluster, probabilité, et décision
    """
    # Créer le vecteur de features dans le bon ordre
    client_df      = pd.DataFrame([client_dict])[features]
    client_array   = client_df.values[0]

    # Assigner au cluster
    cluster_id     = assigner_cluster(client_array)

    # Récupérer le modèle du cluster
    gb_model       = models[cluster_id]['gradient_boosting']
    nb_model       = models[cluster_id]['naive_bayes']

    # Prédire avec Gradient Boosting
    proba_gb       = gb_model.predict_proba(client_df)[0][1]
    decision_gb    = "⚠️  DÉFAUT" if proba_gb >= seuil else "✅ NON-DÉFAUT"

    # Prédire avec Naive Bayes
    proba_nb       = nb_model.predict_proba(client_df)[0][1]
    decision_nb    = "⚠️  DÉFAUT" if proba_nb >= seuil else "✅ NON-DÉFAUT"

    return {
        'cluster':      cluster_id,
        'proba_gb':     proba_gb,
        'decision_gb':  decision_gb,
        'proba_nb':     proba_nb,
        'decision_nb':  decision_nb,
        'seuil':        seuil
    }

# ============================================
# 5. AFFICHAGE DU RÉSULTAT
# ============================================
def afficher_resultat(resultat: dict):
    print(f"\n{'='*50}")
    print(f"  RÉSULTAT DE LA PRÉDICTION")
    print(f"{'='*50}")
    print(f"  Cluster assigné      : {resultat['cluster']}")
    print(f"  Seuil utilisé        : {resultat['seuil']}")
    print(f"\n  Gradient Boosting    : {resultat['decision_gb']}")
    print(f"  Probabilité défaut   : {resultat['proba_gb']:.2%}")
    print(f"\n  Naive Bayes          : {resultat['decision_nb']}")
    print(f"  Probabilité défaut   : {resultat['proba_nb']:.2%}")
    print(f"{'='*50}")

    # Niveau de risque
    proba = resultat['proba_gb']
    if proba >= 0.7:
        niveau = "🔴 RISQUE ÉLEVÉ"
    elif proba >= 0.35:
        niveau = "🟠 RISQUE MODÉRÉ"
    elif proba >= 0.2:
        niveau = "🟡 RISQUE FAIBLE"
    else:
        niveau = "🟢 RISQUE TRÈS FAIBLE"

    print(f"\n  Niveau de risque : {niveau}")
    print(f"{'='*50}\n")

# ============================================
# 6. EXEMPLE — NOUVEAU CLIENT
# ============================================
# Remplis les valeurs de ton client ici
nouveau_client = {
    'LIMIT_BAL'     : 50000,   # Limite de crédit
    'SEX'           : 2,        # 1=Homme, 2=Femme
    'EDUCATION'     : 2,        # 1=grad, 2=université, 3=lycée
    'MARRIAGE'      : 1,        # 1=marié, 2=célibataire, 3=autre
    'AGE'           : 35,
    'PAY_0'         : 0,        # Statut paiement mois 1 (-1=OK, 0=min, 1=1mois retard...)
    'PAY_2'         : 0,
    'PAY_3'         : 0,
    'PAY_4'         : 0,
    'PAY_5'         : 0,
    'PAY_6'         : 0,
    'BILL_AMT1'     : 20000,    # Montant facture mois 1
    'BILL_AMT2'     : 18000,
    'BILL_AMT3'     : 15000,
    'BILL_AMT4'     : 12000,
    'BILL_AMT5'     : 10000,
    'BILL_AMT6'     : 8000,
    'PAY_AMT1'      : 2000,     # Montant payé mois 1
    'PAY_AMT2'      : 2000,
    'PAY_AMT3'      : 1500,
    'PAY_AMT4'      : 1500,
    'PAY_AMT5'      : 1000,
    'PAY_AMT6'      : 1000,
    'AVG_PAY_DELAY' : 0.0,      # Moyenne des retards
    'AVG_BILL_AMT'  : 13833.0,  # Moyenne des factures
    'AVG_PAY_AMT'   : 1500.0,   # Moyenne des paiements
    'PAY_RATIO'     : 0.108,    # AVG_PAY_AMT / AVG_BILL_AMT
    'LIMIT_BAL_log' : np.log(50000),  # Log de la limite
}

# Lancer la prédiction
resultat = predire_client(nouveau_client, seuil=0.3)
afficher_resultat(resultat)