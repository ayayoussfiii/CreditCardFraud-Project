import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import shap

# ============================================
# 1. CHARGEMENT DES MODÈLES ET DONNÉES
# ============================================
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE_DIR, 'data', 'cleaned_data_with_clusters.csv')
MODELS_PATH  = os.path.join(BASE_DIR, 'results', 'models.pkl')
SHAP_PATH    = os.path.join(BASE_DIR, 'results', 'shap_plots')
os.makedirs(SHAP_PATH, exist_ok=True)

# Charger les modèles
with open(MODELS_PATH, 'rb') as f:
    models = pickle.load(f)

# Charger les données
df = pd.read_csv(DATA_PATH)

# Nettoyer les NaN
df = df.fillna(df.median(numeric_only=True))

CLUSTER_COL  = 'Cluster'
TARGET       = 'DEFAULT'
EXCLUDE_COLS = [CLUSTER_COL, TARGET]
features     = df.drop(columns=EXCLUDE_COLS).columns.tolist()

print("✅ Données et modèles chargés")
print(f"✅ Features : {len(features)}\n")

# ============================================
# 2. ANALYSE SHAP PAR CLUSTER
# ============================================
def analyser_shap(df, cluster_id):
    print(f"\n{'='*55}")
    print(f"  SHAP — CLUSTER {cluster_id}")
    print(f"{'='*55}")

    # Filtrer le cluster
    df_c     = df[df[CLUSTER_COL] == cluster_id].copy()
    X        = df_c[features]
    gb_model = models[cluster_id]['gradient_boosting']

    # Échantillon pour accélérer (max 500 lignes)
    sample_size = min(500, len(X))
    X_sample    = X.sample(sample_size, random_state=42)

    print(f"  Calcul SHAP sur {sample_size} clients...")

    # Créer l'explainer SHAP
    explainer   = shap.TreeExplainer(gb_model)
    shap_values = explainer.shap_values(X_sample)

    # ==========================================
    # GRAPHIQUE 1 : Summary Plot (vue globale)
    # ==========================================
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False,
        max_display=10
    )
    plt.title(f"Cluster {cluster_id} — Importance SHAP globale", fontsize=13)
    plt.tight_layout()
    path1 = os.path.join(SHAP_PATH, f'shap_bar_cluster{cluster_id}.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  💾 Sauvegardé → {path1}")

    # ==========================================
    # GRAPHIQUE 2 : Beeswarm (impact + direction)
    # ==========================================
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
        max_display=10
    )
    plt.title(f"Cluster {cluster_id} — Impact des features (SHAP)", fontsize=13)
    plt.tight_layout()
    path2 = os.path.join(SHAP_PATH, f'shap_beeswarm_cluster{cluster_id}.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  💾 Sauvegardé → {path2}")

    return explainer, shap_values, X_sample

# Lancer pour chaque cluster
shap_results = {}
for cluster_id in sorted(df[CLUSTER_COL].unique()):
    explainer, shap_values, X_sample = analyser_shap(df, cluster_id)
    shap_results[cluster_id] = {
        'explainer':   explainer,
        'shap_values': shap_values,
        'X_sample':    X_sample
    }

# ============================================
# 3. EXPLICATION D'UN CLIENT SPÉCIFIQUE
# ============================================
print(f"\n{'='*55}")
print("  EXPLICATION CLIENT INDIVIDUEL")
print(f"{'='*55}")

# Le même client que dans predict_new.py
nouveau_client = {
    'LIMIT_BAL'     : 50000,
    'SEX'           : 2,
    'EDUCATION'     : 2,
    'MARRIAGE'      : 1,
    'AGE'           : 35,
    'PAY_0'         : 0,
    'PAY_2'         : 0,
    'PAY_3'         : 0,
    'PAY_4'         : 0,
    'PAY_5'         : 0,
    'PAY_6'         : 0,
    'BILL_AMT1'     : 20000,
    'BILL_AMT2'     : 18000,
    'BILL_AMT3'     : 15000,
    'BILL_AMT4'     : 12000,
    'BILL_AMT5'     : 10000,
    'BILL_AMT6'     : 8000,
    'PAY_AMT1'      : 2000,
    'PAY_AMT2'      : 2000,
    'PAY_AMT3'      : 1500,
    'PAY_AMT4'      : 1500,
    'PAY_AMT5'      : 1000,
    'PAY_AMT6'      : 1000,
    'AVG_PAY_DELAY' : 0.0,
    'AVG_BILL_AMT'  : 13833.0,
    'AVG_PAY_AMT'   : 1500.0,
    'PAY_RATIO'     : 0.108,
    'LIMIT_BAL_log' : np.log(50000),
}

# Cluster assigné = 2 (d'après predict_new.py)
cluster_client   = 2
client_df        = pd.DataFrame([nouveau_client])[features]
explainer_client = shap_results[cluster_client]['explainer']
shap_client      = explainer_client.shap_values(client_df)

# Waterfall plot — explication feature par feature
print(f"\n  Client assigné au Cluster {cluster_client}")
print(f"  Voici pourquoi le modèle prédit DÉFAUT :\n")

# Afficher les contributions triées
contributions = pd.Series(shap_client[0], index=features).sort_values(key=abs, ascending=False)
print(f"  {'Feature':<20} {'Contribution':>12}  {'Direction'}")
print(f"  {'-'*45}")
for feat, val in contributions.head(10).items():
    direction = "↑ vers DÉFAUT" if val > 0 else "↓ vers NON-DÉFAUT"
    print(f"  {feat:<20} {val:>+12.4f}  {direction}")

# Graphique waterfall
plt.figure(figsize=(10, 6))
shap.waterfall_plot(
    shap.Explanation(
        values        = shap_client[0],
        base_values   = explainer_client.expected_value,
        data          = client_df.values[0],
        feature_names = features
    ),
    show=False,
    max_display=10
)
plt.title(f"Explication client — Cluster {cluster_client}", fontsize=13)
plt.tight_layout()
path3 = os.path.join(SHAP_PATH, f'shap_waterfall_client.png')
plt.savefig(path3, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  💾 Sauvegardé → {path3}")

print("\n✅ Analyse SHAP terminée !")