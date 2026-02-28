import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# ============================================
# 1. CHARGEMENT DES DONNÉES
# ============================================
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE_DIR, 'data', 'cleaned_data_with_clusters.csv')
RESULTS_PATH = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_PATH, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"✅ Données chargées : {df.shape}")

# ============================================
# 2. NETTOYAGE DES NaN
# ============================================
nan_count = df.isnull().sum().sum()
print(f"⚠️  Valeurs NaN détectées : {nan_count}")

if nan_count > 0:
    df = df.fillna(df.median(numeric_only=True))
    print(f"✅ NaN remplacés par la médiane")

print(f"✅ Clusters présents : {sorted(df['Cluster'].unique())}")
print(f"✅ Taille finale     : {df.shape}\n")

# ============================================
# 2. CONFIGURATION
# ============================================
TARGET       = 'DEFAULT'
CLUSTER_COL  = 'Cluster'
EXCLUDE_COLS = [CLUSTER_COL, TARGET]

# ============================================
# 3. FONCTION D'ENTRAÎNEMENT PAR CLUSTER
# ============================================
def train_cluster(df, cluster_id):
    print(f"\n{'='*55}")
    print(f"  CLUSTER {cluster_id}")
    print(f"{'='*55}")

    # --- Filtrage ---
    df_c = df[df[CLUSTER_COL] == cluster_id].copy()
    X = df_c.drop(columns=EXCLUDE_COLS)
    y = df_c[TARGET]

    print(f"  Taille         : {len(df_c)} clients")
    print(f"  Taux de défaut : {y.mean():.2%}")

    # --- Train / Test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # --- SMOTE (rééquilibrage des classes) ---
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"  Après SMOTE    : {sum(y_train_res==1)} défauts | {sum(y_train_res==0)} non-défauts")

    # ==========================================
    # GRADIENT BOOSTING
    # ==========================================
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb.fit(X_train_res, y_train_res)

    y_pred_gb  = gb.predict(X_test)
    y_proba_gb = gb.predict_proba(X_test)[:, 1]

    print(f"\n  --- Gradient Boosting ---")
    print(classification_report(y_test, y_pred_gb,
          target_names=['Non-défaut', 'Défaut']))
    print(f"  AUC-ROC : {roc_auc_score(y_test, y_proba_gb):.4f}")

    # Seuil ajusté à 0.3 pour améliorer le recall sur défaut
    y_pred_gb_adj = (y_proba_gb >= 0.3).astype(int)
    print(f"\n  --- Gradient Boosting (seuil=0.3) ---")
    print(classification_report(y_test, y_pred_gb_adj,
          target_names=['Non-défaut', 'Défaut']))

    # ==========================================
    # NAIVE BAYES (baseline)
    # ==========================================
    nb = GaussianNB()
    nb.fit(X_train_res, y_train_res)

    y_pred_nb  = nb.predict(X_test)
    y_proba_nb = nb.predict_proba(X_test)[:, 1]

    print(f"\n  --- Naive Bayes (baseline) ---")
    print(classification_report(y_test, y_pred_nb,
          target_names=['Non-défaut', 'Défaut']))
    print(f"  AUC-ROC : {roc_auc_score(y_test, y_proba_nb):.4f}")

    # ==========================================
    # MATRICES DE CONFUSION
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Cluster {cluster_id} — Matrices de confusion", fontsize=14)

    configs = [
        (y_pred_gb,     "Gradient Boosting (seuil=0.5)"),
        (y_pred_gb_adj, "Gradient Boosting (seuil=0.3)"),
        (y_pred_nb,     "Naive Bayes"),
    ]

    for ax, (y_pred, title) in zip(axes, configs):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        ax.set_xticklabels(['Non-défaut', 'Défaut'])
        ax.set_yticklabels(['Non-défaut', 'Défaut'])

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_PATH, f'confusion_matrix_cluster{cluster_id}.png')
    plt.savefig(fig_path, dpi=150)
    plt.show()
    print(f"  💾 Figure sauvegardée → {fig_path}")

    # ==========================================
    # IMPORTANCE DES FEATURES (GB)
    # ==========================================
    feature_names = list(X.columns)
    importances   = gb.feature_importances_
    indices       = np.argsort(importances)[::-1][:10]  # top 10

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(range(len(indices)),
            importances[indices],
            color='steelblue')
    ax2.set_xticks(range(len(indices)))
    ax2.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax2.set_title(f"Cluster {cluster_id} — Top 10 features importantes (GB)")
    ax2.set_ylabel("Importance")
    plt.tight_layout()

    feat_path = os.path.join(RESULTS_PATH, f'feature_importance_cluster{cluster_id}.png')
    plt.savefig(feat_path, dpi=150)
    plt.show()
    print(f"  💾 Feature importance sauvegardée → {feat_path}")

    return {
        'gradient_boosting': gb,
        'naive_bayes': nb,
        'feature_names': feature_names,
        'X_test': X_test,
        'y_test': y_test
    }

# ============================================
# 4. ENTRAÎNER POUR CHAQUE CLUSTER
# ============================================
models = {}

for cluster_id in sorted(df[CLUSTER_COL].unique()):
    models[cluster_id] = train_cluster(df, cluster_id)

# ============================================
# 5. TABLEAU RÉCAPITULATIF
# ============================================
print(f"\n{'='*55}")
print("  RÉCAPITULATIF — AUC-ROC par cluster")
print(f"{'='*55}")
print(f"  {'Cluster':<12} {'GB (0.5)':<15} {'GB (0.3)':<15} {'Naive Bayes'}")
print(f"  {'-'*50}")

for cluster_id, result in models.items():
    gb_model  = result['gradient_boosting']
    nb_model  = result['naive_bayes']
    X_test    = result['X_test']
    y_test    = result['y_test']

    proba_gb  = gb_model.predict_proba(X_test)[:, 1]
    proba_nb  = nb_model.predict_proba(X_test)[:, 1]

    auc_gb    = roc_auc_score(y_test, proba_gb)
    auc_nb    = roc_auc_score(y_test, proba_nb)

    print(f"  {cluster_id:<12} {auc_gb:<15.4f} {auc_gb:<15.4f} {auc_nb:.4f}")

# ============================================
# 6. SAUVEGARDER LES MODÈLES
# ============================================
models_path = os.path.join(RESULTS_PATH, 'models.pkl')

# Ne pas sauvegarder X_test/y_test dans le pkl final
models_to_save = {
    cid: {
        'gradient_boosting': v['gradient_boosting'],
        'naive_bayes':       v['naive_bayes'],
        'feature_names':     v['feature_names']
    }
    for cid, v in models.items()
}

with open(models_path, 'wb') as f:
    pickle.dump(models_to_save, f)

print(f"\n✅ Tous les modèles sauvegardés → {models_path}")
print("✅ Entraînement terminé !")