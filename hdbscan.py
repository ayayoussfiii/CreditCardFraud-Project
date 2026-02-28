import pandas as pd
import numpy as np
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
import os

# -----------------------------
# 1. Charger le dataset nettoyé
# -----------------------------
csv_file = os.path.join(os.path.dirname(__file__), "../data/cleaned_data.csv")
df = pd.read_csv(csv_file)
print(f"Dataset chargé : {df.shape}")

# -----------------------------
# 2. Sélection des features pour le clustering
# -----------------------------
features = [
    "LIMIT_BAL",
    "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
]

X = df[features].copy()

# -----------------------------
# 3. Standardisation
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Standardisation terminée ✅")

# -----------------------------
# 4. Réduction PCA à 3 composantes
# -----------------------------
pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Variance expliquée par PCA : {pca.explained_variance_ratio_.sum():.2%} ✅")

# -----------------------------
# 5. HDBSCAN Clustering
# -----------------------------
print("Lancement HDBSCAN...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=300,
    min_samples=15,
    metric='euclidean',
    cluster_selection_method='eom',
    cluster_selection_epsilon=0.2
)
cluster_labels = clusterer.fit_predict(X_pca)

n_clusters_raw = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_outliers_raw = sum(cluster_labels == -1)
print(f"Clusters trouvés (avant réassignation) : {n_clusters_raw}")
print(f"Outliers détectés : {n_outliers_raw} ({n_outliers_raw/len(df)*100:.1f}%)")

# -----------------------------
# 6. Réassigner les outliers au cluster le plus proche
# -----------------------------
unique_clusters = [c for c in set(cluster_labels) if c != -1]
centroids = np.array([X_pca[cluster_labels == c].mean(axis=0) for c in unique_clusters])

outlier_mask = cluster_labels == -1
if outlier_mask.sum() > 0:
    closest, _ = pairwise_distances_argmin_min(X_pca[outlier_mask], centroids)
    cluster_labels[outlier_mask] = np.array(unique_clusters)[closest]
    print(f"✅ {outlier_mask.sum()} outliers réassignés au cluster le plus proche")

df["Cluster"] = cluster_labels

n_clusters_final = len(set(cluster_labels))
print(f"\n✅ Clustering terminé ! Nombre final de clusters : {n_clusters_final}")

# -----------------------------
# 7. Analyse des clusters
# -----------------------------
print("\n--- Répartition par cluster ---")
cluster_analysis = df.groupby("Cluster").agg(
    Nb_clients=("DEFAULT", "count"),
    Taux_defaut=("DEFAULT", "mean"),
    Limite_moy=("LIMIT_BAL", "mean"),
    Age_moy=("AGE", "mean")
).round(3)
print(cluster_analysis)

# -----------------------------
# 8. Sauvegarder
# -----------------------------
output_path = os.path.join(os.path.dirname(__file__), "../data/cleaned_data_with_clusters.csv")
df.to_csv(output_path, index=False)
print(f"\n✅ Dataset sauvegardé dans data/cleaned_data_with_clusters.csv")

# -----------------------------
# 9. Visualisations
# -----------------------------

# Plot 1 : Distribution des clusters
plt.figure(figsize=(10, 5))
sns.countplot(x="Cluster", data=df, palette="tab10", hue="Cluster", legend=False)
plt.title("Distribution des clusters HDBSCAN")
plt.xlabel("Cluster")
plt.ylabel("Nombre de clients")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "../results/cluster_distribution.png"))
plt.show()
print("✅ Plot distribution sauvegardé")

# Plot 2 : Taux de défaut par cluster
cluster_default = df.groupby("Cluster")["DEFAULT"].mean().reset_index()
plt.figure(figsize=(10, 5))
sns.barplot(x="Cluster", y="DEFAULT", data=cluster_default, palette="Reds", hue="Cluster", legend=False)
plt.title("Taux de défaut par cluster")
plt.xlabel("Cluster")
plt.ylabel("Taux de défaut")
plt.axhline(df["DEFAULT"].mean(), color="blue", linestyle="--", label=f"Moyenne globale ({df['DEFAULT'].mean():.2%})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "../results/default_rate_by_cluster.png"))
plt.show()
print("✅ Plot taux de défaut sauvegardé")

# Plot 3 : PCA 2D visualization
pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(12, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, cmap="tab10", alpha=0.4, s=5)
plt.colorbar(scatter, label="Cluster")
plt.title("Visualisation PCA 2D des clusters HDBSCAN")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "../results/pca_clusters.png"))
plt.show()
print("✅ Plot PCA 2D sauvegardé")