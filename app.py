from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
import json
from datetime import datetime

app = Flask(__name__)

# ============================================
# CHARGEMENT DES MODÈLES ET DONNÉES
# ============================================
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE_DIR, 'data', 'cleaned_data_with_clusters.csv')
MODELS_PATH  = os.path.join(BASE_DIR, 'results', 'models.pkl')
HISTORY_PATH = os.path.join(BASE_DIR, 'results', 'prediction_history.json')

with open(MODELS_PATH, 'rb') as f:
    models = pickle.load(f)

df = pd.read_csv(DATA_PATH)
df = df.fillna(df.median(numeric_only=True))

CLUSTER_COL  = 'Cluster'
TARGET       = 'DEFAULT'
EXCLUDE_COLS = [CLUSTER_COL, TARGET]
features     = df.drop(columns=EXCLUDE_COLS).columns.tolist()

# Centroïdes
centroids = {}
for cid in sorted(df[CLUSTER_COL].unique()):
    centroids[cid] = df[df[CLUSTER_COL] == cid][features].mean().values

# Stats globales
cluster_stats = {}
for cid in sorted(df[CLUSTER_COL].unique()):
    df_c = df[df[CLUSTER_COL] == cid]
    cluster_stats[int(cid)] = {
        'size':          int(len(df_c)),
        'default_rate':  round(float(df_c[TARGET].mean()) * 100, 1),
        'avg_limit':     round(float(df_c['LIMIT_BAL'].mean()), 0),
        'avg_age':       round(float(df_c['AGE'].mean()), 1),
        'avg_pay_delay': round(float(df_c['AVG_PAY_DELAY'].mean()), 2),
    }

print("✅ App Flask prête — http://localhost:5000")

# ============================================
# HISTORIQUE
# ============================================
def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'r') as f:
            return json.load(f)
    return []

def save_history(record):
    history = load_history()
    history.insert(0, record)
    history = history[:50]
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f)

# ============================================
# UTILITAIRES
# ============================================
def assigner_cluster(client_array):
    distances = {}
    for cid, centroid in centroids.items():
        distances[int(cid)] = float(np.linalg.norm(client_array - centroid))
    return min(distances, key=distances.get), distances

def generer_shap_waterfall(client_df, cluster_id):
    gb_model    = models[cluster_id]['gradient_boosting']
    explainer   = shap.TreeExplainer(gb_model)
    shap_values = explainer.shap_values(client_df)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_values[0],
            base_values   = explainer.expected_value,
            data          = client_df.values[0],
            feature_names = features
        ),
        show=False, max_display=8
    )
    plt.title(f"Contribution des variables - Cluster {cluster_id}", fontsize=11, pad=15)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generer_shap_contributions(client_df, cluster_id):
    gb_model    = models[cluster_id]['gradient_boosting']
    explainer   = shap.TreeExplainer(gb_model)
    shap_values = explainer.shap_values(client_df)
    contribs    = pd.Series(shap_values[0], index=features).sort_values(key=abs, ascending=False).head(6)
    result = []
    for feat, val in contribs.items():
        result.append({
            'feature':    feat,
            'value':      float(round(val, 4)),
            'direction':  'defaut' if val > 0 else 'safe',
            'client_val': float(client_df[feat].values[0])
        })
    return result

# ============================================
# HTML INTÉGRÉ - VERSION CLAIRE ET PRO
# ============================================
CLUSTER_STATS_JSON = json.dumps(cluster_stats)

HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>CreditRisk Pro • Analyse Scoring</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
<style>
  /* ===== VARIABLES MODE CLAIR ===== */
  :root {
    --bg: #f8fafc;
    --surface: #ffffff;
    --surface-2: #f1f5f9;
    --border: #e2e8f0;
    --text: #0f172a;
    --text-light: #475569;
    --text-lighter: #64748b;
    --accent: #2563eb;
    --accent-light: #3b82f6;
    --accent-soft: #dbeafe;
    --success: #10b981;
    --success-soft: #d1fae5;
    --warning: #f59e0b;
    --warning-soft: #fed7aa;
    --danger: #ef4444;
    --danger-soft: #fee2e2;
    --info: #6366f1;
    --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.05), 0 4px 6px -4px rgb(0 0 0 / 0.05);
    --radius: 12px;
    --radius-sm: 8px;
    --font: 'Inter', sans-serif;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }
  
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    line-height: 1.5;
  }

  /* ===== HEADER ===== */
  .header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 1rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 10;
    backdrop-filter: blur(8px);
    background: rgba(255,255,255,0.9);
  }

  .logo {
    font-weight: 600;
    font-size: 1.25rem;
    color: var(--text);
    letter-spacing: -0.02em;
  }
  
  .logo span {
    color: var(--accent);
    background: var(--accent-soft);
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
    margin-left: 0.5rem;
    font-size: 0.75rem;
  }

  .nav-tabs {
    display: flex;
    gap: 0.5rem;
  }

  .nav-tab {
    padding: 0.5rem 1rem;
    border: none;
    background: transparent;
    color: var(--text-light);
    font-weight: 500;
    font-size: 0.9rem;
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: all 0.2s;
  }

  .nav-tab:hover {
    background: var(--surface-2);
    color: var(--text);
  }

  .nav-tab.active {
    background: var(--accent-soft);
    color: var(--accent);
    font-weight: 600;
  }

  /* ===== CONTAINER ===== */
  .container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem;
  }

  .page { display: none; }
  .page.active { display: block; }

  .page-title {
    font-size: 1.5rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
  }

  .page-sub {
    color: var(--text-light);
    font-size: 0.9rem;
    margin-bottom: 2rem;
  }

  /* ===== DASHBOARD CARDS ===== */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.5rem;
  }

  .cluster-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    box-shadow: var(--shadow);
    transition: all 0.2s;
  }

  .cluster-card:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-2px);
  }

  .cluster-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-lighter);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
  }

  .cluster-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 1.25rem;
  }

  .cluster-stat {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.75rem;
    font-size: 0.9rem;
  }

  .cluster-stat-label {
    color: var(--text-light);
  }

  .cluster-stat-val {
    font-weight: 600;
    font-family: 'Inter', monospace;
  }

  .risk-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    margin-top: 0.75rem;
  }

  .risk-high { background: var(--danger-soft); color: var(--danger); }
  .risk-medium { background: var(--warning-soft); color: #b45309; }
  .risk-low { background: var(--success-soft); color: var(--success); }

  /* ===== FORMULAIRE ===== */
  .form-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
  }

  .form-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    box-shadow: var(--shadow);
  }

  .section-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 1.25rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
  }

  .field-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }

  .field {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .field label {
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text-light);
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }

  .field input,
  .field select {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.6rem 0.75rem;
    color: var(--text);
    font-family: 'Inter', monospace;
    font-size: 0.85rem;
    transition: all 0.2s;
    outline: none;
  }

  .field input:focus,
  .field select:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-soft);
  }

  .btn-predict {
    width: 100%;
    margin-top: 1.5rem;
    padding: 1rem;
    background: var(--accent);
    border: none;
    border-radius: var(--radius-sm);
    color: white;
    font-weight: 600;
    font-size: 0.95rem;
    cursor: pointer;
    transition: all 0.2s;
    letter-spacing: 0.02em;
  }

  .btn-predict:hover {
    background: var(--accent-light);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
  }

  .btn-predict:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
  }

  /* ===== RÉSULTATS ===== */
  .result-panel {
    margin-top: 2rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem;
    box-shadow: var(--shadow-lg);
    display: none;
  }

  .result-panel.visible { display: block; }

  .result-header {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.5rem;
    margin-bottom: 2rem;
  }

  .result-metric {
    background: var(--surface-2);
    border-radius: var(--radius-sm);
    padding: 1.25rem;
    text-align: center;
  }

  .result-metric-label {
    font-size: 0.7rem;
    color: var(--text-light);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
  }

  .result-metric-val {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'Inter', monospace;
    line-height: 1.2;
  }

  .val-green { color: var(--success); }
  .val-yellow { color: #b45309; }
  .val-orange { color: #c2410c; }
  .val-red { color: var(--danger); }

  .distances {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
  }

  .dist-chip {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.4rem 1rem;
    font-family: 'Inter', monospace;
    font-size: 0.8rem;
  }

  .dist-chip.assigned {
    background: var(--accent-soft);
    border-color: var(--accent);
    color: var(--accent);
    font-weight: 500;
  }

  /* ===== SHAP VISUALISATION ===== */
  .shap-section {
    margin-top: 2rem;
  }

  .shap-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-light);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
  }

  .shap-bars {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    margin-bottom: 2rem;
  }

  .shap-bar-row {
    display: grid;
    grid-template-columns: 120px 1fr 60px;
    gap: 0.75rem;
    align-items: center;
  }

  .shap-feat {
    font-family: 'Inter', monospace;
    font-size: 0.8rem;
    color: var(--text);
    font-weight: 500;
    text-align: right;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .shap-bar-track {
    background: var(--surface-2);
    border-radius: 4px;
    height: 24px;
    overflow: hidden;
  }

  .shap-bar-fill {
    height: 100%;
    border-radius: 4px;
    display: flex;
    align-items: center;
    padding: 0 0.5rem;
    font-size: 0.7rem;
    font-weight: 600;
    font-family: 'Inter', monospace;
    white-space: nowrap;
    transition: width 0.5s ease;
  }

  .shap-bar-fill.defaut {
    background: linear-gradient(90deg, #fee2e2, #fecaca);
    color: var(--danger);
  }

  .shap-bar-fill.safe {
    background: linear-gradient(90deg, #d1fae5, #a7f3d0);
    color: #065f46;
  }

  .shap-val {
    font-family: 'Inter', monospace;
    font-size: 0.8rem;
    color: var(--text-light);
    font-weight: 500;
  }

  .shap-img {
    width: 100%;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
    background: white;
    padding: 0.5rem;
  }

  /* ===== HISTORIQUE ===== */
  .history-table {
    width: 100%;
    border-collapse: collapse;
  }

  .history-table th {
    text-align: left;
    padding: 0.75rem 1rem;
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-lighter);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border);
  }

  .history-table td {
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    border-bottom: 1px solid var(--border);
  }

  .history-table tr:hover td {
    background: var(--surface-2);
  }

  .badge {
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
  }

  /* ===== LOADER ===== */
  .loader {
    display: none;
    text-align: center;
    padding: 3rem;
  }

  .loader.visible { display: block; }

  .spinner {
    width: 40px;
    height: 40px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: 0 auto 1rem;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

  .empty-state {
    text-align: center;
    padding: 4rem;
    color: var(--text-light);
  }

  .empty-icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.5; }

  /* ===== RESPONSIVE ===== */
  @media (max-width: 768px) {
    .form-grid,
    .stats-grid,
    .result-header {
      grid-template-columns: 1fr;
    }
    
    .container { padding: 1rem; }
    .header { flex-direction: column; gap: 1rem; }
  }
</style>
</head>
<body>
<div class="header">
  <div class="logo">CreditRisk <span>PRO</span></div>
  <div class="nav-tabs">
    <button class="nav-tab active" onclick="showPage('dashboard',this)">Dashboard</button>
    <button class="nav-tab" onclick="showPage('predict',this)">Analyse Scoring</button>
    <button class="nav-tab" onclick="showPage('history',this);loadHistory()">Historique</button>
  </div>
</div>

<div class="container">
  <!-- DASHBOARD -->
  <div class="page active" id="page-dashboard">
    <h1 class="page-title">Vue d'ensemble des segments</h1>
    <p class="page-sub">Analyse des clusters de risque • Données mises à jour</p>
    <div class="stats-grid" id="stats-grid"></div>
  </div>

  <!-- PREDICTION -->
  <div class="page" id="page-predict">
    <h1 class="page-title">Analyse scoring client</h1>
    <p class="page-sub">Saisissez les informations pour évaluer le risque de défaut</p>
    
    <div class="form-grid">
      <div>
        <div class="form-section">
          <div class="section-title">Profil client</div>
          <div class="field-grid">
            <div class="field">
              <label>Limite crédit (NT$)</label>
              <input type="number" id="limit_bal" value="50000" step="1000"/>
            </div>
            <div class="field">
              <label>Âge</label>
              <input type="number" id="age" value="35" min="18" max="100"/>
            </div>
            <div class="field">
              <label>Sexe</label>
              <select id="sex">
                <option value="2">Femme</option>
                <option value="1">Homme</option>
              </select>
            </div>
            <div class="field">
              <label>Statut marital</label>
              <select id="marriage">
                <option value="1">Marié(e)</option>
                <option value="2">Célibataire</option>
                <option value="3">Autre</option>
              </select>
            </div>
            <div class="field">
              <label>Éducation</label>
              <select id="education">
                <option value="1">Master/Doctorat</option>
                <option value="2" selected>Université</option>
                <option value="3">Lycée</option>
                <option value="4">Autre</option>
              </select>
            </div>
          </div>
        </div>
        
        <div class="form-section" style="margin-top:1.5rem">
          <div class="section-title">Historique paiement</div>
          <div class="field-grid">
            <div class="field"><label>PAY_0</label><select id="pay_0"><option value="-1">En avance</option><option value="0" selected>Minimum</option><option value="1">1 mois retard</option><option value="2">2 mois</option><option value="3">3 mois+</option></select></div>
            <div class="field"><label>PAY_2</label><select id="pay_2"><option value="-1">En avance</option><option value="0" selected>Minimum</option><option value="1">1 mois</option><option value="2">2 mois</option></select></div>
            <div class="field"><label>PAY_3</label><select id="pay_3"><option value="-1">En avance</option><option value="0" selected>Minimum</option><option value="1">1 mois</option><option value="2">2 mois</option></select></div>
            <div class="field"><label>PAY_4</label><select id="pay_4"><option value="-1">En avance</option><option value="0" selected>Minimum</option><option value="1">1 mois</option><option value="2">2 mois</option></select></div>
            <div class="field"><label>PAY_5</label><select id="pay_5"><option value="-1">En avance</option><option value="0" selected>Minimum</option><option value="1">1 mois</option></select></div>
            <div class="field"><label>PAY_6</label><select id="pay_6"><option value="-1">En avance</option><option value="0" selected>Minimum</option><option value="1">1 mois</option></select></div>
          </div>
        </div>
      </div>

      <div>
        <div class="form-section">
          <div class="section-title">Montants factures</div>
          <div class="field-grid">
            <div class="field"><label>BILL_AMT1</label><input type="number" id="bill_amt1" value="20000"/></div>
            <div class="field"><label>BILL_AMT2</label><input type="number" id="bill_amt2" value="18000"/></div>
            <div class="field"><label>BILL_AMT3</label><input type="number" id="bill_amt3" value="15000"/></div>
            <div class="field"><label>BILL_AMT4</label><input type="number" id="bill_amt4" value="12000"/></div>
            <div class="field"><label>BILL_AMT5</label><input type="number" id="bill_amt5" value="10000"/></div>
            <div class="field"><label>BILL_AMT6</label><input type="number" id="bill_amt6" value="8000"/></div>
          </div>
        </div>

        <div class="form-section" style="margin-top:1.5rem">
          <div class="section-title">Montants payés</div>
          <div class="field-grid">
            <div class="field"><label>PAY_AMT1</label><input type="number" id="pay_amt1" value="2000"/></div>
            <div class="field"><label>PAY_AMT2</label><input type="number" id="pay_amt2" value="2000"/></div>
            <div class="field"><label>PAY_AMT3</label><input type="number" id="pay_amt3" value="1500"/></div>
            <div class="field"><label>PAY_AMT4</label><input type="number" id="pay_amt4" value="1500"/></div>
            <div class="field"><label>PAY_AMT5</label><input type="number" id="pay_amt5" value="1000"/></div>
            <div class="field"><label>PAY_AMT6</label><input type="number" id="pay_amt6" value="1000"/></div>
          </div>
        </div>

        <button class="btn-predict" id="btn-predict" onclick="predict()">
          Calculer le score de risque
        </button>
      </div>
    </div>

    <div class="loader" id="loader">
      <div class="spinner"></div>
      <p style="color:var(--text-light)">Analyse en cours...</p>
    </div>

    <div class="result-panel" id="result-panel">
      <div class="section-title">Résultats de l'analyse</div>
      
      <div class="result-header">
        <div class="result-metric">
          <div class="result-metric-label">Cluster assigné</div>
          <div class="result-metric-val val-green" id="res-cluster">—</div>
        </div>
        <div class="result-metric">
          <div class="result-metric-label">Probabilité défaut</div>
          <div class="result-metric-val" id="res-proba">—</div>
        </div>
        <div class="result-metric">
          <div class="result-metric-label">Niveau risque</div>
          <div class="result-metric-val" id="res-risk">—</div>
        </div>
      </div>

      <div class="shap-section">
        <div class="shap-title">Distances aux centroïdes</div>
        <div class="distances" id="res-distances"></div>
      </div>

      <div class="shap-section">
        <div class="shap-title">Facteurs influençant la décision</div>
        <div class="shap-bars" id="shap-bars"></div>
      </div>

      <div class="shap-section">
        <div class="shap-title">Visualisation SHAP</div>
        <img id="shap-img" class="shap-img" src="" alt="SHAP Waterfall Plot"/>
      </div>
    </div>
  </div>

  <!-- HISTORIQUE -->
  <div class="page" id="page-history">
    <h1 class="page-title">Historique des analyses</h1>
    <p class="page-sub">50 dernières prédictions</p>
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);overflow:auto">
      <div id="history-container">
        <div class="empty-state">
          <div class="empty-icon">📊</div>
          <p>Aucune analyse pour le moment</p>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const clusterStats = """ + CLUSTER_STATS_JSON + """;
const clusterNames = {0:'🔴 Segment Risque Élevé',1:'🟢 Segment Sain',2:'🟠 Segment Fragile'};
const colorClass = {'TRÈS FAIBLE':'val-green','FAIBLE':'val-yellow','MODÉRÉ':'val-orange','ÉLEVÉ':'val-red'};

function initDashboard() {
  const grid = document.getElementById('stats-grid');
  grid.innerHTML = '';
  Object.entries(clusterStats).forEach(([cid,stats]) => {
    const rate = stats.default_rate;
    const rClass = rate > 50 ? 'risk-high' : rate > 30 ? 'risk-medium' : 'risk-low';
    const rLabel = rate > 50 ? 'Risque Élevé' : rate > 30 ? 'Risque Modéré' : 'Risque Faible';
    const rColor = rate > 50 ? 'var(--danger)' : rate > 30 ? 'var(--warning)' : 'var(--success)';
    
    grid.innerHTML += `
      <div class="cluster-card">
        <div class="cluster-label">Cluster ${cid}</div>
        <div class="cluster-title">${clusterNames[cid]}</div>
        <div class="cluster-stat">
          <span class="cluster-stat-label">Effectif</span>
          <span class="cluster-stat-val">${stats.size.toLocaleString()}</span>
        </div>
        <div class="cluster-stat">
          <span class="cluster-stat-label">Taux défaut</span>
          <span class="cluster-stat-val" style="color:${rColor}">${stats.default_rate}%</span>
        </div>
        <div class="cluster-stat">
          <span class="cluster-stat-label">Limite moyenne</span>
          <span class="cluster-stat-val">${stats.avg_limit.toLocaleString()} NT$</span>
        </div>
        <div class="cluster-stat">
          <span class="cluster-stat-label">Âge moyen</span>
          <span class="cluster-stat-val">${stats.avg_age} ans</span>
        </div>
        <div class="cluster-stat">
          <span class="cluster-stat-label">Délai paiement</span>
          <span class="cluster-stat-val">${stats.avg_pay_delay} mois</span>
        </div>
        <span class="risk-badge ${rClass}">${rLabel}</span>
      </div>
    `;
  });
}
initDashboard();

function showPage(name, btn) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  btn.classList.add('active');
}

function getVal(id) { return document.getElementById(id).value; }

async function predict() {
  const btn = document.getElementById('btn-predict');
  const loader = document.getElementById('loader');
  const panel = document.getElementById('result-panel');
  
  btn.disabled = true;
  loader.classList.add('visible');
  panel.classList.remove('visible');

  const payload = {
    limit_bal: getVal('limit_bal'), age: getVal('age'), sex: getVal('sex'),
    marriage: getVal('marriage'), education: getVal('education'),
    pay_0: getVal('pay_0'), pay_2: getVal('pay_2'), pay_3: getVal('pay_3'),
    pay_4: getVal('pay_4'), pay_5: getVal('pay_5'), pay_6: getVal('pay_6'),
    bill_amt1: getVal('bill_amt1'), bill_amt2: getVal('bill_amt2'), bill_amt3: getVal('bill_amt3'),
    bill_amt4: getVal('bill_amt4'), bill_amt5: getVal('bill_amt5'), bill_amt6: getVal('bill_amt6'),
    pay_amt1: getVal('pay_amt1'), pay_amt2: getVal('pay_amt2'), pay_amt3: getVal('pay_amt3'),
    pay_amt4: getVal('pay_amt4'), pay_amt5: getVal('pay_amt5'), pay_amt6: getVal('pay_amt6'),
  };

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.error);

    // Mise à jour résultats
    document.getElementById('res-cluster').textContent = 'Cluster ' + data.cluster;
    
    const probaEl = document.getElementById('res-proba');
    probaEl.textContent = data.proba_gb + '%';
    probaEl.className = 'result-metric-val ' + (colorClass[data.risk_level] || 'val-green');
    
    const riskEl = document.getElementById('res-risk');
    riskEl.textContent = data.risk_level;
    riskEl.className = 'result-metric-val ' + (colorClass[data.risk_level] || 'val-green');

    // Distances
    const distDiv = document.getElementById('res-distances');
    distDiv.innerHTML = '';
    for (const [cid, dist] of Object.entries(data.distances)) {
      const chip = document.createElement('div');
      chip.className = 'dist-chip' + (parseInt(cid) === data.cluster ? ' assigned' : '');
      chip.textContent = `Cluster ${cid}: ${Number(dist).toFixed(0)}` + (parseInt(cid) === data.cluster ? ' • assigné' : '');
      distDiv.appendChild(chip);
    }

    // SHAP bars
    const barsDiv = document.getElementById('shap-bars');
    barsDiv.innerHTML = '';
    const maxVal = Math.max(...data.shap_contributions.map(c => Math.abs(c.value)));
    
    data.shap_contributions.forEach(c => {
      const pct = Math.min((Math.abs(c.value) / maxVal) * 100, 100);
      barsDiv.innerHTML += `
        <div class="shap-bar-row">
          <div class="shap-feat">${c.feature}</div>
          <div class="shap-bar-track">
            <div class="shap-bar-fill ${c.direction}" style="width: ${pct}%">
              ${c.value > 0 ? '↑ défaut' : '↓ sûr'}
            </div>
          </div>
          <div class="shap-val">${c.value > 0 ? '+' : ''}${c.value.toFixed(3)}</div>
        </div>
      `;
    });

    document.getElementById('shap-img').src = 'data:image/png;base64,' + data.shap_img;
    panel.classList.add('visible');
    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });

  } catch (e) {
    alert('Erreur : ' + e.message);
  } finally {
    btn.disabled = false;
    loader.classList.remove('visible');
  }
}

async function loadHistory() {
  const res = await fetch('/history');
  const history = await res.json();
  const container = document.getElementById('history-container');
  
  if (!history.length) {
    container.innerHTML = '<div class="empty-state"><div class="empty-icon">📊</div><p>Aucune analyse pour le moment</p></div>';
    return;
  }

  const colorMap = {
    'TRÈS FAIBLE': '#10b981',
    'FAIBLE': '#f59e0b',
    'MODÉRÉ': '#c2410c',
    'ÉLEVÉ': '#ef4444'
  };
  
  const bgMap = {
    'TRÈS FAIBLE': '#d1fae5',
    'FAIBLE': '#fed7aa',
    'MODÉRÉ': '#ffedd5',
    'ÉLEVÉ': '#fee2e2'
  };

  let html = `<table class="history-table">
    <thead>
      <tr>
        <th>Date</th>
        <th>Cluster</th>
        <th>Prob. GB</th>
        <th>Risque</th>
        <th>Âge</th>
        <th>Limite</th>
      </tr>
    </thead>
    <tbody>`;

  history.forEach(h => {
    const color = colorMap[h.risk_level] || '#64748b';
    const bg = bgMap[h.risk_level] || 'transparent';
    html += `<tr>
      <td style="font-size:0.8rem;color:var(--text-light)">${h.timestamp}</td>
      <td><span class="badge" style="background:${bg};color:${color}">C${h.cluster}</span></td>
      <td style="font-weight:600">${h.proba_gb}%</td>
      <td><span class="badge" style="background:${bg};color:${color}">${h.risk_level}</span></td>
      <td>${h.age} ans</td>
      <td>${Number(h.limit_bal).toLocaleString()} NT$</td>
    </tr>`;
  });

  html += '</tbody></table>';
  container.innerHTML = html;
}
</script>
</body>
</html>"""

# ============================================
# ROUTES (inchangées)
# ============================================
@app.route('/')
def index():
    return HTML

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        client_dict = {
            'LIMIT_BAL' : float(data['limit_bal']),
            'SEX'       : int(data['sex']),
            'EDUCATION' : int(data['education']),
            'MARRIAGE'  : int(data['marriage']),
            'AGE'       : int(data['age']),
            'PAY_0'     : int(data['pay_0']),
            'PAY_2'     : int(data['pay_2']),
            'PAY_3'     : int(data['pay_3']),
            'PAY_4'     : int(data['pay_4']),
            'PAY_5'     : int(data['pay_5']),
            'PAY_6'     : int(data['pay_6']),
            'BILL_AMT1' : float(data['bill_amt1']),
            'BILL_AMT2' : float(data['bill_amt2']),
            'BILL_AMT3' : float(data['bill_amt3']),
            'BILL_AMT4' : float(data['bill_amt4']),
            'BILL_AMT5' : float(data['bill_amt5']),
            'BILL_AMT6' : float(data['bill_amt6']),
            'PAY_AMT1'  : float(data['pay_amt1']),
            'PAY_AMT2'  : float(data['pay_amt2']),
            'PAY_AMT3'  : float(data['pay_amt3']),
            'PAY_AMT4'  : float(data['pay_amt4']),
            'PAY_AMT5'  : float(data['pay_amt5']),
            'PAY_AMT6'  : float(data['pay_amt6']),
        }

        pay_cols   = [client_dict[f'PAY_{i}'] for i in [0,2,3,4,5,6]]
        bill_cols  = [client_dict[f'BILL_AMT{i}'] for i in range(1,7)]
        pay_a_cols = [client_dict[f'PAY_AMT{i}']  for i in range(1,7)]

        client_dict['AVG_PAY_DELAY'] = float(np.mean(pay_cols))
        client_dict['AVG_BILL_AMT']  = float(np.mean(bill_cols))
        client_dict['AVG_PAY_AMT']   = float(np.mean(pay_a_cols))
        client_dict['PAY_RATIO']     = float(client_dict['AVG_PAY_AMT'] / (client_dict['AVG_BILL_AMT'] + 1))
        client_dict['LIMIT_BAL_log'] = float(np.log(client_dict['LIMIT_BAL'] + 1))

        client_df    = pd.DataFrame([client_dict])[features]
        client_array = client_df.values[0]

        cluster_id, distances = assigner_cluster(client_array)
        gb_model = models[cluster_id]['gradient_boosting']
        nb_model = models[cluster_id]['naive_bayes']
        proba_gb = float(gb_model.predict_proba(client_df)[0][1])
        proba_nb = float(nb_model.predict_proba(client_df)[0][1])

        if proba_gb >= 0.7:
            risk_level, risk_color = "ÉLEVÉ", "red"
        elif proba_gb >= 0.35:
            risk_level, risk_color = "MODÉRÉ", "orange"
        elif proba_gb >= 0.2:
            risk_level, risk_color = "FAIBLE", "yellow"
        else:
            risk_level, risk_color = "TRÈS FAIBLE", "green"

        shap_img           = generer_shap_waterfall(client_df, cluster_id)
        shap_contributions = generer_shap_contributions(client_df, cluster_id)

        record = {
            'timestamp':  datetime.now().strftime('%d/%m/%Y %H:%M'),
            'cluster':    int(cluster_id),
            'proba_gb':   round(proba_gb * 100, 1),
            'proba_nb':   round(proba_nb * 100, 1),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'age':        client_dict['AGE'],
            'limit_bal':  client_dict['LIMIT_BAL'],
        }
        save_history(record)

        return jsonify({
            'success':            True,
            'cluster':            int(cluster_id),
            'distances':          {str(k): round(v, 0) for k, v in distances.items()},
            'proba_gb':           round(proba_gb * 100, 1),
            'proba_nb':           round(proba_nb * 100, 1),
            'risk_level':         risk_level,
            'risk_color':         risk_color,
            'shap_img':           shap_img,
            'shap_contributions': shap_contributions,
            'cluster_info':       cluster_stats[int(cluster_id)],
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/history')
def history():
    return jsonify(load_history())

if __name__ == '__main__':
    app.run(debug=True, port=5000) 
    