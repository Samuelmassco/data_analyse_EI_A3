import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from data_processing import analyse_et_nettoyage

def analyse_pca_5axe(filename):
    df = analyse_et_nettoyage(filename)
    if df is None: return

    # Création de l'indicateur (Ton seuil de 25%)
    df['loss_ratio'] = df['tripinfo_timeLoss'] / df['tripinfo_duration']
    df['is_unhappy'] = (df['loss_ratio'] > 0.25).astype(int)

    # Variables explicatives
    features = ['tripinfo_routeLength', 'tripinfo_waitingCount', 
                'tripinfo_stopTime', 'tripinfo_rerouteNo',
                'emissions_CO2_abs', 'tripinfo_speedFactor']
    
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(X_scaled)
    df['pca1'], df['pca2'] = pca_results[:, 0], pca_results[:, 1]

    # --- VISUALISATION AMÉLIORÉE ---
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Trace les points avec une transparence (alpha) pour voir la densité
    sns.scatterplot(x='pca1', y='pca2', hue='is_unhappy', data=df, 
                    palette={0: '#3498db', 1: '#e74c3c'}, alpha=0.4, ax=ax)

    # AJOUT DES FLÈCHES DE VARIABLES (Biplot)
    # Cela permet de voir quelle variable "pousse" les points vers le rouge
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    for i, feature in enumerate(features):
        ax.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3, 
                 color='black', alpha=0.8, head_width=0.1, head_length=0.1)
        ax.text(loadings[i, 0]*3.5, loadings[i, 1]*3.5, feature, 
                color='black', fontsize=10, fontweight='bold')

    plt.title(f"Biplot PCA : Facteurs d'insatisfaction\n(PC1: {pca.explained_variance_ratio_[0]:.1%} | PC2: {pca.explained_variance_ratio_[1]:.1%})")
    plt.xlabel("Direction des fortes émissions et longues distances (PC1)")
    plt.ylabel("Complexité du trajet / Arrêts (PC2)")
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)
    
    plt.savefig('pca_5dimension.png')
    
    # --- TABLEAU D'INTERPRÉTATION ---
    print("\n--- QUE SIGNIFIENT LES AXES ? ---")
    corr_table = pd.DataFrame(pca.components_.T, index=features, columns=['PC1', 'PC2'])
    print(corr_table)