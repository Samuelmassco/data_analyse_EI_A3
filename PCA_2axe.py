import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


from data_processing import analyse_et_nettoyage

def analyse_pca_2axe(filename):
    # 1. Appel de ta fonction de nettoyage
    df = analyse_et_nettoyage(filename)
    
    if df is None:
        print("Erreur : Impossible de charger les données.")
        return

    # 2. Création de l'indicateur de satisfaction (Ton seuil de 25%)
    # On calcule le ratio : temps perdu / durée totale
    # On utilise .get() ou on s'assure que les colonnes existent
    df['loss_ratio'] = df['tripinfo_timeLoss'] / df['tripinfo_duration']
    
    # is_unhappy = 1 si plus de 25% de temps perdu, sinon 0
    df['is_unhappy'] = (df['loss_ratio'] > 0.25).astype(int)

    # 3. Sélection des variables qui pourraient expliquer ce mécontentement
    # On exclut les colonnes de texte et les résultats (duration/timeLoss) pour ne pas fausser la PCA
    features = [
        'tripinfo_departPos', 'tripinfo_departSpeed', 'tripinfo_routeLength', 
        'tripinfo_waitingCount', 'tripinfo_stopTime', 'tripinfo_rerouteNo',
        'emissions_CO2_abs', 'emissions_fuel_abs', 'tripinfo_speedFactor'
    ]
    
    # On vérifie que ces colonnes existent bien dans le DF nettoyé
    features = [f for f in features if f in df.columns]
    X = df[features].fillna(0)

    # 4. Standardisation (Étape cruciale pour la PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Application de la PCA (2 composantes pour pouvoir l'afficher en 2D)
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(X_scaled)
    
    df['pca1'] = pca_results[:, 0]
    df['pca2'] = pca_results[:, 1]

    # 6. Visualisation
    plt.figure(figsize=(10, 7))
    
    # On colorie les points selon s'ils sont "contents" ou "pas contents"
    sns.scatterplot(
        x='pca1', y='pca2', 
        hue='is_unhappy', 
        data=df, 
        palette={0: 'skyblue', 1: 'red'}, 
        alpha=0.6
    )
    
    plt.title("Analyse PCA : Profils des trajets\n(Rouge = Mécontent : >25% de temps perdu)")
    plt.xlabel(f"Composante 1 ({pca.explained_variance_ratio_[0]:.1%} de variance)")
    plt.ylabel(f"Composante 2 ({pca.explained_variance_ratio_[1]:.1%} de variance)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Mécontent", labels=["Non ( < 25%)", "Oui ( > 25%)"])
    
    # Sauvegarde
    plt.savefig('resultat_pca_satisfaction_2axe.png')
    print("✅ Analyse PCA terminée. Graphique 'resultat_pca_satisfaction.png' généré.")

    # 7. Bonus : Afficher quelles variables influencent le plus le résultat
    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=['PC1', 'PC2'], 
        index=features
    )
    print("\nImportance des variables sur les axes PCA :")
    print(loadings.abs().sort_values(by='PC1', ascending=False).head(5))