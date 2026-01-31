import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data_processing import analyse_et_nettoyage

def creer_histogramme_pourcentage(filename):
    # 1. Récupération des données nettoyées
    df = analyse_et_nettoyage(filename)
    if df is None: return

    # 2. Calcul du pourcentage de retard par trajet
    # (temps perdu / durée totale) * 100
    df['pourcentage_retard'] = (df['tripinfo_timeLoss'] / df['tripinfo_duration']) * 100

    # 3. Préparation des données pour la courbe de répartition (1 - CDF)
    # On trie les retards pour calculer la distribution cumulée
    sorted_retards = np.sort(df['pourcentage_retard'])
    # y = pourcentage de véhicules ayant un retard >= à la valeur en x
    y_repartition_inverse = 100 * (1 - np.arange(len(sorted_retards)) / len(sorted_retards))

    # 4. Création du graphique
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # --- Histogramme en pourcentage de la flotte ---
    # stat='percent' transforme l'axe Y en % du total des véhicules
    sns.histplot(df['pourcentage_retard'], bins=40, color='skyblue', 
                 stat='percent', alpha=0.6, label='Densité (% de véhicules)', ax=ax1)
    
    ax1.set_ylabel("Pourcentage du nombre total de véhicules (%)", color='steelblue')
    ax1.set_xlabel("Retard subi (en % du temps de trajet)")

    # --- Courbe de répartition inverse (1 - CDF) ---
    # On crée un deuxième axe Y pour la courbe
    ax2 = ax1.twinx()
    ax2.plot(sorted_retards, y_repartition_inverse, color='red', linewidth=3, label='Cumul inverse (1 - CDF)')
    ax2.set_ylabel("Véhicules ayant AU MOINS ce retard (%)", color='red')
    ax2.set_ylim(0, 105)

    # 5. Ajout de ton seuil de 25% et annotations
    ax1.axvline(25, color='black', linestyle='--', linewidth=2, label='Seuil 25%')
    
    # Calcul du % exact de gens mécontents pour l'affichage
    nb_mecontents = (df['pourcentage_retard'] > 25).mean() * 100
    plt.text(26, 80, f"{nb_mecontents:.1f}% des usagers\ndépassent le seuil", 
             color='red', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))

    plt.title("Analyse Statistique des Retards Relatifs")
    
    # Fusion des légendes des deux axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.grid(alpha=0.2)
    plt.savefig('analyse_statistique_retards.png')
    print(f"✅ Graphique généré : 'analyse_statistique_retards.png'")
    print(f"   Proportion d'usagers mécontents (>25% de perte) : {nb_mecontents:.2f}%")