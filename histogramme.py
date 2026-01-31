import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_processing import analyse_et_nettoyage

def creer_histogramme_retard(filename):
    # 1. Récupération des données nettoyées
    df = analyse_et_nettoyage(filename)
    
    if df is None:
        return

    # 2. Calcul du pourcentage de retard (ratio temps perdu / durée totale)
    # On multiplie par 100 pour avoir un vrai pourcentage
    df['pourcentage_retard'] = (df['tripinfo_timeLoss'] / df['tripinfo_duration']) * 100

    # 3. Création de la figure
    plt.figure(figsize=(12, 6))
    
    # Utilisation de Seaborn pour un rendu plus propre (KDE = courbe de densité)
    sns.histplot(df['pourcentage_retard'], bins=50, kde=True, color='skyblue', edgecolor='black')

    # 4. Ajout d'une ligne pour ton seuil de 25%
    plt.axvline(25, color='red', linestyle='--', linewidth=2, label='Seuil de mécontentement (25%)')

    # 5. Habillage du graphique
    plt.title("Répartition du retard des usagers (en % du temps de trajet total)")
    plt.xlabel("Pourcentage de temps perdu (%)")
    plt.ylabel("Nombre de véhicules")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    # 6. Statistiques rapides en texte sur le graphique
    moyenne = df['pourcentage_retard'].mean()
    mediane = df['pourcentage_retard'].median()
    plt.text(df['pourcentage_retard'].max()*0.7, plt.ylim()[1]*0.8, 
             f"Moyenne : {moyenne:.1f}%\nMédiane : {mediane:.1f}%", 
             bbox=dict(facecolor='white', alpha=0.5))

    # Sauvegarde
    plt.savefig('histogramme_retard_nb_personne.png')
    print(f"✅ Histogramme généré : 'histogramme_retard_nb_personne.png'")
    print(f"   Moyenne du retard : {moyenne:.2f}%")