
import pandas as pd
import matplotlib.pyplot as plt


# 3. Analyse des performances (Temps perdu et attente)
from data_processing import analyse_et_nettoyage



def analyse_performance(filename):
    df = analyse_et_nettoyage(filename)
    print("--- Statistiques de Performance (en secondes) ---")
    stats_performance = df[['tripinfo_duration', 'tripinfo_waitingTime', 'tripinfo_timeLoss']].describe()
    #trip info waiting time = temps auquel le véhicule est a moins de 0.1m/s
    #trip info time loss = temps perdu par conducteur au delas de la vitesse qui est ideal (temps perdu dans les embouteillages/ les interesections etc )
    print(stats_performance.loc[['mean', 'min', 'max']])
    print("\n")

    # 4. Analyse par type de véhicule (vType)
    # On regarde qui pollue le plus et qui attend le plus
    analyse_vtype = df.groupby('tripinfo_vType').agg({
        'tripinfo_id': 'count',
        'tripinfo_timeLoss': 'mean',
        'emissions_CO2_abs': 'mean'
    }).rename(columns={'tripinfo_id': 'Nombre', 'tripinfo_timeLoss': 'Temps_Perdu_Moyen', 'emissions_CO2_abs': 'CO2_Moyen'})
    
    print("--- Analyse par type de véhicule ---")
    print(analyse_vtype)

    # 5. Génération de graphiques
    creer_graphiques2(df)
    creer_graphiques(df)

def creer_graphiques(df):
    # Graphique 1 : Répartition du temps perdu
    plt.figure(figsize=(10, 5))
    plt.hist(df['tripinfo_timeLoss'], bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution du Temps Perdu par les véhicules')
    plt.xlabel('Temps perdu (secondes)')
    plt.ylabel('Nombre de véhicules')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('distribution_delais.png')
    print("\nGraphique 'distribution_delais.png' sauvegardé.")

    # Graphique 2 : Pollution CO2 moyenne par type
    plt.figure(figsize=(10, 5))
    df.groupby('tripinfo_vType')['emissions_CO2_abs'].mean().sort_values().plot(kind='barh', color='salmon')
    plt.title('Émissions moyennes de CO2 par type de véhicule')
    plt.xlabel('CO2 (mg)')
    plt.ylabel('Type de véhicule')
    plt.tight_layout()
    plt.savefig('pollution_par_type.png')
    print("Graphique 'pollution_par_type.png' sauvegardé.")






def creer_graphiques2(df):
    
    # Définition des capacités moyennes (estimations pour le plateau de Saclay)
    occupancy_map = {
        'voiture_standard': 1,   # 1 personne par voiture (cas pessimiste/standard)
        'bus': 60,               # Moyenne d'un bus articulé/standard
        'pt_train': 200          # Capacité d'un train/tramway
    }
    
    # On applique le mapping pour créer une colonne d'occupation
    df['occupancy'] = df['tripinfo_vType'].map(occupancy_map)
    
    # Calcul des émissions par personne (en mg)
    df['CO2_par_passager'] = df['emissions_CO2_abs'] / df['occupancy']

    # --- GRAPHIQUE 1 : Distribution du temps perdu ---
    plt.figure(figsize=(10, 5))
    plt.hist(df['tripinfo_timeLoss'], bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution du Temps Perdu par les véhicules')
    plt.xlabel('Temps perdu (secondes)')
    plt.ylabel('Nombre de véhicules')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('distribution_delais.png')
    plt.close()

    # --- GRAPHIQUE 2 : Pollution CO2 MOYENNE PAR PERSONNE ---
    plt.figure(figsize=(10, 5))
    # On groupe par type et on prend la moyenne de la nouvelle colonne
    df.groupby('tripinfo_vType')['CO2_par_passager'].mean().sort_values(ascending=False).plot(kind='barh', color='lightgreen')
    
    plt.title('Émissions moyennes de CO2 par passager')
    plt.xlabel('CO2 par passager (mg)')
    plt.ylabel('Type de véhicule')
    plt.tight_layout()
    plt.savefig('pollution_par_passager.png')
    plt.close()
    
    print("Graphiques sauvegardés : 'distribution_delais.png' et 'pollution_par_passager.png'.")

# Pour tester avec votre fonction analyse_performance :
# def analyse_performance(filename):
#     df = pd.read_csv(filename, sep=';')
#     creer_graphiques(df)