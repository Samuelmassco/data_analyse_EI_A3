import pandas as pd
import matplotlib.pyplot as plt

def analyse_et_nettoyage(filename):
    # 1. Chargement avec le bon séparateur
    try:
        data = pd.read_csv(filename, sep=';')
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{filename}' est introuvable.")
        return

    # 2. Nettoyage : On supprime les colonnes qui ne contiennent que des NaN
    # Le fichier SUMO contient beaucoup de colonnes vides par défaut (taxi, batterie, etc.)
    df = data.dropna(axis=1, how='all')
    return df
    
  
