
import pandas as pd
import matplotlib.pyplot as plt
from analyse_perfo import analyse_performance
from data_processing import analyse_et_nettoyage
from PCA_5axe import analyse_pca_5axe
from PCA_2axe import analyse_pca_2axe
from histogramme import creer_histogramme_retard#donné brute donc vraiment combien de voiture 
from histograme_pourcentage import creer_histogramme_pourcentage #donne en pourcentage le nb de voiture en retard 


def main():

    filename = "tripinfos_2028.csv"
    analyse_performance(filename)
    analyse_pca_2axe(filename)
    analyse_pca_5axe(filename)
    creer_histogramme_pourcentage(filename)
    creer_histogramme_retard(filename)


if __name__ == "__main__":
    main()

