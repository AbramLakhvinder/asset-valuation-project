"""Fonctions de préparation utilisées pas à pas par le notebook 01.

Ce sont les briques étudiées dans le notebook ; `preparation_03.Preparation03`
en est la forme industrialisée. Chaque fonction travaille sur une copie et
renvoie un nouveau DataFrame : l'original n'est jamais modifié.
"""
import numpy as np
import pandas as pd


def regrouper_categories_rares(train, test, colonne, seuil):
    """Les modalités trop peu fréquentes dans l'apprentissage deviennent "Other".

    Le comptage se fait sur l'apprentissage seul : la composition du jeu à
    prédire ne peut pas décider des catégories retenues.
    """
    train2 = train.copy()
    test2 = test.copy()
    effectifs = train2[colonne].value_counts()
    categories_gardees = effectifs[effectifs > seuil].index
    train2.loc[~train2[colonne].isin(categories_gardees), colonne] = "Other"
    test2.loc[~test2[colonne].isin(categories_gardees), colonne] = "Other"
    return train2, test2


def nettoyer_hors_bornes(dataframe, colonne, borne_min, borne_max):
    """Les valeurs hors de l'intervalle deviennent manquantes, pas supprimées.

    La borne haute peut être une série : pour YearMade, c'est l'année de vente
    de chaque ligne.
    """
    df = dataframe.copy()
    hors_bornes = (df[colonne] < borne_min) | (df[colonne] > borne_max)
    df.loc[hors_bornes, colonne] = np.nan
    return df


def zero_vers_nan(dataframe, colonne):
    """Un compteur à zéro signale une absence de relevé, pas une machine neuve."""
    df = dataframe.copy()
    df.loc[df[colonne] == 0, colonne] = np.nan
    return df


def transformation_binaire(dataframe, colonne, nouvelle_colonne):
    """Garde la trace de l'information disponible avant toute imputation."""
    df = dataframe.copy()
    df[nouvelle_colonne] = df[colonne].notna().astype(int)
    return df


def remplacer_au_dessus_seuil(dataframe, colonne, seuil):
    df = dataframe.copy()
    df.loc[df[colonne] > seuil, colonne] = np.nan
    return df


def imputation_mediane_a_nan(train, test, colonne_groupement, colonne_imputer, seuil):
    """Impute par la médiane du groupe, calculée sur l'apprentissage seul.

    Un groupe comptant moins de `seuil` valeurs observées retombe sur la
    médiane globale : une médiane sur trois observations n'est pas fiable.
    """
    train2 = train.copy()
    test2 = test.copy()
    effectifs = train2.groupby(colonne_groupement)[colonne_imputer].count()
    groupes_fiables = effectifs[effectifs >= seuil].index
    medianes = train2.groupby(colonne_groupement)[colonne_imputer].median()
    medianes_gardees = medianes[medianes.index.isin(groupes_fiables)]
    mediane_globale = train2[colonne_imputer].median()

    valeurs_train = train2[colonne_groupement].map(medianes_gardees).fillna(mediane_globale)
    valeurs_test = test2[colonne_groupement].map(medianes_gardees).fillna(mediane_globale)
    train2[colonne_imputer] = train2[colonne_imputer].fillna(valeurs_train)
    test2[colonne_imputer] = test2[colonne_imputer].fillna(valeurs_test)
    return train2, test2


def variable_nan_structurel(base, colonne_groupe, colonnes, seuil_bas=10, seuil_haut=80,
                            taille_min_groupe=385):
    """Repère les colonnes dont l'absence dépend du type de machine.

    Une colonne est retenue s'il existe au moins un groupe où elle est presque
    toujours renseignée et au moins un groupe où elle est presque toujours
    absente : l'absence ressemble alors à un équipement inexistant sur ce type
    de machine, pas à une saisie oubliée.

    Deux garde-fous :
    - `taille_min_groupe` (385, taille d'échantillon pour ±5 % à 95 %) : un taux
      calculé sur un groupe minuscule ne prouve rien ;
    - la « zone grise » : si plus de 5 % des lignes tombent dans des groupes au
      taux intermédiaire, le profil n'est pas franc et la colonne est rejetée.
    """
    variables_gardees = []

    for colonne in colonnes:
        groupes_presque_remplis = 0
        groupes_presque_vides = 0
        lignes_zone_grise = 0
        lignes_totales = 0

        taux_nan = base.groupby(colonne_groupe)[colonne].apply(lambda x: x.isnull().mean() * 100)
        tailles = base.groupby(colonne_groupe)[colonne].size()

        for modalite, pct_nan in taux_nan.items():
            taille_groupe = tailles.loc[modalite]
            lignes_totales += taille_groupe

            if seuil_bas < pct_nan < seuil_haut:
                lignes_zone_grise += taille_groupe
            elif taille_groupe >= taille_min_groupe:
                if pct_nan <= seuil_bas:
                    groupes_presque_remplis += 1
                elif pct_nan >= seuil_haut:
                    groupes_presque_vides += 1

        zone_grise_acceptable = (lignes_zone_grise / lignes_totales) < 0.05
        if groupes_presque_remplis >= 1 and groupes_presque_vides >= 1 and zone_grise_acceptable:
            variables_gardees.append(colonne)

    return variables_gardees


def imputer_valeurs_rares(dataframe, col_groupe, col_calcul, seuil_pourcentage, valeur_remplacement):
    """Dans les groupes où la colonne est presque toujours absente, l'étiqueter."""
    df = dataframe.copy()
    for colonne in col_calcul:
        taux_nan = df.groupby(col_groupe)[colonne].apply(lambda x: x.isnull().mean() * 100)
        groupes_a_changer = taux_nan[taux_nan > seuil_pourcentage].index
        a_remplacer = df[col_groupe].isin(groupes_a_changer) & df[colonne].isna()
        df.loc[a_remplacer, colonne] = valeur_remplacement
    return df


def split_X_y(trainA, testA, trainB, testB, y, col_exclue):
    """Sépare variables et cible pour les deux méthodes, en excluant le prix brut."""
    a_retirer = [y] + col_exclue
    return (trainA.drop(columns=a_retirer), trainA[y],
            testA.drop(columns=a_retirer), testA[y],
            trainB.drop(columns=a_retirer), trainB[y],
            testB.drop(columns=a_retirer), testB[y])
