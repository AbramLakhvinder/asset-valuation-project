import pandas as pd
import numpy as np

def regrouper_categories_rares(train, test, colonne, seuil):
    train2 = train.copy()
    test2 = test.copy()
    nb_y = train2[colonne].value_counts()
    categories_gardees = nb_y[nb_y > seuil].index
    train2.loc[~(train2[colonne].isin(categories_gardees)), colonne] = "Other"
    test2.loc[~(test2[colonne].isin(categories_gardees)), colonne] = "Other"
    return train2, test2

import numpy as np

def nettoyer_hors_bornes(dataframe, colonne, borne_min, borne_max):
    mask_invalide = (dataframe[colonne] < borne_min) | (dataframe[colonne] > borne_max)
    dataframe.loc[mask_invalide, colonne] = np.nan
    return dataframe

def zero_vers_nan(dataframe, colonne): 
    dataframe.loc[dataframe[colonne]==0, colonne] = np.nan
    return dataframe

def Transformation_binaire(dataframe, colonne, nouvelleColonne):
    dataframe[nouvelleColonne]=dataframe[colonne].notna().astype(int)
    return dataframe

def remplacer_au_dessus_seuil(dataframe, colonne, seuil):
    dataframe.loc[dataframe[colonne]>seuil, colonne] = np.nan
    return dataframe

def fusionner_categories(dataframe, colonne, mapping):
    dataframe[colonne] = dataframe[colonne].replace(mapping)
    return dataframe 

def extraire_dates(df, colonne_date): 
    df2 = df.copy()
    df2["saleyear"] = df2[colonne_date].dt.year
    df2["salemonth"] = df2[colonne_date].dt.month
    df2["salequarter"] = df2[colonne_date].dt.quarter
    df2 = df2.drop(columns=[colonne_date])
    return df2

def imputation_mediane_a_nan(train, test, colonnegroupement, colonneimputer, seuil): 
    train2 = train.copy()
    test2 = test.copy()
    CountObs = train2.groupby(colonnegroupement)[colonneimputer].count() 
    CountKeep = CountObs[CountObs>=seuil].index 
    medianecalcule = train2.groupby(colonnegroupement)[colonneimputer].median()
    medianegardee = medianecalcule[medianecalcule.index.isin(CountKeep)]
    medianeglobale = train2[colonneimputer].median() 
    valeur_a_imputer_train = train2[colonnegroupement].map(medianegardee).fillna(medianeglobale)
    valeur_a_imputer_test = test2[colonnegroupement].map(medianegardee).fillna(medianeglobale)
    train2[colonneimputer] = train2[colonneimputer].fillna(valeur_a_imputer_train)
    test2[colonneimputer] = test2[colonneimputer].fillna(valeur_a_imputer_test)
    return train2, test2

def variable_nan_structurel(base, groupbycolonne, listecolonne, seuilbas=10, seuilhaut=80): 
    variablegerdees = []
    
    for i in listecolonne: 
        a = 0  
        b = 0  
        # On va compter le volume total de données dans la zone grise
        volume_ventre_mou = 0
        total_observations = 0
        
        df_nan_pct = base.groupby(groupbycolonne)[i].apply(lambda x: x.isnull().mean() * 100)
        df_total_size = base.groupby(groupbycolonne)[i].size()
        
        for modalite, pct_nan in df_nan_pct.items():
            taille_groupe = df_total_size.loc[modalite]
            total_observations += taille_groupe
            
            # Si on est dans la zone grise (indépendamment de la taille du groupe)
            if seuilbas < pct_nan < seuilhaut:
                volume_ventre_mou += taille_groupe
                
            # On vérifie les extrêmes uniquement pour les groupes représentatifs (>= 385)
            elif taille_groupe >= 385:
                if pct_nan <= seuilbas:
                    a += 1
                elif pct_nan >= seuilhaut:
                    b += 1
                    
        # RÈGLE MÉTIER : On rejette si plus de 5% du jeu de données total est dans la zone grise
        tolerance_ventre_mou = (volume_ventre_mou / total_observations) < 0.05
                        
        if (a >= 1) and (b >= 1) and tolerance_ventre_mou:
            variablegerdees.append(i)

    return variablegerdees

def imputer_valeurs_rares(dataframe, col_groupe, col_calcul, seuil_pourcentage, valeur_remplacement): 

    for i in col_calcul:
         taux_nan = dataframe.groupby(col_groupe)[i].apply(lambda x: x.isnull().mean() * 100)
         groupes_a_changer = taux_nan[taux_nan > seuil_pourcentage].index 
         mask = dataframe[col_groupe].isin(groupes_a_changer) & dataframe[i].isna()
         dataframe.loc[mask, i] = valeur_remplacement
    return dataframe
        