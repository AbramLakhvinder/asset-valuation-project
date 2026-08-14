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

def traitement_valeurs_incohérentes(dataframe, colonne):
    dataframe.loc[((dataframe[colonne]==1000)|(dataframe[colonne]==1919)|(dataframe[colonne]>2012)), colonne] = np.nan
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

