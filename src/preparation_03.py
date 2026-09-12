"""Préparation complète, ajustée exclusivement sur l'apprentissage.

Forme industrialisée des traitements étudiés pas à pas dans le notebook 01 :
toutes les statistiques sont apprises dans `fit` et seulement appliquées dans
`transform`, ce qui interdit à la période à prédire d'influencer la préparation.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.fonctions_preparation_03 import variable_nan_structurel

# Les quatre unités présentes dans fiProductClassDesc, incomparables entre elles :
# une colonne numérique par unité plutôt qu'une colonne fourre-tout.
SPEC_UNITS = {
    'Horsepower': 'spec_horsepower',
    'Metric Tons': 'spec_metric_tons',
    'Digging Depth': 'spec_digging_depth_ft',
    'Operating Capacity': 'spec_lb_operating_capacity',
}

# Colonnes toujours soumises au contrôle d'absence par famille, en plus de
# celles que le diagnostic repère de lui-même.
COLONNES_ABSENCE_B = ['ProductSize', 'Forks', 'Ride_Control', 'Transmission', 'Coupler']


class Preparation03(BaseEstimator, TransformerMixin):
    """Transformer scikit-learn : `fit` sur l'apprentissage, `transform` partout.

    Deux traitements des catégories manquantes :
    A : `Missing` pour toutes ;
    B : `Absent dans ce segment` quand l'absence dépasse 95 % dans une famille.

    Les catégories restent textuelles : CatBoost les prend telles quelles.
    """

    def __init__(self, methode='A', seuil_rare=500, min_observations=30):
        self.methode = methode
        self.seuil_rare = seuil_rare
        self.min_observations = min_observations

    @staticmethod
    def normaliser_identifiant(values, colonne):
        """Un identifiant est une catégorie, pas une quantité.

        1, 1.0, '1' et '01' doivent donner la même modalité, sans quoi une même
        source apparaîtrait sous plusieurs catégories distinctes.
        """
        if colonne not in ['datasource', 'auctioneerID']:
            raise ValueError('Identifiant non pris en charge : ' + colonne)
        nombres = pd.to_numeric(values, errors='raise')
        presents = nombres.dropna()
        if not np.isfinite(presents).all() or (presents % 1 != 0).any():
            raise ValueError(colonne + ' doit contenir des identifiants entiers ou manquants.')
        if colonne == 'datasource':
            return nombres.astype('Int64').astype('string')
        return nombres.astype('Float64').astype('string')

    @staticmethod
    def _base(raw):
        """Traitements ligne à ligne, indépendants de tout apprentissage."""
        d = raw.copy()
        dates = pd.to_datetime(d['saledate'])
        if dates.isna().any():
            raise ValueError('Dates de vente manquantes.')

        # Identifiants de ligne : aucune valeur prédictive pour une machine inédite.
        d = d.drop(columns=['SalesID', 'MachineID', 'ModelID', 'SalePrice',
                            'logSalePrice', 'saledate', 'ProductGroupDesc'],
                   errors='ignore')
        for col in ['datasource', 'auctioneerID']:
            d[col] = Preparation03.normaliser_identifiant(d[col], col)

        # Spécifications techniques : centre de l'intervalle, ou borne basse si ouverte.
        specification = d['fiProductClassDesc'].astype('string').str.split(' - ', n=1).str[1]
        bornes = specification.str.extract(r'(\d+\.?\d*)(?:\s+to\s+(\d+\.?\d*))?').astype(float)
        valeur = bornes.mean(axis=1)
        for motif, colonne in SPEC_UNITS.items():
            d[colonne] = valeur.where(specification.str.contains(motif, na=False))
        # Conserver les descriptions originales évite de perdre les classes
        # Unidentified (Compact Construction) lors de l'extraction numérique.

        # Compteur horaire : zéro et valeurs aberrantes deviennent manquants,
        # mais on garde la trace de ce qui était réellement renseigné.
        heures = pd.to_numeric(d['MachineHoursCurrentMeter'], errors='coerce')
        valide = heures.gt(0) & heures.le(40000)
        d['hours_reported'] = valide.astype(int)
        d['MachineHoursCurrentMeter'] = heures.where(valide)

        # Année de fabrication : ni avant 1950, ni après la vente.
        annee = pd.to_numeric(d['YearMade'], errors='coerce')
        d['YearMade'] = annee.where(annee.ge(1950) & annee.le(dates.dt.year))

        for partie in ['year', 'month', 'day', 'quarter']:
            d['sale_' + partie] = getattr(dates.dt, partie)

        # Âge calculé avant toute imputation : une année imputée ne doit pas
        # produire un âge présenté comme observé.
        age = d['sale_year'] - d['YearMade']
        d['machine_age'] = age
        d['heures_par_an'] = d['MachineHoursCurrentMeter'] / age.where(age.gt(0))

        d['is_recession'] = d['sale_year'].isin([2008, 2009]).astype(int)
        d['model_prefix'] = d['fiBaseModel'].astype('string').str.extract(
            r'^([A-Za-z]+)', expand=False).fillna('Numeric')
        return d

    @staticmethod
    def _enclosure(d):
        d = d.copy()
        # NO ROPS reste distinct d'OROPS : aucune équivalence métier supposée.
        d['Enclosure'] = d['Enclosure'].replace(
            {'EROPS AC': 'EROPS w AC', 'None or Unspecified': np.nan})
        return d

    def fit(self, X, y=None):
        if self.methode not in ['A', 'B'] or len(X) == 0:
            raise ValueError('Méthode A/B et apprentissage non vide requis.')
        d = self._base(X)
        self.fit_max_date_ = pd.to_datetime(X['saledate']).max()
        self.n_fit_ = len(X)

        # Méthode B : quelles colonnes sont structurellement absentes, et dans
        # quelles familles ? Appris ici, jamais recalculé dans transform.
        self.absence_groups_ = {}
        if self.methode == 'B':
            categorielles = d.select_dtypes(exclude='number').columns
            candidates = [c for c in categorielles if d[c].isna().any()]
            structurelles = variable_nan_structurel(d, 'ProductGroup', candidates, 5, 95)
            for colonne in sorted(set(structurelles) | set(COLONNES_ABSENCE_B)):
                taux = d[colonne].isna().groupby(d['ProductGroup']).mean()
                self.absence_groups_[colonne] = set(taux[taux > .95].index)

        d = self._fill_categories(d)

        self.model_counts_ = d['fiBaseModel'].value_counts().to_dict()
        effectifs = d['Hydraulics'].value_counts()
        self.hydraulics_kept_ = set(effectifs[effectifs > self.seuil_rare].index)

        self.medians_ = {}
        for colonne in ['YearMade', 'MachineHoursCurrentMeter']:
            stats = d.groupby('ProductGroup')[colonne].agg(['count', 'median'])
            fiables = stats.loc[stats['count'] >= self.min_observations, 'median']
            self.medians_[colonne] = (fiables.to_dict(), d[colonne].median())

        resultat = self._finish(d)
        self.feature_names_ = list(resultat.columns)
        self.cat_features_ = list(resultat.select_dtypes(exclude='number').columns)
        return self

    def _fill_categories(self, d):
        """Applique la règle d'absence apprise, puis complète le reste."""
        d = d.copy()
        for colonne, familles in self.absence_groups_.items():
            structurel = d[colonne].isna() & d['ProductGroup'].isin(familles)
            d.loc[structurel, colonne] = 'Absent dans ce segment'
        d = self._enclosure(d)
        for colonne in d.select_dtypes(exclude='number').columns:
            d[colonne] = d[colonne].astype('string').fillna('Missing').astype(object)
        return d

    def _finish(self, d):
        """Applique les statistiques apprises : fréquences, rares, médianes."""
        d = d.copy()
        d['model_frequency'] = d['fiBaseModel'].map(self.model_counts_).fillna(0)
        d['Hydraulics'] = d['Hydraulics'].where(
            d['Hydraulics'].isin(self.hydraulics_kept_), 'Other')

        for colonne, (medianes_par_famille, mediane_globale) in self.medians_.items():
            d[colonne + '_was_nan'] = d[colonne].isna().astype(int)
            remplacement = d['ProductGroup'].map(medianes_par_famille).fillna(mediane_globale)
            d[colonne] = d[colonne].fillna(remplacement)

        # Sentinelle plutôt qu'imputation : un âge inconnu n'est pas un âge moyen.
        derivees = list(SPEC_UNITS.values()) + ['machine_age', 'heures_par_an']
        for colonne in derivees:
            d[colonne] = d[colonne].replace([np.inf, -np.inf], np.nan).fillna(-1)
        return d

    def transform(self, X):
        if not hasattr(self, 'feature_names_'):
            raise RuntimeError('Appeler fit sur apprentissage avant transform.')
        d = self._finish(self._fill_categories(self._base(X)))
        if set(d.columns) != set(self.feature_names_):
            raise ValueError('Schéma de variables différent de celui appris.')
        return d[self.feature_names_]
