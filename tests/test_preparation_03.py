"""Contrôles de non-fuite et d'équivalence des identifiants, sur données synthétiques.

Ces tests ne touchent jamais le jeu Kaggle : ils vérifient des propriétés de la
préparation, pas une performance prédictive.
"""
import pickle
import unittest

import numpy as np
import pandas as pd

from src.preparation_03 import Preparation03

PARAMS_TEST = dict(iterations=3, depth=2, learning_rate=.1, loss_function='RMSE',
                   random_seed=2012, thread_count=2, allow_writing_files=False,
                   verbose=False)


def sample():
    """24 ventes fictives : deux familles, ProductSize absente sur toute l'une d'elles."""
    lignes = []
    for i in range(24):
        lignes.append(dict(
            SalesID=i,
            SalePrice=10000 + i * 100,
            saledate=pd.Timestamp('2010-01-01') + pd.Timedelta(days=i),
            ProductGroup='BL' if i < 12 else 'TEX',
            ProductGroupDesc='description',
            fiBaseModel='M1' if i % 2 else 'M2',
            fiModelDesc='Model ' + str(i % 2),
            fiProductClassDesc='Machine - 10.0 to 20.0 Horsepower',
            YearMade=2000,
            MachineHoursCurrentMeter=100 + i,
            datasource=1,
            auctioneerID=1,
            ProductSize=None if i < 12 else 'Small',
            Forks=None, Ride_Control=None, Transmission=None, Coupler=None,
            Enclosure='OROPS', Hydraulics='Standard',
        ))
    return pd.DataFrame(lignes)


class TestPreparation03(unittest.TestCase):

    def setUp(self):
        self.train = sample()
        self.val = self.train.iloc[:4].copy()
        self.val.index = np.arange(100, 104)
        self.val['saledate'] = pd.Timestamp('2011-01-01')

    def test_statistiques_figees_et_independance_au_lot(self):
        """transform ne doit apprendre rien : ni des voisins, ni des nouvelles modalités."""
        prep = Preparation03('B', seuil_rare=0, min_observations=1).fit(self.train)
        etat = pickle.dumps(prep.__dict__)

        ensemble = prep.transform(self.val)
        seule = prep.transform(self.val.iloc[[0]])
        pd.testing.assert_frame_equal(ensemble.iloc[[0]], seule)

        inedit = self.val.copy()
        inedit['ProductGroup'] = 'NEW'
        inedit['fiBaseModel'] = 'NEW'
        transforme = prep.transform(inedit)
        self.assertTrue(transforme.model_frequency.eq(0).all())

        self.assertEqual(etat, pickle.dumps(prep.__dict__))

    def test_la_cible_n_entre_ni_dans_les_variables_ni_dans_les_statistiques(self):
        prep = Preparation03('A').fit(self.train)
        prix_modifies = self.train.copy()
        prix_modifies['SalePrice'] = 999999
        autre = Preparation03('A').fit(prix_modifies)

        pd.testing.assert_frame_equal(prep.transform(self.train), autre.transform(prix_modifies))
        self.assertFalse({'SalePrice', 'logSalePrice', 'saledate'} & set(prep.feature_names_))

    def test_identifiants_equivalents_et_rechargement(self):
        """1, 1.0, '1' et '01' doivent donner la même catégorie, et la même prédiction."""
        from catboost import CatBoostRegressor

        for methode in ['A', 'B']:
            prep = Preparation03(methode).fit(self.train)
            modele = CatBoostRegressor(**PARAMS_TEST)
            modele.fit(prep.transform(self.train), np.log1p(self.train.SalePrice),
                       cat_features=prep.cat_features_)
            prep, modele = pickle.loads(pickle.dumps((prep, modele)))
            reference = prep.transform(self.val)

            for valeur in [1, 1.0, '1', '1.0', '01']:
                entree = self.val.copy()
                entree['datasource'] = valeur
                entree['auctioneerID'] = valeur
                resultat = prep.transform(entree)
                pd.testing.assert_frame_equal(reference, resultat)
                np.testing.assert_array_equal(modele.predict(reference), modele.predict(resultat))

            for manquant in [None, np.nan, pd.NA]:
                entree = self.val.copy()
                entree['datasource'] = manquant
                entree['auctioneerID'] = manquant
                resultat = prep.transform(entree)
                self.assertTrue(resultat[['datasource', 'auctioneerID']].eq('Missing').all().all())

    def test_format_des_identifiants_et_valeurs_invalides(self):
        for colonne, attendu in [('datasource', '99'), ('auctioneerID', '99.0')]:
            valeurs = pd.Series([99, 99.0, '99', '99.0', None], dtype=object)
            resultat = Preparation03.normaliser_identifiant(valeurs, colonne)
            self.assertTrue(resultat.iloc[:4].eq(attendu).all())
            self.assertTrue(pd.isna(resultat.iloc[4]))

            for invalide in [1.5, float('inf'), 'invalide']:
                with self.assertRaises(ValueError):
                    Preparation03.normaliser_identifiant(pd.Series([invalide]), colonne)

    def test_age_invalide_et_division_par_zero(self):
        prep = Preparation03('A', min_observations=1).fit(self.train)
        modifie = self.val.copy()
        modifie.loc[100, 'YearMade'] = 2012  # postérieure à la vente
        modifie.loc[101, 'YearMade'] = 2011  # âge nul : ratio impossible
        resultat = prep.transform(modifie)

        self.assertEqual(resultat.loc[100, 'machine_age'], -1)
        self.assertEqual(resultat.loc[100, 'YearMade_was_nan'], 1)
        self.assertEqual(resultat.loc[101, 'heures_par_an'], -1)
        self.assertFalse(np.isinf(resultat.select_dtypes(include='number').to_numpy()).any())

    def test_la_regle_B_est_apprise_sur_l_apprentissage(self):
        a = Preparation03('A').fit(self.train).transform(self.val)
        b = Preparation03('B').fit(self.train).transform(self.val)
        self.assertTrue(a.ProductSize.eq('Missing').all())
        self.assertTrue(b.ProductSize.eq('Absent dans ce segment').all())


if __name__ == '__main__':
    unittest.main()
