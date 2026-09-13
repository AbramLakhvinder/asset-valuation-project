"""Contrôles de garde-fous temporels et de routage par segment, sur données synthétiques.

Ces tests ne touchent jamais le jeu Kaggle : ils vérifient des propriétés des
expériences (check_split, fit_pair, architecture_gain), pas une performance prédictive.
"""
import unittest

import numpy as np
import pandas as pd

from src.experiences_02 import check_split, fit_pair, architecture_gain

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


class TestExperiences02(unittest.TestCase):

    def setUp(self):
        self.train = sample()
        self.val = self.train.iloc[:4].copy()
        self.val.index = np.arange(100, 104)
        self.val['saledate'] = pd.Timestamp('2011-01-01')

    def test_garde_fous_temporels_et_2012(self):
        futur = self.val.copy()
        futur['saledate'] = pd.Timestamp('2012-01-01')
        with self.assertRaises(ValueError):
            check_split(self.train, futur)
        with self.assertRaises(ValueError):
            check_split(self.train, self.train.iloc[:4])

    def test_routage_par_segment_et_repli_sur_famille_inedite(self):
        validation = self.val.copy()
        validation.loc[100, 'ProductGroup'] = 'NEW'
        resultat = fit_pair(self.train, validation, 'A', PARAMS_TEST, min_segment=3)

        # La famille inédite garde la prédiction du global, et reste dans le score.
        self.assertEqual(resultat['scores'].iloc[1].replis, 1)
        self.assertEqual(resultat['predictions']['global'][0],
                         resultat['predictions']['segmente'][0])
        self.assertTrue(np.isfinite(resultat['predictions']['segmente']).all())
        self.assertEqual(sum(resultat['segments'].n), len(validation))
        self.assertTrue(np.isfinite(architecture_gain(resultat, n_bootstrap=10)['gain']))


if __name__ == '__main__':
    unittest.main()
